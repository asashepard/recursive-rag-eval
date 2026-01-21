#!/usr/bin/env python3
"""
build_corpus.py
---------------
Reads the download log, organizes originals by state into corpus/raw/,
and produces normalized text + metadata in corpus/normalized/.
"""

import csv
import json
import re
import shutil
import argparse
from datetime import date
from pathlib import Path
from urllib.parse import urlparse, unquote

import chardet
import fitz  # pymupdf
from bs4 import BeautifulSoup
from bs4 import FeatureNotFound
from docx import Document as DocxDocument

TODAY = date.today().isoformat()

# Mapping: substring in URL → (state_code, jurisdiction, utility_domain, source_type_override)
URL_STATE_MAP = [
    # Federal / NERC / DOE
    ("nerc.com", ("FED", "federal", "electric", None)),
    ("doe417.pnnl.gov", ("FED", "federal", "electric", None)),
    # ERCOT / Texas
    ("ercot.com", ("TX", "state", "electric", None)),
    # States (alphabetical)
    ("legis.ga.gov", ("GA", "state", "electric", None)),
    ("psc.state.md.us", ("MD", "state", "electric", None)),
    ("lara.state.mi.us", ("MI", "state", "electric", None)),
    ("energy.nh.gov", ("NH", "state", "electric", None)),
    ("nj.gov/bpu", ("NJ", "state", "electric", None)),
    ("oklahoma.gov", ("OK", "state", "electric", None)),
    ("pacodeandbulletin.gov", ("PA", "state", "electric", None)),
    ("legislature.vermont.gov", ("VT", "state", "electric", None)),
    # NARUC reference
    ("naruc.org", ("NARUC", "reference", "general", None)),
]

def detect_state(url: str):
    """Return (state, jurisdiction, utility_domain) for a URL."""
    for pattern, info in URL_STATE_MAP:
        if pattern in url:
            return info[:3]
    return ("UNK", "unknown", "unknown")


def slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug."""
    text = unquote(text)
    text = re.sub(r"[^\w\s\-]", "_", text)
    text = re.sub(r"[\s_]+", "_", text).strip("_")
    return text[:max_len]


def derive_doc_id(state: str, url: str, filename: str) -> str:
    """Build a unique doc_id like TX_ERCOT_NPRR928."""
    uri = urlparse(url)
    host_slug = slugify(uri.netloc.replace("www.", "").split(".")[0], 20).upper()
    name_slug = slugify(Path(filename).stem, 40).upper()
    return f"{state}_{host_slug}_{name_slug}"


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> str:
    """Extract text from PDF with [PAGE n] markers; skip repeated headers/footers."""
    doc = fitz.open(path)
    pages = []
    prev_first_line = ""
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        lines = text.strip().splitlines()
        # Crude header/footer removal: drop first/last line if identical to prior page
        if lines:
            if lines[0] == prev_first_line:
                lines = lines[1:]
            if lines:
                prev_first_line = lines[0]
        pages.append(f"[PAGE {i}]\n" + "\n".join(lines))
    doc.close()
    return "\n\n".join(pages)


def extract_html(path: Path) -> str:
    """Extract text from HTML preserving headings."""
    raw = path.read_bytes()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    decoded = raw.decode(enc, errors="replace")
    try:
        soup = BeautifulSoup(decoded, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(decoded, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            prefix = ""
            if el.name.startswith("h"):
                prefix = "#" * int(el.name[1]) + " "
            lines.append(prefix + txt)
    return "\n".join(lines)


def extract_docx(path: Path) -> str:
    """Extract text from DOCX."""
    doc = DocxDocument(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


def guess_source_type(filename: str, content_bytes: bytes) -> str:
    """Guess file type from extension or magic bytes."""
    ext = Path(filename).suffix.lower()
    if ext in (".pdf",):
        return "pdf"
    if ext in (".html", ".htm"):
        return "html"
    if ext in (".docx",):
        return "docx"
    # Heuristic for extensionless files
    if content_bytes[:5] == b"%PDF-":
        return "pdf"
    if content_bytes[:5] in (b"<!DOC", b"<html", b"<HTML", b"<!doc"):
        return "html"
    if content_bytes[:4] == b"PK\x03\x04":
        return "docx"
    # Fallback: treat as HTML if mostly text
    try:
        content_bytes[:2000].decode("utf-8")
        return "html"
    except Exception:
        return "binary"


def extract_text(path: Path, source_type: str) -> str:
    if source_type == "pdf":
        return extract_pdf(path)
    if source_type == "html":
        return extract_html(path)
    if source_type == "docx":
        return extract_docx(path)
    return ""


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the local corpus from downloads/_download-log.csv by extracting text and metadata."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing downloads/ and corpus/ (default: script directory)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download sources from urls.txt before building the corpus",
    )
    parser.add_argument(
        "--urls",
        type=Path,
        default=None,
        help="Path to urls.txt (default: <root>/urls.txt)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    root = args.root
    downloads = root / "downloads"
    corpus = root / "corpus"
    raw_dir = corpus / "raw"
    norm_dir = corpus / "normalized"
    doc_text_dir = norm_dir / "doc_text"
    doc_meta_dir = norm_dir / "doc_meta"
    docs_jsonl_path = norm_dir / "docs.jsonl"

    if args.download:
        urls_path = args.urls or (root / "urls.txt")
        if not urls_path.exists():
            print(f"ERROR: urls file not found: {urls_path}")
            return

        # Import downloader lazily to keep build_corpus.py usable standalone.
        try:
            from download_sources import (
                parse_urls_file,
                download_one,
                write_download_log,
                DEFAULT_USER_AGENT,
                DownloadResult,
            )
        except Exception as e:
            print(f"ERROR: failed to import downloader: {e}")
            return

        urls = parse_urls_file(urls_path)
        if not urls:
            print(f"No URLs found in {urls_path}")
            return

        print(f"Downloading {len(urls)} sources → {downloads}")
        results: list[DownloadResult] = []
        for i, url in enumerate(urls, start=1):
            print(f"[{i:02d}/{len(urls):02d}] {url}")
            results.append(
                download_one(
                    url=url,
                    downloads_dir=downloads,
                    timeout_s=60.0,
                    user_agent=DEFAULT_USER_AGENT,
                    force=False,
                    sleep_s=0.0,
                )
            )

        write_download_log(results, downloads / "_download-log.csv")
        print(f"Wrote {downloads / '_download-log.csv'}")

    # Create dirs
    for d in (raw_dir, doc_text_dir, doc_meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = downloads / "_download-log.csv"
    if not log_path.exists():
        print("No download log found. Run downloader first (python download_sources.py) or rerun with --download.")
        return

    records = []
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status") != "OK":
                continue
            url = row["Url"]
            saved = Path(row["SavedAs"])
            if not saved.exists():
                continue

            state, jurisdiction, utility_domain = detect_state(url)
            raw_bytes = saved.read_bytes()
            source_type = guess_source_type(saved.name, raw_bytes)
            doc_id = derive_doc_id(state, url, saved.name)

            # Copy original to raw/<STATE>/
            state_dir = raw_dir / state
            state_dir.mkdir(exist_ok=True)
            dest_raw = state_dir / saved.name
            if not dest_raw.exists():
                shutil.copy2(saved, dest_raw)

            # Extract text
            text = extract_text(saved, source_type)
            if not text:
                print(f"[SKIP] No text extracted: {saved.name}")
                continue

            # Write doc_text
            (doc_text_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

            # Build metadata
            meta = {
                "doc_id": doc_id,
                "state": state,
                "jurisdiction": jurisdiction,
                "utility_domain": utility_domain,
                "source_type": source_type,
                "title": Path(saved.name).stem,
                "url": url,
                "retrieved_at": TODAY,
                "version_hint": "web",
                "structure": "sections" if source_type in ("html", "docx") else "pages",
            }
            (doc_meta_dir / f"{doc_id}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            records.append(meta)
            print(f"[OK] {doc_id}")

    # Write docs.jsonl
    with open(docs_jsonl_path, "w", encoding="utf-8") as jl:
        for rec in records:
            jl.write(json.dumps(rec) + "\n")
    print(f"\nDone. {len(records)} documents indexed → {docs_jsonl_path}")


if __name__ == "__main__":
    main()
