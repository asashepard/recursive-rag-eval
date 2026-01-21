#!/usr/bin/env python3
"""download_sources.py

Downloads the source documents listed in urls.txt into downloads/ and writes
downloads/_download-log.csv in the format consumed by build_corpus.py.

This keeps the corpus build process reproducible:
  urls.txt -> downloads/* + downloads/_download-log.csv -> build_corpus.py -> corpus/

Usage:
  python download_sources.py
  python download_sources.py --urls urls.txt --root .
  python download_sources.py --force

Notes:
- Uses only the Python standard library (urllib) to avoid extra dependencies.
- Filenames are derived from URL path and/or Content-Type.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "rlm-utilities/1.0 (+https://example.invalid)"


@dataclass(frozen=True)
class DownloadResult:
    url: str
    saved_as: Path
    status: str
    error: str | None = None


def _sanitize_filename(name: str, max_len: int = 180) -> str:
    name = unquote(name)
    name = name.strip().strip(".")
    if not name:
        name = "download"

    # Windows-forbidden characters and control chars
    name = re.sub(r"[<>:\\/*?\"|]", "_", name)
    name = re.sub(r"[\x00-\x1f]", "_", name)

    # Avoid reserved device names on Windows
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    stem, dot, suffix = name.partition(".")
    if stem.upper() in reserved:
        stem = f"_{stem}"
    name = stem + (dot + suffix if dot else "")

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    if len(name) > max_len:
        if "." in name:
            base, ext = name.rsplit(".", 1)
            base = base[: max_len - (len(ext) + 1)]
            name = f"{base}.{ext}"
        else:
            name = name[:max_len]

    return name


def parse_urls_file(path: Path) -> list[str]:
    urls: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _choose_filename(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    basename = Path(parsed.path).name
    if basename:
        name = _sanitize_filename(basename)
    else:
        # For URLs ending with '/', or empty path
        host = _sanitize_filename(parsed.netloc)
        path_slug = _sanitize_filename(parsed.path.replace("/", "_"))
        name = f"{host}{path_slug or ''}".strip("_") or "download"

    if "." not in name:
        ct = (content_type or "").split(";", 1)[0].strip().lower()
        if ct in {"text/html", "application/xhtml+xml"}:
            name = name + ".html"
        elif ct in {"application/json", "text/json"}:
            name = name + ".json"
        elif ct == "text/plain":
            name = name + ".txt"

    return name


def download_one(
    url: str,
    downloads_dir: Path,
    timeout_s: float,
    user_agent: str,
    force: bool,
    sleep_s: float,
) -> DownloadResult:
    downloads_dir.mkdir(parents=True, exist_ok=True)

    req = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "*/*",
        },
        method="GET",
    )

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            content_type = resp.headers.get("Content-Type")
            filename = _choose_filename(url, content_type)
            dest = downloads_dir / filename

            if dest.exists() and not force:
                return DownloadResult(url=url, saved_as=dest.resolve(), status="OK")

            tmp = downloads_dir / (filename + ".part")
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            os.replace(tmp, dest)

        if sleep_s:
            time.sleep(sleep_s)

        return DownloadResult(url=url, saved_as=dest.resolve(), status="OK")

    except Exception as e:
        return DownloadResult(url=url, saved_as=(downloads_dir / "").resolve(), status="ERROR", error=str(e))


def write_download_log(results: Iterable[DownloadResult], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Url", "SavedAs", "Status"])
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Url": r.url,
                    "SavedAs": str(r.saved_as),
                    "Status": r.status,
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Download corpus source files listed in urls.txt")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing urls.txt and downloads/ (default: script directory)",
    )
    parser.add_argument(
        "--urls",
        type=Path,
        default=None,
        help="Path to urls.txt (default: <root>/urls.txt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if file already exists",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between downloads in seconds (default: 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    urls_path = args.urls or (root / "urls.txt")
    downloads_dir = root / "downloads"
    log_path = downloads_dir / "_download-log.csv"

    if not urls_path.exists():
        print(f"ERROR: urls file not found: {urls_path}")
        return 2

    urls = parse_urls_file(urls_path)
    if not urls:
        print(f"No URLs found in {urls_path}")
        return 0

    print(f"Downloading {len(urls)} sources → {downloads_dir}")

    results: list[DownloadResult] = []
    ok = 0
    for i, url in enumerate(urls, start=1):
        print(f"[{i:02d}/{len(urls):02d}] {url}")
        res = download_one(
            url=url,
            downloads_dir=downloads_dir,
            timeout_s=args.timeout,
            user_agent=DEFAULT_USER_AGENT,
            force=args.force,
            sleep_s=args.sleep,
        )
        results.append(res)
        if res.status == "OK":
            ok += 1
        else:
            print(f"  [ERROR] {res.error}")

    write_download_log(results, log_path)
    print(f"\nWrote {log_path}")
    print(f"OK: {ok}/{len(urls)}")
    return 0 if ok == len(urls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
