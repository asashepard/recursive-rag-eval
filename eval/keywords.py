"""
keywords.py
-----------
Shared constants for relevance filtering across RAG and RLM runners.

These patterns help reduce false positives by:
1. Pre-LLM abstain gate (RAG): Skip LLM if context lacks strong obligation signals AND activity keywords
2. Post-extraction filter (RLM): Drop obligations that don't look like compliance requirements
3. Pass B validation: Stricter quote acceptance criteria
"""

import re

# ---------------------------------------------------------------------------
# ACTIVITY KEYWORDS - Map activity types to relevant keywords
# ---------------------------------------------------------------------------
ACTIVITY_KEYWORDS: dict[str, list[str]] = {
    "incident reporting": [
        "incident", "report", "notify", "notification", "cyber", "security", "breach", "event",
        "security event", "unauthorized access", "occurrence"
    ],
    "breach notification": [
        "breach", "notify", "notification", "report", "cyber", "security", "unauthorized", "compromise",
        "disclosure", "personal information"
    ],
    "response plan": [
        "plan", "procedure", "response", "incident response", "cyber", "program", "policy", 
        "governance", "testing"
    ],
    "annual certification": [
        "certif", "annual", "file", "submit", "attest", "report", "compliance"
    ],
    "event reporting": [
        "event", "report", "notify", "notification", "incident", "occurrence"
    ],
    "plan testing": [
        "test", "plan", "exercise", "drill", "simulation", "tabletop"
    ],
    "attempt notification": [
        "attempt", "notify", "notification", "report", "unsuccessful"
    ],
}

# ---------------------------------------------------------------------------
# COMPILED REGEX PATTERNS
# ---------------------------------------------------------------------------

# Modal verbs indicating obligation.
# Avoid matching bare "required" / "obligated" without an infinitive (too many false positives).
MODAL_RE = re.compile(
    r"\b(shall|must|is\s+required\s+to|required\s+to|obligated\s+to)\b",
    re.IGNORECASE,
)

# Deadline indicators.
# NOTE: avoid bare "by" (too ambiguous); only treat "by" as a deadline when followed by
# a date/time-like token (month/day, numeric date, clock time, etc.).
_MONTHS_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_BY_DATE_TIME_RE = (
    r"by\s+(?:"
    r"\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?"
    r"|\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|" + _MONTHS_RE + r"(?:\s+\d{1,2})?(?:,\s*\d{4})?"
    r")"
)

DEADLINE_RE = re.compile(
    r"\b(?:within|no\s+later\s+than|immediately|promptly)\b"
    r"|\b\d+\s*(?:hour|day|business\s+day|calendar\s+day|month|year|week)s?\b"
    r"|\b" + _BY_DATE_TIME_RE + r"\b",
    re.IGNORECASE,
)

# Action verbs for compliance obligations.
# Include common inflections to avoid missing real requirements like "reporting" / "submitted".
ACTION_RE = re.compile(
    r"\b(?:"
    r"notif(?:y|ies|ied|ying)"
    r"|report(?:s|ed|ing)?"
    r"|file|files|filed|filing"
    r"|submit|submits|submitted|submitting"
    r"|provide(?:s|d|ing)?"
    r"|send|sends|sent|sending"
    r"|deliver(?:s|ed|ing)?"
    r"|transmit|transmits|transmitted|transmitting"
    r"|disclos(?:e|es|ed|ing)"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# RECIPIENT SIGNALS - Entities that receive compliance notifications
# ---------------------------------------------------------------------------
RECIPIENT_SIGNALS = [
    "commission",
    "director", 
    "department",
    "e-isac",
    "nccic",
    "njccic",
    "psc",
    "puc",
    "puct",
    "ercot",
    "attorney general",
    "fusion center",
    "secretary",
    "administrator",
    "ferc",
    "nerc",
    "regional entity",
    "reliability coordinator",
]

# Entity word extraction for recipient validation.
TITLECASE_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")


def get_activity_keywords(activity: str) -> list[str]:
    """
    Get keywords for an activity, falling back to splitting the activity name.
    
    Args:
        activity: Activity name (e.g., "incident reporting")
    
    Returns:
        List of keywords to search for
    """
    # First try exact match
    if activity.lower() in ACTIVITY_KEYWORDS:
        return ACTIVITY_KEYWORDS[activity.lower()]
    
    # Try partial match
    for key, keywords in ACTIVITY_KEYWORDS.items():
        if key in activity.lower() or activity.lower() in key:
            return keywords
    
    # Fallback: split activity into words
    return activity.lower().split()


def has_modal_language(text: str) -> bool:
    """Check if text contains modal obligation language."""
    return MODAL_RE.search(text) is not None


def has_activity_evidence(text: str, activity: str) -> bool:
    """Check if text contains evidence of the target activity."""
    keywords = get_activity_keywords(activity)
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def looks_like_compliance_obligation(text: str) -> bool:
    """
    Check if text looks like a compliance obligation.
    
    Requires modal language AND (action verb OR deadline indicator).
    """
    text_lower = text.lower()
    if "not required to" in text_lower or "not obligated to" in text_lower:
        return False

    has_modal = MODAL_RE.search(text) is not None
    has_action = ACTION_RE.search(text) is not None
    has_deadline = DEADLINE_RE.search(text) is not None
    
    return has_modal and (has_action or has_deadline)


def is_strong_obligation(text: str) -> bool:
    """
    Check if text is a strong compliance obligation worth keeping.
    
    Returns True if text has MODAL + (ACTION or DEADLINE).
    This is the key check for deciding whether to keep an obligation
    even if activity keywords are missing.
    
    Args:
        text: Combined obligation text + citation quotes
        
    Returns:
        True if this looks like a real compliance obligation
    """
    text_lower = text.lower()
    if "not required to" in text_lower or "not obligated to" in text_lower:
        return False

    has_modal = MODAL_RE.search(text) is not None
    has_action = ACTION_RE.search(text) is not None
    has_deadline = DEADLINE_RE.search(text) is not None
    
    return has_modal and (has_action or has_deadline)


def validate_recipient_in_quote(quote: str, notify_who: str) -> bool:
    """
    Validate that a quote contains legitimate recipient evidence.
    
    Accepts if:
    1. Quote contains a known recipient signal, OR
    2. Quote contains at least one capitalized entity name
    
    Args:
        quote: The citation quote
        notify_who: The extracted notify_who value
        
    Returns:
        True if quote contains valid recipient evidence
    """
    if not quote:
        return False
    
    quote_lower = quote.lower()
    
    # Check for known recipient signals
    for signal in RECIPIENT_SIGNALS:
        if signal in quote_lower:
            return True

    # If the extracted recipient itself appears (or a strong token from it), accept.
    if notify_who:
        notify_lower = notify_who.lower()
        if notify_lower in quote_lower:
            return True
        notify_words = [w for w in re.split(r"\W+", notify_lower) if len(w) >= 4]
        if any(w in quote_lower for w in notify_words):
            return True

    # Acronyms (e.g., FERC, NJCCIC) are strong evidence.
    if ACRONYM_RE.search(quote):
        return True

    # Otherwise require multiple title-case words (avoid accepting a single generic capitalized word).
    stopwords = {
        "section",
        "appendix",
        "chapter",
        "article",
        "rule",
        "schedule",
        "table",
        "figure",
        "page",
    }
    title_words = [m.group(0) for m in TITLECASE_WORD_RE.finditer(quote)]
    meaningful = [w for w in title_words if w.lower() not in stopwords]
    if len(set(meaningful)) >= 2:
        return True
    
    return False
