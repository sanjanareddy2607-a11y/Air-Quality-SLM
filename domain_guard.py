"""domain_guard.py
Intelligent domain guardrails for the Air Quality SLM Assistant.

Routes user messages to one of three modes:
  MODE A: Greeting / capability introduction
  MODE B: Supported air-quality requests (historical retrieval or AH calculation)
  MODE C: Out-of-domain questions (refusal)

Key design decisions:
  - No model calls for greetings or out-of-domain questions
  - Numerical values from Python lookups are authoritative; the SLM only provides NL explanation
  - \"How humid was the air?\" is treated as a possible AH request — asks for T/RH if missing
  - Date/time lookups use exact UCI dataset matching; missing records get a clear \"unavailable\" reply
"""

import re
from datetime import datetime


# ── Greeting / help detection ─────────────────────────────────────────────

def is_greeting(message: str) -> bool:
    """Return True if the message is a greeting or capability inquiry."""
    m = message.strip().lower()
    # Greeting words at start
    if re.search(r"(?i)^\s*(hi|hello|hey)", m):
        return True
    # Standalone "what can you do" or "what can you do?"
    if re.search(r"(?i)^\s*what can (you|you do)[?!]?\s*$", m):
        return True
    # "who are you"
    if re.search(r"(?i)^\s*who (are you|is this model)[?!]?\s*$", m):
        return True
    # "how do I use you"
    if re.search(r"(?i)^\s*how (do I use|do we use)\s+you\s*$", m):
        return True
    # "what tasks can you perform"
    if re.search(r"(?i)^\s*what tasks (can you|do you) perform\s*$", m):
        return True
    # "help" standalone
    if re.search(r"(?i)^\s*help\??\s*$", m):
        return True
    return False


# ── Out-of-domain detection ───────────────────────────────────────────────

OUTOFDOMAIN_KEYWORDS = [
    "diwali", "cricket world cup", "capital of japan",
    "quantum computing", "population of bengaluru",
    "weather forecast", "real-time", "current air quality",
    "stock price", "news", "sports score", "lottery",
]


def is_out_of_domain(message: str) -> bool:
    """Return True if the message is unrelated to supported air-quality tasks."""
    m = message.strip().lower()
    # Quick keyword check
    for kw in OUTOFDOMAIN_KEYWORDS:
        if kw in m:
            return True
    # If it contains a question mark, check if it's air-quality related
    if "?" in m:
        # Air-quality terms that suggest an in-scope question
        aq_terms = ["co", "nox", "no2", "benzene", "c6h6", "temperature",
                     "absolute humidity", "pt08", "sensor", "ug/m3", "g/m3",
                     "pollution", "pollutant", "air quality", "measurement",
                     "concentration", "reading", "record",
                     "08/12/2004", "12/2004", "march 2004", "april 2004",
                     "may 2004", "june 2004", "july 2004", "august 2004",
                     "september 2004", "october 2004"]
        has_aq = any(term in m for term in aq_terms)
        # An explicit date/time is always an in-domain retrieval attempt (even if
        # the record does not exist, we want the unavailable-record message).
        if re.search(r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}", m):
            has_aq = True
        # Special case: 'humid/humidity' questions are AH requests, not OOD
        has_humid = "humid" in m or "humidity" in m
        if has_humid:
            # If 'humid/humidity' is present and there's no other AQ term suggesting OOD,
            # treat as an AH request (not OOD)
            other_aq = any(term in m for term in ["co", "nox", "no2", "benzene", "c6h6"])
            if has_aq and not other_aq:
                # Has AQ terms but only humidity - could be valid
                pass
            # If only humid/humidity without other AQ terms, NOT OOD
            elif not other_aq:
                return False  # Let intent detection handle it as AH request
        if not has_aq:
            return True
    return False


# ── Historical retrieval intent detection ──────────────────────────────────

# Keywords/phrases that suggest a historical retrieval request
RETRIEVAL_PHRASES = [
    "concentration", "level", "measurement", "value",
    "on", "at", "for", "date",
    "CO", "NOx", "NO2", "C6H6", "NMHC",
    "benzene", "temperature", "humidity",
    "absolute humidity", "show me", "get", "give me",
    "retrieve", "record", "data",
    "air quality",
    "pollutant", "sensor", "reading",
]


def _has_retrieval_phrase(message: str) -> bool:
    """Check if message contains retrieval-related phrases."""
    m = message.strip().lower()
    for phrase in RETRIEVAL_PHRASES:
        if phrase in m:
            return True
    return False


# ── Absolute humidity calculation intent detection ──────────────────────────

def _has_ah_keywords(message: str) -> bool:
    """Check if message contains absolute humidity calculation keywords."""
    m = message.strip().lower()
    # Must contain both temperature-related and humidity-related keywords
    temp_keywords = ["temperature", "°c", "celsius", "temp"]
    hum_keywords = ["humid", "humidity", "rh", "relative humidity"]

    has_temp = any(kw in m for kw in temp_keywords)
    has_hum = any(kw in m for kw in hum_keywords)

    # Special case: "how humid was the air?" without explicit T/RH
    # Still counts as AH intent - will prompt for missing values
    how_humid = re.search(r"(?i)^\s*how humid", m) is not None

    # "calculate absolute humidity for T = ... and RH = ..." pattern
    calculate_ah = re.search(r"(?i)calculate.*absolute.*humidity", m) is not None

    # Return True if we have enough indicators
    return (has_temp and has_hum) or how_humid or calculate_ah


# ── Supported intent detection ─────────────────────────────────────────────

def detect_intent(message: str) -> str:
    """Detect the user's intent category.
    Returns one of: 'greeting', 'retrieval', 'ah_calculation', 'out_of_domain', 'unknown'
    """
    m = message.strip().lower()

    # 1. Greeting
    if is_greeting(message):
        return "greeting"

    # 2. Out-of-domain
    if is_out_of_domain(message):
        return "out_of_domain"

    # 3. Absolute humidity calculation
    if _has_ah_keywords(message):
        return "ah_calculation"

    # 4. Historical retrieval (has retrieval phrases)
    if _has_retrieval_phrase(message):
        return "retrieval"

    # 5. fallback: any question mark with air-quality-like terms
    aq_terms = ["co", "nox", "no2", "benzene", "c6h6", "temperature", "humidity",
                 "absolute humidity", "pt08", "sensor", "ug/m3", "g/m3"]
    if "?" in m and any(term in m for term in aq_terms):
        return "retrieval"

    return "unknown"


# ── Conversation reset ─────────────────────────────────────────────────────

def should_reset(message: str) -> bool:
    """Return True if the user wants to reset the conversation."""
    m = message.strip().lower()
    reset_phrases = ["reset", "start over", "begin again"]
    for phrase in reset_phrases:
        if phrase in m:
            return True
    return False


# ── Historical retrieval helpers ────────────────────────────────────────────

DATASET_DATE_FMT = "%d/%m/%Y"
DATASET_TIME_FMT = "%H.%M.%S"


def parse_date_time(date_str: str, time_str: str):
    """Parse date and time strings; return (date_str, time_str) normalized or None."""
    try:
        dt = datetime.strptime(date_str.strip() + " " + time_str.strip(), "%d/%m/%Y %H.%M.%S")
        return dt.strftime(DATASET_DATE_FMT), dt.strftime(DATASET_TIME_FMT)
    except Exception:
        return None


# ── Export helpers ─────────────────────────────────────────────────────────

__all__ = [
    "is_greeting", "detect_intent", "is_out_of_domain",
    "should_reset", "parse_date_time"
]