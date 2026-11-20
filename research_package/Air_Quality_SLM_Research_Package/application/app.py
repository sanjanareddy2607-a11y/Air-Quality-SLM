"""app.py
Main application for the Air Quality SLM Assistant.

Provides an interactive chat interface with three response modes:
  MODE A: Greeting and capability introduction
  MODE B: Supported air-quality requests (retrieval or AH calculation)
  MODE C: Out-of-domain questions (refusal)

Uses Python lookups for numerical values (authoritative) and the Ollama SLM
only for natural-language explanation.
"""

import sys
import json
import re
from pathlib import Path

# Ensure src/ is on the path for our modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from domain_guard import is_greeting, detect_intent, is_out_of_domain, should_reset, parse_date_time
from data_retrieval import (
    get_pollutant_concentration, get_temperature, get_relative_humidity,
    get_absolute_humidity, get_all_sensor_values, example_lookup,
    normalise_date, normalise_time
)
from humidity_calculator import calculate_absolute_humidity, calculate_with_units
from ollama_client import (
    generate_response, is_model_available, verify_model_untouched,
    OllamaClient, MODEL_NAME
)


# ── Constants ───────────────────────────────────────────────────────────────

WELCOME_MESSAGE = """Hello! Welcome to the Air Quality SLM Assistant.

I am an experimental, domain-specific Small Language Model developed using the UCI Air Quality dataset.

I can help you with two main tasks:

1. HISTORICAL AIR QUALITY DATA RETRIEVAL

I can retrieve available measurements such as CO, NOx, NO2, benzene, temperature, relative humidity and
absolute humidity using the date and time.

Example:
What was the NOx concentration on 08/12/2004 at 15:00?

2. ABSOLUTE HUMIDITY CALCULATION

I can estimate absolute humidity from temperature and relative humidity using the project's dataset-calibrated
calculation method.

Example:
Calculate absolute humidity for Temperature = 16.7 C and Relative Humidity = 54.7%.

Please provide your question in one of these formats.

I work with a specific historical dataset and supported calculations. I do not provide real-time monitoring or
general-purpose knowledge."""

EXAMPLE_PROMPTS = [
    "What was the NOx concentration on 08/12/2004 at 15:00?",
    "Calculate absolute humidity for T = 16.7 C and RH = 54.7%",
    "How was the air quality on 12/03/2004 at 08.00.00?",
]

UNSUPPORTED_EXAMPLES = [
    "When is Diwali in 2004?",
    "Who won the Cricket World Cup?",
    "What is the capital of Japan?",
]

# ── Routing logic ───────────────────────────────────────────────────────────

def handle_greeting() -> str:
    """Return the welcome message for greetings."""
    return WELCOME_MESSAGE


def handle_out_of_domain() -> str:
    """Return the out-of-domain refusal message."""
    return ("I understand your question, but this is outside my current capabilities.\n"
            "I am an experimental Air Quality SLM designed for historical air-quality data retrieval and "
            "dataset-calibrated absolute-humidity calculations. "
            "My dataset does not contain the information required to answer this question. "
            "You can ask me about pollutant measurements, sensor readings, temperature, humidity or supported "
            "historical air-quality records. "
            "Please try one of my supported question formats.")


def handle_retrieval(date: str, time: str, detail_level: str = "full") -> str:
    """Handle a historical retrieval request.

    Parameters
    ----------
    date : str
        Date string
    time : str
        Time string
    detail_level : str
        "full" for all values, "pollutant" for pollutants only, "sensor" for all sensor values
    """
    # Try validation set first, then test set, then training set
    record = None
    for split in ["val", "test", "train"]:
        rec = find_record_with_retry(date, time, split)
        if rec is not None:
            record = rec
            break

    if record is None:
        return ("I'm sorry, but I couldn't find a record for that date and time in any available "
                "split of the dataset. The UCI Air Quality dataset covers April–October 2004.")

    # Build response
    co = record.get("CO(GT)")
    nox = record.get("NOx(GT)")
    no2 = record.get("NO2(GT)")
    c6h6 = record.get("C6H6(GT)")
    nmhc = record.get("NMHC(GT)")
    temp = record.get("T")
    rh = record.get("RH")
    ah = record.get("AH")

    parts = []

    if detail_level in ("full", "sensor"):
        # All sensor values
        sensor_text = _format_sensor_values(record)
        parts.append(sensor_text)

    if detail_level in ("full", "pollutant"):
        # Pollutants
        pollutant_parts = []
        if not pd_isna(co):
            pollutant_parts.append(f"CO: {co} ug/m3")
        if not pd_isna(nox):
            pollutant_parts.append(f"NOx: {nox} ug/m3")
        if not pd_isna(no2):
            pollutant_parts.append(f"NO2: {no2} ug/m3")
        if not pd_isna(c6h6):
            pollutant_parts.append(f"Benzene (C6H6): {c6h6} ug/m3")
        if not pd_isna(nmhc):
            pollutant_parts.append(f"NMHC: {nmhc} ug/m3")
        if pollutant_parts:
            parts.append("Pollutant concentrations: " + "; ".join(pollutant_parts))

    if not pd_isna(temp):
        parts.append(f"Temperature: {temp} degrees Celsius")
    if not pd_isna(rh):
        parts.append(f"Relative Humidity: {rh} %")
    if not pd_isna(ah):
        parts.append(f"Absolute Humidity: {ah} g/m3")

    if not parts:
        return "Record found but no measurable values available."

    return "\n".join(parts)


def pd_isna(val):
    """Check if a value is NaN or None, compatible with both pandas and native types."""
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _format_sensor_values(record) -> str:
    """Format all sensor/meteorological values from a record dict."""
    lines = []
    # Temperature
    temp = record.get("T")
    if temp is not None and not pd_isna(temp):
        lines.append(f"  Temperature: {temp} °C")
    # Relative Humidity
    rh = record.get("RH")
    if rh is not None and not pd_isna(rh):
        lines.append(f"  Relative Humidity: {rh} %")
    # Absolute Humidity
    ah = record.get("AH")
    if ah is not None and not pd_isna(ah):
        lines.append(f"  Absolute Humidity: {ah} g/m3")
    # Pollutants
    co = record.get("CO(GT)")
    if co is not None and not pd_isna(co):
        lines.append(f"  CO: {co} ug/m3")
    nox = record.get("NOx(GT)")
    if nox is not None and not pd_isna(nox):
        lines.append(f"  NOx: {nox} ug/m3")
    no2 = record.get("NO2(GT)")
    if no2 is not None and not pd_isna(no2):
        lines.append(f"  NO2: {no2} ug/m3")
    c6h6 = record.get("C6H6(GT)")
    if c6h6 is not None and not pd_isna(c6h6):
        lines.append(f"  Benzene (C6H6): {c6h6} ug/m3")
    nmhc = record.get("NMHC(GT)")
    if nmhc is not None and not pd_isna(nmhc):
        lines.append(f"  NMHC: {nmhc} ug/m3")
    # PT08 sensors
    for sensor in ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]:
        val = record.get(sensor)
        if val is not None and not pd_isna(val):
            lines.append(f"  {sensor}: {val}")
    if not lines:
        return "No sensor values available."
    return "Available measurements:\n" + "\n".join(lines)


def find_record_with_retry(date: str, time: str, split: str):
    """Try to find a record, trying exact and normalized matching."""
    from data_retrieval import find_record, normalise_date, normalise_time
    # Try exact first
    rec = find_record(date, time, split)
    if rec is not None:
        return rec
    # Try normalised
    norm_date = normalise_date(date)
    norm_time = normalise_time(time)
    rec = find_record(norm_date, norm_time, split)
    if rec is not None:
        return rec
    return None


def handle_ah_calculation(temp_str: str, rh_str: str) -> str:
    """Handle an absolute humidity calculation request.

    Uses the dataset-calibrated formula from humidity_calculator.py.
    """
    try:
        t_c = float(temp_str)
    except (ValueError, TypeError):
        return ("I need a valid temperature value in degrees Celsius. "
                "Please provide something like '16.7' for the temperature.")

    try:
        rh_percent = float(rh_str)
    except (ValueError, TypeError):
        return ("I need a valid relative humidity percentage. "
                "Please provide something like '54.7' for the relative humidity.")

    # Use the dataset-calibrated calculator
    result = calculate_absolute_humidity(t_c, rh_percent)
    return (f"Absolute Humidity: {result['result']:.4f} g/m3\n"
            f"Method: {result['method']}\n"
            f"Note: {result['note']}")


def parse_ah_request(message: str) -> tuple | None:
    """Parse an absolute humidity request from a message.

    Returns (temp_str, rh_str) if we can extract T and RH, or None.
    """
    m = message.strip()
    
    # Extract temperature value - handle multiple formats
    # Patterns: T=20, T = 20, 20C, 20°C, temperature 20, at 20°C, for 20°C
    # Also handle "XX°C and YY%" and "XX degrees C and YY %"
    temp_match = re.search(
        r'(?:T|temperature)\s*=\s*([\d.]+)|(?:at|for)\s+(\d+\.?\d*)\s*[°]?C|'
        r'temperature\s+([\d.]+)|(\d+\.?\d*)\s*[°]?C', m, re.I
    )
    # Extract relative humidity value - handle multiple formats
    # Patterns: RH=60, RH = 60, 60%, humidity 60, and 60%, XX% at end
    # Also handle "XX°C and YY%" format
    rh_match = re.search(
        r'[Rr][Hh]\s*=\s*([\d.]+)|(?:and|with)\s+(\d+\.?\d*)\s*[%]|'
        r'(\d+\.?\d*)\s*[%%]$|([\d.]+)\s*[%%]$|'
        r'(?:at)\s+(?:the\s+)?(\d+\.?\d*)\s*[°]?C\s+and\s+(?:the\s+)?(\d+\.?\d*)\s*[%]', m, re.I
    )

    temp_vals = [v for v in temp_match.groups() if v is not None] if temp_match else []
    rh_vals = [v for v in rh_match.groups() if v is not None] if rh_match else []

    if not temp_vals or not rh_vals:
        # Could not extract both; ask user
        return None

    # Return the first valid temperature and RH found
    temp_str = temp_vals[0]
    rh_str = rh_vals[0]

    return (temp_str, rh_str)


def _extract_date_time(message: str) -> tuple:
    """Extract date and time from a user message.

    Returns (date_str, time_str) or (None, None) if not found.
    """
    m = message.strip()

    date_str = None
    time_str = None

    # Look for date patterns like DD/MM/YYYY or DD/MM/YY
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4})',       # 08/12/2004 or 08/12/04
        r'(\d{1,2}-\d{1,2}-\d{2,4})',       # 08-12-2004
        r'(on\s+(\d{1,2}/\d{1,2}/\d{2,4}))',  # "on 08/12/2004"
    ]

    # Try to find date
    for pat in date_patterns:
        dm = re.search(pat, m)
        if dm:
            if pat == date_patterns[2]:
                # "on DD/MM/YYYY at HH:MM" case
                full = dm.group(1)
                # Try to find time after "at"
                time_search = re.search(r'at\s+(\d{1,2}[:.\d]{1,4})', m, re.I)
                if time_search:
                    time_str = time_search.group(1)
                # Normalise the date
                date_str = dm.group(2) if dm.lastindex > 1 else dm.group(0)
            else:
                date_str = dm.group(1)
            break

    # If no date found with pattern, try to find "on [date] at [time]" structure
    if not date_str:
        # Pattern: "on DD/MM/YYYY at HH:MM" or "on DD/MM/YYYY at HH.MM.SS"
        date_time_pattern = r'on\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+at\s+(\d{1,2}[:.]\d{1,4})'
        dtm = re.search(date_time_pattern, m, re.I)
        if dtm:
            date_str = dtm.group(1)
            time_str = dtm.group(2)

    # Try to find time if not already found
    if not time_str:
        # Pattern: HH:MM, HH.MM.SS, HH:MM:SS
        time_patterns = [
            r'(?:at\s+)?(\d{1,2}:\d{2}(?::\d{2})?)',
            r'(?:on\s+)?(\d{1,2}\.\d{2}(?:\.\d{2})?)',
        ]
        for pat in time_patterns:
            tm = re.search(pat, m)
            if tm:
                time_str = tm.group(1)
                break

    # Normalise formats
    if date_str:
        date_str = normalise_date(date_str)
    if time_str:
        time_str = normalise_time(time_str)

    return date_str, time_str


def handle_unknown() -> str:
    """Return a helpful message when the intent is unknown."""
    return ("I'm not sure how to help with that request. I can assist with:\n"
            "- Historical air-quality data retrieval (pollutant measurements, temperature, humidity)\n"
            "- Absolute humidity calculation from temperature and relative humidity\n"
            "- Greetings and capability introduction\n\n"
            "Please use one of my supported question formats.")


# ── Main chat loop ──────────────────────────────────────────────────────────

def chat_loop():
    """Run the interactive chat loop."""
    print("\n" + "=" * 70)
    print("Air Quality SLM Assistant - Initialising...")
    print("=" * 70 + "\n")

    # Display welcome on new conversation
    print(WELCOME_MESSAGE + "\n")

    # Check Ollama availability
    ollama_available = is_model_available()
    if ollama_available:
        print("(Ollama available)")
    else:
        print("! Ollama not available - will use Python lookups/calculations only")

    print("\n" + "-" * 70)
    print("Start typing your question below, or type 'help' for options.")
    print("-" * 70 + "\n")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        # Check for reset
        if should_reset(user_input):
            print("\nConversation reset. Welcome message displayed again.\n")
            print(WELCOME_MESSAGE + "\n")
            conversation_history = []
            continue

        # Check for help
        if user_input.lower() in ("help", "?"):
            print("\nSupported request formats:")
            print("  1. Historical retrieval:")
            print("     'What was the NOx concentration on 08/12/2004 at 15:00?'")
            print("     'Show me pollutant levels on 12/03/2004 at 08.00.00'")
            print()
            print("  2. Absolute humidity calculation:")
            print("     'Calculate absolute humidity for T = 16.7 C and RH = 54.7%'")
            print("     'How humid was the air at 20°C and 60% RH?'")
            print()
            print("  3. Type 'help' for more options")
            print()
            print("  4. Out-of-domain questions will be refused politely")
            continue

        # Empty input
        if not user_input:
            continue

        # Step 1: Greeting detection
        if is_greeting(user_input):
            print("\n" + "=" * 70)
            print(handle_greeting())
            print("=" * 70 + "\n")
            conversation_history.append(("assistant", handle_greeting()))
            continue

        # Step 2: Out-of-domain detection
        if is_out_of_domain(user_input):
            print("\n" + "=" * 70)
            print(handle_out_of_domain())
            print("=" * 70 + "\n")
            conversation_history.append(("assistant", handle_out_of_domain()))
            continue

        # Step 3: Intent detection
        intent = detect_intent(user_input)
        conversation_history.append(("user", user_input))

        if intent == "retrieval":
            # Extract date and time from the message
            date_str, time_str = _extract_date_time(user_input)
            if date_str and time_str:
                result = handle_retrieval(date_str, time_str, detail_level="full")
                print(f"\nAssistant: {result}")
            else:
                # Try to get examples
                print("\nI'd like to retrieve historical data for you, but I need a date and time.")
                print("Please provide a date (e.g., 08/12/2004) and time (e.g., 15:00 or 15.00.00).")
                # Show example
                print(f"\nExample: {EXAMPLE_PROMPTS[0]}")

        elif intent == "ah_calculation":
            # Parse for temperature and RH
            parsed = parse_ah_request(user_input)
            if parsed:
                temp_str, rh_str = parsed
                result = handle_ah_calculation(temp_str, rh_str)
                print(f"\nAssistant: {result}")
            else:
                # Ask for missing inputs
                print("\nI can calculate absolute humidity for you, but I need both temperature and")
                print("relative humidity values. Please provide something like:")
                print("  'Calculate absolute humidity for T = 16.7 C and RH = 54.7%'")
                print()
                print("Or ask me: 'How humid was the air?' and I'll prompt for the missing values.")

        elif intent == "unknown":
            print("\n" + "=" * 70)
            print(handle_unknown())
            print("=" * 70 + "\n")

        else:
            # Try as a generic retrieval; if that fails, show help
            print("\nI'm not sure how to help with that request. Let me check if it's something I can help with...")
            # Try extraction anyway
            date_str, time_str = _extract_date_time(user_input)
            if date_str and time_str:
                result = handle_retrieval(date_str, time_str, detail_level="pollutant")
                print(f"\nAssistant: {result}")
            else:
                print("\n" + "=" * 70)
                print(handle_unknown())
                print("=" * 70 + "\n")

        conversation_history.append(("assistant", ""))


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    """Entry point for the Air Quality SLM Assistant."""
    print("\n" + "=" * 70)
    print("Air Quality SLM Assistant")
    print("=" * 70 + "\n")

    # Verify model status
    try:
        model_ok = verify_model_untouched()
        if not model_ok:
            print("! Warning: Model verification inconclusive, but continuing...")
    except Exception:
        pass

    # Start chat loop
    chat_loop()


if __name__ == "__main__":
    main()