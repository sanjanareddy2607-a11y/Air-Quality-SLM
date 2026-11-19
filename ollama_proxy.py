#!/usr/bin/env python
"""Ollama API-compatible local proxy for the Air Quality SLM Assistant.

This proxy intercepts Ollama API requests and routes them through Python
domain-validation, humidity calculation, and dataset retrieval functions,
while forwarding model-management endpoints to the original Ollama backend.

The proxy binds to 127.0.0.1 only and does not expose publicly.

Architecture:
  Ollama Desktop client  ->  THIS PROXY (127.0.0.1:11434)
                         ->  Ollama backend   (127.0.0.1:11435)

NDJSON streaming is implemented for /api/chat and /api/generate so that
Ollama Desktop's chat window displays responses correctly.
"""

import json
import os
import sys
import socket
import re
import math
import time
import datetime as dt
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# -- Project path setup -------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from domain_guard import is_greeting, is_out_of_domain, detect_intent, _has_ah_keywords
from data_retrieval import find_record, normalise_date, normalise_time
from humidity_calculator import calculate_absolute_humidity
from ollama_client import MODEL_NAME

# -- Configuration ------------------------------------------------------------
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 11434
ORIGINAL_OLLAMA_PORT = 11435

# Headers that must not be replayed from the backend (hop-by-hop / framing).
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
    "content-encoding",
}

# -- Canned responses ---------------------------------------------------------
WELCOME_MESSAGE = """Hello! Welcome to the Air Quality SLM Assistant.

I am an experimental, domain-specific Small Language Model developed using the UCI Air Quality dataset.

I can help you with two main tasks:

1. HISTORICAL AIR QUALITY DATA RETRIEVAL
   I can retrieve available measurements such as CO, NOx, NO2, benzene, temperature,
   relative humidity and absolute humidity using the date and time.

   Example:
     What was the NOx concentration on 08/12/2004 at 15:00?
     Show me CO, NOx and NO2 on 08/12/2004 at 15:00.

2. ABSOLUTE HUMIDITY CALCULATION
   I can estimate absolute humidity from temperature and relative humidity using the
   project's dataset-calibrated calculation method.

   Example:
     Calculate absolute humidity for T = 16.7 C and RH = 54.7%.
     How humid was the air at 20 C and 60% RH?

Please provide your question in one of these formats.

I work with a specific historical dataset and supported calculations. I do not provide
real-time monitoring or general-purpose knowledge."""

OUT_OF_DOMAIN_MESSAGE = (
    "Sorry, this question is outside my current capabilities. "
    "I am an experimental Air Quality SLM designed for historical air-quality measurements "
    "and supported humidity calculations. "
    "Please ask me about pollutant concentrations, sensor readings, temperature or humidity instead."
)

MISSING_INPUTS_MESSAGE = (
    "Please provide the temperature and relative humidity so I can calculate absolute humidity.\n\n"
    "For example:\n"
    "  Calculate absolute humidity for T = 16.7 C and RH = 54.7%."
)

MISSING_DATE_TIME_MESSAGE = (
    "I can retrieve historical air quality data, but I need a date and time.\n"
    "Please provide them in this format:\n"
    "  What was the NOx concentration on 08/12/2004 at 15:00?"
)

UNKNOWN_MESSAGE = (
    "I'm not sure how to help with that request.\n\n"
    "I support two types of questions:\n"
    "1. Historical data: What was the NOx concentration on 08/12/2004 at 15:00?\n"
    "2. Humidity calculation: Calculate absolute humidity for T = 20 C and RH = 60%."
)


# -- Unit mapping for pollutants ----------------------------------------------
# UCI Air Quality dataset units: CO in mg/m3, NOx in ppb, NO2/C6H6/NMHC in ug/m3.
_POLLUTANT_UNITS = {
    "CO(GT)":    "mg/m3",
    "NOx(GT)":   "ppb",
    "NO2(GT)":   "ug/m3",
    "C6H6(GT)":  "ug/m3",
    "NMHC(GT)":  "ug/m3",
}
_POLLUTANT_LABELS = {
    "CO(GT)":   "CO",
    "NOx(GT)":  "NOx",
    "NO2(GT)":  "NO2",
    "C6H6(GT)": "C6H6 (Benzene)",
    "NMHC(GT)": "NMHC",
}


def _is_nan(val) -> bool:
    try:
        return isinstance(val, float) and math.isnan(val)
    except Exception:
        return False


# -- Helper: format a retrieved record as human-readable text ------------------
def _format_record(record: dict, date_str: str, time_str: str) -> str:
    """Format a dataset record into a human-readable response string."""
    lines = [f"Air Quality Data - {date_str} at {time_str}:", ""]

    pollutants_found = []
    for col, label in _POLLUTANT_LABELS.items():
        val = record.get(col)
        if val is None or _is_nan(val):
            continue
        unit = _POLLUTANT_UNITS.get(col, "units")
        pollutants_found.append(f"  {label}: {val} {unit}")

    if pollutants_found:
        lines.append("Pollutant concentrations:")
        lines.extend(pollutants_found)
        lines.append("")

    t_val = record.get("T")
    if t_val is not None and not _is_nan(t_val):
        lines.append(f"Temperature: {t_val} C")

    rh_val = record.get("RH")
    if rh_val is not None and not _is_nan(rh_val):
        lines.append(f"Relative Humidity: {rh_val} %")

    ah_val = record.get("AH")
    if ah_val is not None and not _is_nan(ah_val):
        lines.append(f"Absolute Humidity: {ah_val} g/m3")

    return "\n".join(lines)


# -- Helper: extract numeric T and RH from a message --------------------------
def _extract_t_rh(message: str):
    """Extract temperature and relative humidity from the message."""
    t_val = None
    rh_val = None

    t_patterns = [
        r'[Tt](?:emperature)?\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)',
        r'([-+]?\d+(?:\.\d+)?)\s*(?:degrees?\s*C|deg\s*C|\u00b0\s*C|\u00b0C)',
        r'([-+]?\d+(?:\.\d+)?)\s*C\b',
    ]
    for pat in t_patterns:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            try:
                t_val = float(m.group(1))
                break
            except ValueError:
                pass

    rh_patterns = [
        r'[Rr][Hh]\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)',
        r'[Rr]elative\s+[Hh]umidity\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)',
        r'([-+]?\d+(?:\.\d+)?)\s*(?:%|percent)',
    ]
    for pat in rh_patterns:
        m = re.search(pat, message)
        if m:
            try:
                rh_val = float(m.group(1))
                break
            except ValueError:
                pass

    return t_val, rh_val


# -- Helper: extract date and time from a message -----------------------------
def _extract_date_time(message: str):
    """Extract date and time strings from a message."""
    date_str = None
    time_str = None

    dp = re.search(r'(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})', message)
    if dp:
        date_str = normalise_date(dp.group(1))

    tp = re.search(r'(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)', message)
    if tp:
        time_str = normalise_time(tp.group(1))

    return date_str, time_str


# -- Core intent handler ------------------------------------------------------
def handle_message(user_message: str) -> str:
    """Route a user message through intent detection and return a response string."""
    if not user_message or not user_message.strip():
        return WELCOME_MESSAGE

    intent = detect_intent(user_message)

    if intent == "greeting":
        return WELCOME_MESSAGE

    if intent == "out_of_domain":
        return OUT_OF_DOMAIN_MESSAGE

    if intent == "ah_calculation":
        t_val, rh_val = _extract_t_rh(user_message)
        if t_val is None or rh_val is None:
            return MISSING_INPUTS_MESSAGE
        result = calculate_absolute_humidity(t_val, rh_val)
        return (
            f"Absolute Humidity Calculation\n"
            f"  Temperature:       {t_val} C\n"
            f"  Relative Humidity: {rh_val} %\n"
            f"  Absolute Humidity: {result['result']:.4f} g/m3\n"
            f"  Method: {result['method']}\n"
            f"  Note: {result['note']}"
        )

    if intent == "retrieval":
        date_str, time_str = _extract_date_time(user_message)
        if not date_str or not time_str:
            return MISSING_DATE_TIME_MESSAGE

        record = None
        for split in ("train", "val", "test"):
            record = find_record(date_str, time_str, split)
            if record is not None:
                break

        if record is None:
            return (
                f"No data found for {date_str} at {time_str}.\n"
                "This date and time is not available in the UCI Air Quality dataset "
                "(which covers 10 March 2004 - 04 April 2005).\n"
                "Please check the date/time format: DD/MM/YYYY at HH:MM."
            )

        return _format_record(record, date_str, time_str)

    return UNKNOWN_MESSAGE


# -- NDJSON streaming helpers -------------------------------------------------
def _now_iso() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_access(kind: str, user_message: str, intent: str, response_text: str):
    """Append a one-line record of each intercepted chat/generate request.

    Used to prove that Ollama Desktop is actually routed through the proxy.
    """
    try:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        preview = (user_message or "").replace("\n", " ")[:80]
        first_line = (response_text or "").split("\n", 1)[0][:80]
        with open(_PROJECT_ROOT / "proxy_access.log", "a", encoding="utf-8") as f:
            f.write(f"{ts} | {kind} | intent={intent} | IN='{preview}' | OUT='{first_line}'\n")
    except Exception:
        pass


def _build_chat_stream_chunks(text: str) -> list[bytes]:
    chunk1 = json.dumps({
        "model": MODEL_NAME,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": text},
        "done": False
    }) + "\n"

    chunk2 = json.dumps({
        "model": MODEL_NAME,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 1,
        "eval_count": len(text.split())
    }) + "\n"
    return [chunk1.encode("utf-8"), chunk2.encode("utf-8")]


def _build_generate_stream_chunks(text: str) -> list[bytes]:
    chunk1 = json.dumps({
        "model": MODEL_NAME,
        "created_at": _now_iso(),
        "response": text,
        "done": False
    }) + "\n"

    chunk2 = json.dumps({
        "model": MODEL_NAME,
        "created_at": _now_iso(),
        "response": "",
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 1,
        "eval_count": len(text.split())
    }) + "\n"
    return [chunk1.encode("utf-8"), chunk2.encode("utf-8")]


# -- HTTP handler -------------------------------------------------------------
def normalize_path(path: str) -> str:
    if "?" in path:
        path = path[:path.index("?")]
    return path.rstrip("/") or "/"


class OllamaProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    # ---- CORS helpers ------------------------------------------------------
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    # ---- request body ------------------------------------------------------
    def _read_body(self) -> bytes:
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            return self._read_chunked()
        content_length = self.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
                if length > 0:
                    return self.rfile.read(length)
            except ValueError:
                pass
        return b""

    def _read_chunked(self) -> bytes:
        data = b""
        while True:
            size_line = self.rfile.readline().strip()
            if not size_line:
                break
            try:
                size = int(size_line.split(b";")[0], 16)
            except ValueError:
                break
            if size == 0:
                self.rfile.readline()
                break
            data += self.rfile.read(size)
            self.rfile.readline()
        return data

    # ---- responses ---------------------------------------------------------
    def _send_json(self, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self._cors_headers()
        self.send_header("X-AQ-Proxy", "1")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_ndjson(self, chunks: list[bytes]):
        body = b"".join(chunks)
        self.send_response(200)
        self._cors_headers()
        self.send_header("X-AQ-Proxy", "1")
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    # ---- forwarding --------------------------------------------------------
    def forward_to_original(self, path: str, method: str, body: bytes = b""):
        """Forward a request to the original Ollama backend, return (status, headers, body)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect((PROXY_HOST, ORIGINAL_OLLAMA_PORT))
            full_path = path
            if method.upper() == "GET":
                req = (
                    f"GET {full_path} HTTP/1.1\r\n"
                    f"Host: {PROXY_HOST}:{ORIGINAL_OLLAMA_PORT}\r\n"
                    f"Connection: close\r\n\r\n"
                )
            else:
                req = (
                    f"{method.upper()} {full_path} HTTP/1.1\r\n"
                    f"Host: {PROXY_HOST}:{ORIGINAL_OLLAMA_PORT}\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Connection: close\r\n\r\n"
                )
            sock.sendall(req.encode("utf-8"))
            if body:
                sock.sendall(body)

            raw = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                raw += chunk
            sock.close()

            sep = raw.find(b"\r\n\r\n")
            if sep == -1:
                return 200, {}, raw

            header_bytes = raw[:sep].decode("utf-8", errors="replace")
            body_bytes = raw[sep + 4:]
            header_lines = header_bytes.split("\r\n")
            try:
                status_code = int(header_lines[0].split()[1])
            except (IndexError, ValueError):
                status_code = 200

            headers = {}
            for line in header_lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip()] = v.strip()

            # De-chunk if the backend used chunked transfer encoding.
            if "chunked" in headers.get("Transfer-Encoding", "").lower():
                body_bytes = self._dechunk(body_bytes)

            return status_code, headers, body_bytes
        except Exception as e:
            return 502, {"Content-Type": "text/plain"}, f"Forward error: {e}".encode()

    @staticmethod
    def _dechunk(data: bytes) -> bytes:
        out = b""
        i = 0
        while i < len(data):
            nl = data.find(b"\r\n", i)
            if nl == -1:
                break
            try:
                size = int(data[i:nl].split(b";")[0], 16)
            except ValueError:
                break
            if size == 0:
                break
            start = nl + 2
            out += data[start:start + size]
            i = start + size + 2
        return out

    def _relay(self, status_code, headers, body_bytes):
        self.send_response(status_code)
        self._cors_headers()
        # Identifies this proxy to the watchdog and the diagnostic script.
        # Metadata only - request routing is unchanged.
        self.send_header("X-AQ-Proxy", "1")
        for k, v in headers.items():
            if k.lower() in _HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        if body_bytes:
            self.wfile.write(body_bytes)
        self.wfile.flush()

    # ---- HTTP verbs --------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = normalize_path(self.path)
        sc, headers, body = self.forward_to_original(self.path, "GET")
        self._relay(sc, headers, body)

    def do_POST(self):
        parsed = normalize_path(self.path)

        if parsed == "/api/chat":
            raw = self._read_body()
            try:
                data = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
            except json.JSONDecodeError:
                data = {}

            messages = data.get("messages", [])
            user_message = ""
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    user_message = m.get("content", "")
                    break

            stream = data.get("stream", True)
            response_text = handle_message(user_message)
            _log_access("chat", user_message, detect_intent(user_message) if user_message else "empty", response_text)

            if stream:
                self._send_ndjson(_build_chat_stream_chunks(response_text))
            else:
                self._send_json({
                    "model": MODEL_NAME,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": response_text},
                    "done": True,
                    "done_reason": "stop"
                })
            return

        if parsed == "/api/generate":
            raw = self._read_body()
            try:
                data = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
            except json.JSONDecodeError:
                data = {}

            user_message = data.get("prompt", "")
            stream = data.get("stream", True)
            response_text = handle_message(user_message)
            _log_access("generate", user_message, detect_intent(user_message) if user_message else "empty", response_text)

            if stream:
                self._send_ndjson(_build_generate_stream_chunks(response_text))
            else:
                self._send_json({
                    "model": MODEL_NAME,
                    "created_at": _now_iso(),
                    "response": response_text,
                    "done": True,
                    "done_reason": "stop"
                })
            return

        raw = self._read_body()
        sc, headers, body = self.forward_to_original(self.path, "POST", raw)
        self._relay(sc, headers, body)


# -- Server socket ------------------------------------------------------------
class ExclusiveProxyServer(ThreadingHTTPServer):
    """HTTP server that owns its port exclusively on Windows.

    Python's HTTPServer sets SO_REUSEADDR, which on Windows lets several sockets
    bind the same port. That is how duplicate proxies used to divide traffic
    unpredictably. SO_EXCLUSIVEADDRUSE makes the owning socket authoritative: a
    second listener (another proxy, or an Ollama server started by Ollama
    Desktop) cannot bind the port at all.
    """
    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        super().server_bind()


def bind_proxy_socket(host: str, port: int, exclusive_seconds: float = 8.0):
    """Bind the proxy port, retrying briefly while the OS releases it.

    A lingering TIME_WAIT socket can block the exclusive bind for a moment after
    a restart, so retry; if exclusivity is still impossible, fall back to a
    normal bind (this only happens when no other process is listening, because
    the caller has already cleared the port).
    """
    deadline = time.time() + exclusive_seconds
    while True:
        try:
            return ExclusiveProxyServer((host, port), OllamaProxyHandler)
        except OSError as exc:
            if time.time() >= deadline:
                print(f"[warn] exclusive bind on {host}:{port} failed ({exc}); "
                      f"falling back to a shared bind", flush=True)
                return ThreadingHTTPServer((host, port), OllamaProxyHandler)
            print(f"[warn] {host}:{port} not available yet ({exc}); retrying", flush=True)
            time.sleep(1.0)


# -- Entry point --------------------------------------------------------------
def main():
    global ORIGINAL_OLLAMA_PORT
    import argparse
    parser = argparse.ArgumentParser(description="Ollama API-compatible proxy for Air Quality SLM")
    parser.add_argument("--host", default=PROXY_HOST)
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    parser.add_argument("--original-port", type=int, default=ORIGINAL_OLLAMA_PORT)
    args = parser.parse_args()

    ORIGINAL_OLLAMA_PORT = args.original_port

    print("=" * 70)
    print("Air Quality SLM - Ollama API Proxy")
    print("=" * 70)
    print(f"Proxy:    http://{args.host}:{args.port}")
    print(f"Backend:  http://{args.host}:{args.original_port}")
    print(f"Model:    {MODEL_NAME}")
    print()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((args.host, args.original_port))
        sock.close()
        if result == 0:
            print(f"[OK]  Original Ollama reachable at :{args.original_port}")
        else:
            print(f"[WARN] Original Ollama NOT reachable at :{args.original_port}")
    except Exception as e:
        print(f"[WARN] Cannot check backend: {e}")

    print()
    print("Routes:")
    print("  POST /api/chat     -> Python intent handler (NDJSON streaming)")
    print("  POST /api/generate -> Python intent handler (NDJSON streaming)")
    print("  GET  /api/tags     -> forwarded to Ollama backend")
    print("  GET  /api/version  -> forwarded to Ollama backend")
    print("  *                  -> forwarded to Ollama backend")
    print()
    print("Press Ctrl+C to stop.")

    try:
        server = bind_proxy_socket(args.host, args.port)
    except OSError as exc:
        print(f"[FATAL] cannot bind {args.host}:{args.port}: {exc}", flush=True)
        print("Another process owns the port. Run start_ollama_proxy.bat to reclaim it.", flush=True)
        sys.exit(1)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping proxy...")
        server.server_close()


if __name__ == "__main__":
    main()
