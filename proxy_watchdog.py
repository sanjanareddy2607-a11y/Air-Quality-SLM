#!/usr/bin/env python
"""proxy_watchdog.py - supervisor for the Air Quality SLM Ollama integration.

It watches the two integration services and restarts only the component that
actually failed:

    backend : ollama serve         on 127.0.0.1:11435
    proxy   : ollama_proxy.py      on 127.0.0.1:11434

Safety rules
------------
* A process is only ever stopped if it belongs to this integration: the proxy
  (``ollama_proxy.py``), a duplicate proxy, or an Ollama server squatting on the
  integration's proxy port. Unrelated Python programs are never touched.
* An already working service is never restarted.
* A second copy of a service is never started on a port that is already held.
* Restarts are limited by a rolling budget plus a cooldown, so a permanent
  failure cannot turn into an endless crash loop.
* Every check, restart and failure is appended to ``watchdog.log``.

Port takeover
-------------
Ollama Desktop starts its own ``ollama serve`` as soon as it loses the proxy on
11434. Such a server answers HTTP 200 and would look healthy, but it bypasses all
domain routing. The proxy therefore sends ``X-AQ-Proxy: 1`` on every response and
this watchdog only accepts 11434 as healthy when that header is present. An
Ollama server squatting on the port is stopped and the proxy is restarted; any
other process on that port is reported and left alone.

Sleep / resume
--------------
Windows keeps processes alive across sleep, but sockets can be stale after a
resume. The watchdog compares wall-clock time with the monotonic clock: a large
wall-clock jump means the machine slept. After such a jump it re-verifies both
services twice and repairs whatever no longer answers.

Usage
-----
    python proxy_watchdog.py                 # supervised loop
    python proxy_watchdog.py --once          # one check pass, then exit
    python proxy_watchdog.py --simulate-resume --interval 2
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
PROXY_SCRIPT = ROOT / "ollama_proxy.py"
HOST = "127.0.0.1"
PROXY_PORT = 11434
BACKEND_PORT = 11435
PROXY_MARKER = ("x-aq-proxy", "1")

CHECK_INTERVAL = 5.0
SLEEP_GAP_SECONDS = 120.0        # wall-clock jump that means "the PC slept"
RESTART_WINDOW = 600.0           # rolling window for the restart budget
MAX_RESTARTS_PER_WINDOW = 6      # per service, per window
COOLDOWN_SECONDS = 600.0         # pause after the budget is exhausted
HEARTBEAT_SECONDS = 300.0
LOG_MAX_BYTES = 1_000_000

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
LOG_PATH = ROOT / "watchdog.log"

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# -- logging -----------------------------------------------------------------
def rotate_log():
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > LOG_MAX_BYTES:
            prev = LOG_PATH.with_suffix(".log.prev")
            if prev.exists():
                prev.unlink()
            LOG_PATH.rename(prev)
    except OSError:
        pass


def log(message):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# -- single instance ---------------------------------------------------------
def acquire_singleton():
    """Create a named mutex; returns (handle, already_running)."""
    if os.name != "nt":
        return None, False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, "Local\\AirQualitySLMProxyWatchdog")
    already = kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
    return handle, already


# -- low level probes --------------------------------------------------------
def listening_pids(port):
    """PIDs owning a LISTENING socket on 127.0.0.1:port (all binds)."""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=15,
            creationflags=CREATE_NO_WINDOW,
        ).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].endswith(f":{port}"):
                try:
                    pids.append(int(parts[4]))
                except ValueError:
                    pass
    return sorted(set(pids))


def http_headers(port, path="/api/version", timeout=3.0):
    """Return (status_or_None, lowercased_headers, body_or_error)."""
    url = f"http://{HOST}:{port}{path}"
    try:
        with _opener.open(url, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, headers, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        headers = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
        return exc.code, headers, ""
    except Exception as exc:
        return None, {}, str(exc)


def http_status(port, path="/api/version", timeout=3.0):
    """Return (status_code_or_None, body_or_error)."""
    status, _headers, body = http_headers(port, path, timeout)
    return status, body


def is_our_proxy(headers):
    return headers.get(PROXY_MARKER[0]) == PROXY_MARKER[1]


def process_name(pid):
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        ).stdout.strip()
        if out.startswith('"'):
            return out.split('","')[0].strip('"')
    except Exception:
        pass
    return "unknown"


def process_cmdline(pid):
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
            capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW,
        ).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return ""


def is_proxy_process(pid):
    return "ollama_proxy.py" in process_cmdline(pid)


def is_backend_process(pid):
    return process_name(pid).lower().startswith("ollama") and "serve" in process_cmdline(pid).lower()


def kill_pid(pid):
    try:
        res = subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/F"],
            capture_output=True, text=True, timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        return res.returncode == 0
    except Exception:
        return False


# -- launching ---------------------------------------------------------------
def ollama_exe():
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ollama")
    return found or "ollama"


def spawn(argv, env_extra=None, log_name="watchdog.boot.log"):
    """Start a detached service process; its output is appended to log_name."""
    env = os.environ.copy()
    env.pop("OLLAMA_HOST", None)
    if env_extra:
        env.update(env_extra)
    fh = open(ROOT / log_name, "a", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            argv, cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=fh, stderr=subprocess.STDOUT, env=env,
            # detached: a restarted service must not depend on this watchdog's
            # console, otherwise closing that console would kill it again
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True,
        )
    finally:
        fh.close()
    return proc


def start_backend():
    return spawn([ollama_exe(), "serve"],
                 env_extra={"OLLAMA_HOST": f"{HOST}:{BACKEND_PORT}"},
                 log_name="backend.log")


def start_proxy():
    return spawn([PYTHON, str(PROXY_SCRIPT), "--host", HOST,
                  "--port", str(PROXY_PORT), "--original-port", str(BACKEND_PORT)],
                 log_name="proxy.log")


def wait_http(port, timeout, need_marker=False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, headers, _ = http_headers(port, timeout=2.0)
        if status == 200 and (is_our_proxy(headers) or not need_marker):
            return True
        time.sleep(0.5)
    return False


# -- restart budget ----------------------------------------------------------
class Budget:
    """Rolling restart budget with a cooldown once the budget is spent."""

    def __init__(self, name, limit=MAX_RESTARTS_PER_WINDOW,
                 window=RESTART_WINDOW, cooldown=COOLDOWN_SECONDS):
        self.name = name
        self.limit = limit
        self.window = window
        self.cooldown = cooldown
        self.events = []
        self.cooldown_until = 0.0

    def allow(self):
        now = time.monotonic()
        if now < self.cooldown_until:
            return False
        self.events = [t for t in self.events if now - t < self.window]
        if len(self.events) >= self.limit:
            self.cooldown_until = now + self.cooldown
            log(f"WARN [{self.name}] restart budget exhausted "
                f"({self.limit} restarts / {int(self.window)}s) - pausing "
                f"recovery for {int(self.cooldown)}s. Manual action needed: run start_ollama_proxy.bat")
            return False
        return True

    def record(self):
        self.events.append(time.monotonic())


proxy_budget = Budget("proxy")
backend_budget = Budget("backend")


# -- recovery actions --------------------------------------------------------
def ensure_backend():
    """Return True when the backend answers on 11435."""
    pids = listening_pids(BACKEND_PORT)
    status, _ = http_status(BACKEND_PORT)
    if status == 200:
        return True

    if pids:
        foreign = [p for p in pids if not is_backend_process(p)]
        if foreign:
            log(f"CRITICAL port {BACKEND_PORT} is held by PID {foreign[0]} "
                f"({process_name(foreign[0])}), not an ollama backend - not touching it")
            return False
        if not backend_budget.allow():
            return False
        for pid in pids:
            if is_backend_process(pid):
                log(f"backend PID {pid} is not responding on :{BACKEND_PORT} - restarting it")
                kill_pid(pid)
        backend_budget.record()
        time.sleep(1.0)
    else:
        if not backend_budget.allow():
            return False
        log(f"backend is down (nothing listening on :{BACKEND_PORT}) - starting ollama serve")
        backend_budget.record()

    proc = start_backend()
    log(f"started ollama serve, PID {proc.pid} on :{BACKEND_PORT}")
    if wait_http(BACKEND_PORT, 60):
        log(f"backend recovered (PID {proc.pid})")
        return True
    log(f"ERROR backend did not become ready within 60s (PID {proc.pid}); see backend.log")
    return False


def desktop_app_pids():
    """PIDs of the Ollama Desktop app ("ollama app.exe")."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ollama app.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW,
        ).stdout
    except Exception:
        return []
    pids = []
    for row in out.splitlines():
        row = row.strip()
        if not row.startswith('"'):
            continue
        parts = row.split('","')
        if len(parts) >= 2:
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


def desktop_exe():
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
    return str(candidate) if candidate.exists() else None


def relaunch_desktop():
    exe = desktop_exe()
    if not exe:
        log("WARN cannot relaunch Ollama Desktop (app not found) - open it manually")
        return
    try:
        os.startfile(exe)
        log("Ollama Desktop relaunched; it connects to the proxy on :11434")
    except OSError as exc:
        log(f"WARN could not relaunch Ollama Desktop: {exc}")


def reclaim_port():
    """Free port 11434. Returns (port_is_free, desktop_was_restarted).

    Ollama Desktop starts its own ``ollama serve`` as soon as the proxy is gone,
    and respawns it if it is merely killed. The port can therefore only be won
    back by restarting Desktop as well, so that it reconnects through the proxy.
    """
    pids = listening_pids(PROXY_PORT)
    if not pids:
        return True, False

    squatters, stale_proxies = [], []
    for pid in pids:
        name = process_name(pid).lower()
        if name.startswith("ollama"):
            squatters.append(pid)
        elif is_proxy_process(pid):
            stale_proxies.append(pid)
        else:
            log(f"CRITICAL port {PROXY_PORT} is held by PID {pid} ({name}), which is not part "
                f"of this integration - leaving it alone. Run start_ollama_proxy.bat")
            return False, False

    if not proxy_budget.allow():
        return False, False
    proxy_budget.record()

    desktop_restarted = False
    if squatters and desktop_app_pids():
        log(f"Ollama Desktop started its own server on :{PROXY_PORT} after the proxy stopped "
            f"(PID {squatters}). Restarting Desktop so it reconnects through the proxy.")
        for pid in desktop_app_pids():
            kill_pid(pid)
        desktop_restarted = True
        time.sleep(1.5)

    for pid in squatters:
        log(f"stopping Ollama server PID {pid} squatting on :{PROXY_PORT}")
        kill_pid(pid)
    for pid in stale_proxies:
        log(f"stopping stale proxy PID {pid} on :{PROXY_PORT}")
        kill_pid(pid)

    # a newly spawned Desktop server may appear while we clean up
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        remaining = listening_pids(PROXY_PORT)
        if not remaining:
            return True, desktop_restarted
        for pid in remaining:
            if process_name(pid).lower().startswith("ollama"):
                kill_pid(pid)
        time.sleep(0.4)
    return not listening_pids(PROXY_PORT), desktop_restarted


def ensure_proxy():
    """Return True when the proxy owns 11434 and forwards to the backend."""
    status, headers, _ = http_headers(PROXY_PORT)

    if status == 200 and is_our_proxy(headers):
        return True
    if status == 502 and is_our_proxy(headers):
        # the proxy is fine, its backend call failed - ensure_backend handles it
        return False

    free, desktop_restarted = reclaim_port()
    if not free:
        return False

    if http_status(BACKEND_PORT)[0] != 200:
        if not ensure_backend():
            log("cannot start the proxy: backend is not available")
            if desktop_restarted:
                relaunch_desktop()
            return False

    proc = start_proxy()
    log(f"started ollama_proxy.py, PID {proc.pid} on :{PROXY_PORT}")
    ok = False
    if wait_http(PROXY_PORT, 30, need_marker=True):
        status, headers, _ = http_headers(PROXY_PORT, "/api/tags", timeout=6)
        if status == 200 and is_our_proxy(headers):
            log(f"proxy recovered (PID {proc.pid}) and is forwarding to the backend")
            ok = True
        else:
            log(f"WARN proxy PID {proc.pid} is listening but /api/tags returned {status}")
    else:
        log(f"ERROR proxy did not become ready with the integration marker within 30s "
            f"(PID {proc.pid}); see proxy.log")

    if desktop_restarted:
        relaunch_desktop()
    return ok


def kill_duplicate_proxies():
    """Remove duplicate proxy listeners on 11434 (keeps nothing, restarts one)."""
    pids = listening_pids(PROXY_PORT)
    if len(pids) <= 1:
        return False
    if not all(is_proxy_process(p) for p in pids):
        return False
    log(f"WARN {len(pids)} proxy listeners on :{PROXY_PORT} ({pids}) - consolidating to one")
    for pid in pids:
        kill_pid(pid)
    time.sleep(1.0)
    return True


# -- main loop ---------------------------------------------------------------
def one_pass(state):
    """Run a single health/recovery pass. Returns True when both services are healthy."""
    if kill_duplicate_proxies():
        state["dirty"] = True

    backend_ok = http_status(BACKEND_PORT)[0] == 200
    status, headers, _ = http_headers(PROXY_PORT)
    proxy_ok = status == 200 and is_our_proxy(headers)

    if backend_ok and proxy_ok:
        return True

    if not backend_ok:
        backend_ok = ensure_backend()

    if not proxy_ok:
        proxy_ok = ensure_proxy()

    return backend_ok and proxy_ok


def main():
    parser = argparse.ArgumentParser(description="Watchdog for the Air Quality SLM Ollama integration")
    parser.add_argument("--interval", type=float, default=CHECK_INTERVAL)
    parser.add_argument("--once", action="store_true", help="run one check pass and exit")
    parser.add_argument("--simulate-resume", action="store_true",
                        help="pretend a sleep/resume just happened, to test the resume path")
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS_PER_WINDOW,
                        help="restart budget per service per window (default %(default)s)")
    parser.add_argument("--restart-window", type=float, default=RESTART_WINDOW,
                        help="rolling window for the restart budget, seconds (default %(default)s)")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SECONDS,
                        help="pause after the budget is exhausted, seconds (default %(default)s)")
    args = parser.parse_args()

    for budget in (proxy_budget, backend_budget):
        budget.limit = args.max_restarts
        budget.window = args.restart_window
        budget.cooldown = args.cooldown

    handle, already = acquire_singleton()
    if already and not args.once:
        print("watchdog already running - exiting", flush=True)
        return 0

    if not args.once:
        rotate_log()
        log(f"watchdog started (PID {os.getpid()}, interval {args.interval}s)")

    state = {"dirty": False, "last_wall": time.time(), "last_mono": time.monotonic()}
    if args.simulate_resume:
        # make the first loop iteration look like it woke from a 5 minute sleep
        state["last_wall"] = time.time() - 300.0
        log("simulated sleep/resume gap injected (test mode)")

    if args.once:
        ok = one_pass(state)
        print("healthy" if ok else "unhealthy", flush=True)
        return 0 if ok else 1

    last_heartbeat = 0.0
    consecutive_failures = 0

    while True:
        try:
            now_wall = time.time()
            now_mono = time.monotonic()
            wall_gap = now_wall - state["last_wall"]
            mono_gap = now_mono - state["last_mono"]
            state["last_wall"] = now_wall
            state["last_mono"] = now_mono

            resumed = False
            if mono_gap > 0 and (wall_gap - mono_gap) > SLEEP_GAP_SECONDS:
                resumed = True
                log(f"SLEEP/RESUME detected: wall-clock gap {wall_gap:.0f}s "
                    f"vs active loop {mono_gap:.0f}s - re-verifying services")

            healthy = one_pass(state)

            if resumed:
                if healthy:
                    time.sleep(2.0)
                    if one_pass(state):
                        log("post-resume check OK: proxy and backend are healthy")
                    else:
                        log("post-resume check FAILED twice - recovery attempted, see log above")
                else:
                    log("post-resume recovery failed - run start_ollama_proxy.bat")

            if healthy:
                state["dirty"] = False
                consecutive_failures = 0
                if now_mono - last_heartbeat > HEARTBEAT_SECONDS:
                    last_heartbeat = now_mono
                    log("heartbeat: proxy and backend healthy")
            else:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 12 == 0:
                    log(f"unhealthy (consecutive checks: {consecutive_failures})")

        except Exception as exc:  # the watchdog must never die on a transient error
            log(f"watchdog loop error: {exc}")

        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("watchdog stopped by user")
        sys.exit(0)
