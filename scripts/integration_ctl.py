#!/usr/bin/env python
"""integration_ctl.py - start / stop / diagnose the Air Quality SLM integration.

Called by the user-facing batch files:

    start_ollama_proxy.bat   ->  python scripts/integration_ctl.py start
    stop_ollama_proxy.bat    ->  python scripts/integration_ctl.py stop
    check_ollama_proxy.bat   ->  python scripts/integration_ctl.py status

Topology (unchanged):

    Ollama Desktop  ->  127.0.0.1:11434  (Python proxy, ollama_proxy.py)
    Python proxy    ->  127.0.0.1:11435  (real "ollama serve")
    proxy_watchdog.py supervises both.

Only processes whose command line matches this integration are ever stopped;
unrelated python.exe programs are never touched. Services are launched with
DETACHED_PROCESS so they do not depend on the launcher console.

Exit codes: start -> number of problems (0 = healthy), stop -> 0,
            status -> 0 when nothing FAILs, 1 otherwise.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# shared low-level helpers (netstat parsing, HTTP probes, detached spawning)
import proxy_watchdog as pw  # noqa: E402

PROXY_PORT = 11434
BACKEND_PORT = 11435
PROXY_SCRIPT = ROOT / "ollama_proxy.py"
WATCHDOG_SCRIPT = ROOT / "proxy_watchdog.py"
ACCESS_LOG = ROOT / "proxy_access.log"
WATCHDOG_LOG = ROOT / "watchdog.log"
RECOVERY_LOG = ROOT / "recovery.log"

CREATE_NO_WINDOW = 0x08000000


# ---------------------------------------------------------------------------
# console / logging
# ---------------------------------------------------------------------------
def say(text=""):
    print(text, flush=True)


def write_log(message):
    try:
        with open(RECOVERY_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | ctl | {message}\n")
    except OSError:
        pass


def rotate(path):
    p = Path(path)
    if not p.exists():
        return
    prev = p.with_name(p.name + ".prev")
    try:
        if prev.exists():
            prev.unlink()
        p.rename(prev)
    except OSError:
        pass


def log_tail(path, lines=10):
    p = Path(path)
    if not p.exists():
        return
    try:
        content = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    say(f"  --- last {lines} lines of {p.name} ---")
    for line in content[-lines:]:
        say("  " + line)


# ---------------------------------------------------------------------------
# Windows helpers
# ---------------------------------------------------------------------------
def run_hidden(argv, timeout=25):
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                             creationflags=CREATE_NO_WINDOW)
        return res.returncode, res.stdout, res.stderr
    except Exception as exc:
        return 1, "", str(exc)


def process_table():
    """[{pid, ppid, name, cmd}] for python/ollama processes, via one PowerShell call."""
    script = ("Get-CimInstance Win32_Process | "
              "Where-Object { $_.Name -match 'python|ollama' } | "
              "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
              "ConvertTo-Json -Compress")
    rc, out, _ = run_hidden(["powershell", "-NoProfile", "-Command", script], timeout=40)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        result.append({
            "pid": int(item.get("ProcessId") or 0),
            "ppid": int(item.get("ParentProcessId") or 0),
            "name": item.get("Name") or "",
            "cmd": item.get("CommandLine") or "",
        })
    return result


def role_of(proc):
    name = proc["name"]
    cmd = proc["cmd"]
    if name == "ollama app.exe":
        return "desktop"
    if name.lower() == "ollama.exe" and "serve" in cmd.lower():
        return "backend"
    if name.lower().startswith("python") and "ollama_proxy.py" in cmd:
        return "proxy"
    if name.lower().startswith("python") and "proxy_watchdog.py" in cmd:
        return "watchdog"
    return None


def integration_processes():
    table = []
    for proc in process_table():
        role = role_of(proc)
        if role:
            proc["role"] = role
            table.append(proc)
    return table


def name_of(pid):
    for proc in process_table():
        if proc["pid"] == int(pid):
            return proc["name"]
    return "unknown"


def port_listeners(port):
    return pw.listening_pids(port)


def established_pids(port):
    rc, out, _ = run_hidden(["netstat", "-ano", "-p", "TCP"])
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "ESTABLISHED":
            if parts[1].startswith("127.0.0.1:") and parts[2] == f"127.0.0.1:{port}":
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    pass
    return sorted(pids)


def http_ok(port, path="/api/version", timeout=4.0):
    return pw.http_status(port, path, timeout)


def wait_http(port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if http_ok(port, timeout=2.0)[0] == 200:
            return True
        time.sleep(0.5)
    return False


def desktop_exe():
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
    return str(candidate) if candidate.exists() else None


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------
def stop_integration(include_desktop=True, reason="manual"):
    # Order matters: watchdog first (so it cannot restart what we stop), then
    # Desktop (otherwise it notices the proxy is gone and starts its own server
    # on 11434), then the proxy and the backend.
    order = ["watchdog"]
    if include_desktop:
        order.append("desktop")
    order += ["proxy", "backend"]

    procs = integration_processes()
    stopped = []
    for role in order:
        targets = [p for p in procs if p["role"] == role]
        for t in targets:
            if role == "proxy" and not pw.is_proxy_process(t["pid"]):
                say(f"  skipped PID {t['pid']} (command line no longer matches the proxy)")
                continue
            if pw.kill_pid(t["pid"]):
                say(f"  stopped {role:<8} PID {t['pid']}")
                stopped.append(f"{role}/{t['pid']}")
            else:
                say(f"  could not stop {role} PID {t['pid']}")
        if role == "watchdog" and targets:
            time.sleep(0.8)       # let the supervisor die before its services
        if role == "desktop" and targets:
            time.sleep(1.2)       # Desktop may spawn its own serve while exiting

    if stopped:
        write_log(f"stop [{reason}] stopped {', '.join(stopped)}")
    else:
        say("  no integration processes were running")

    # Wait for the ports to be released, clearing a stray ollama server that
    # grabbed an integration port while we were shutting down.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        busy = sorted(set(port_listeners(PROXY_PORT)) | set(port_listeners(BACKEND_PORT)))
        if not busy:
            return True
        for owner in busy:
            name = name_of(owner).lower()
            if name.startswith("ollama"):
                say(f"  stopped stray {name} PID {owner} holding an integration port")
                pw.kill_pid(owner)
        time.sleep(0.5)
    return not port_listeners(PROXY_PORT) and not port_listeners(BACKEND_PORT)


def cmd_stop(args):
    say("=" * 76)
    say(" Stopping Air Quality SLM integration")
    say("=" * 76)
    say()
    released = stop_integration(include_desktop=not args.keep_desktop, reason="stop script")
    if not released:
        say()
        say("  WARNING: ports are still held by processes this script does not own:")
        for port in (PROXY_PORT, BACKEND_PORT):
            for owner in port_listeners(port):
                say(f"    port {port} -> PID {owner} ({name_of(owner)})")
    say()
    say(" Done. Ports 11434 and 11435 are released.")
    say()
    say(" ROLLBACK to plain Ollama (no proxy):")
    say("   1. Run this script (already done).")
    say("   2. Start Ollama Desktop. It launches its own server on 11434")
    say("      and behaves exactly as before this integration.")
    say()
    say(" To go back to the proxy setup later: start_ollama_proxy.bat")
    return 0


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def start_backend():
    rotate(ROOT / "backend.log")
    proc = pw.spawn([pw.ollama_exe(), "serve"],
                    env_extra={"OLLAMA_HOST": f"127.0.0.1:{BACKEND_PORT}"},
                    log_name="backend.log")
    return proc.pid


def start_proxy():
    rotate(ROOT / "proxy.log")
    proc = pw.spawn([sys.executable, str(PROXY_SCRIPT), "--host", "127.0.0.1",
                     "--port", str(PROXY_PORT), "--original-port", str(BACKEND_PORT)],
                    log_name="proxy.log")
    return proc.pid


def start_watchdog():
    if any(p["role"] == "watchdog" for p in integration_processes()):
        return None
    rotate(ROOT / "watchdog.boot.log")
    proc = pw.spawn([sys.executable, str(WATCHDOG_SCRIPT)], log_name="watchdog.boot.log")
    return proc.pid


def report_foreign_ports():
    say("  ERROR: ports 11434/11435 are still held by processes this script does not own:")
    for port in (PROXY_PORT, BACKEND_PORT):
        for owner in port_listeners(port):
            say(f"    port {port} -> PID {owner} ({name_of(owner)})")
    say("  Close those applications and run this again.")


def cmd_start(args):
    failures = 0
    say("=" * 76)
    say(" Starting Air Quality SLM integration")
    say("=" * 76)
    say()

    say("[1/8] Stopping any existing integration processes")
    if not stop_integration(include_desktop=True, reason="start"):
        report_foreign_ports()
        write_log("start aborted: foreign port owner on 11434/11435")
        return 1

    say("[2/8] Confirming ports 11434 and 11435 are free")
    if port_listeners(PROXY_PORT) or port_listeners(BACKEND_PORT):
        say("  ERROR: ports still busy after cleanup.")
        return 1
    say("  OK")

    # -- backend ------------------------------------------------------------
    say(f"[3/8] Starting real Ollama backend on 127.0.0.1:{BACKEND_PORT}")
    try:
        backend_pid = start_backend()
    except Exception as exc:
        say(f"  ERROR: cannot start ollama serve: {exc}")
        return 1
    say(f"  launched ollama serve, PID {backend_pid}")

    if not wait_http(BACKEND_PORT, 45):
        say("  ERROR: backend did not answer on 11435 within 45s.")
        log_tail(ROOT / "backend.log")
        write_log("start failed: backend not ready")
        return 1
    owners = port_listeners(BACKEND_PORT)
    if owners != [backend_pid]:
        say(f"  ERROR: port 11435 listener(s) are {owners}, expected only PID {backend_pid}")
        write_log("start failed: backend port owner mismatch")
        return 1
    say(f"  backend ready and owns port {BACKEND_PORT} (PID {backend_pid})")

    # -- models -------------------------------------------------------------
    say("[4/8] Checking models on the backend")
    status, body = http_ok(BACKEND_PORT, "/api/tags", timeout=5)
    if status == 200:
        try:
            names = [m.get("name", "") for m in json.loads(body).get("models", [])]
            aq = [n for n in names if n.startswith("air-quality-slm")]
            if aq:
                say(f"  OK - {len(names)} models, air-quality SLM present ({', '.join(aq)})")
            else:
                say("  WARNING: no air-quality-slm model found. Other models still work.")
        except ValueError:
            say("  WARNING: could not parse model list.")
    else:
        say("  WARNING: /api/tags did not answer.")

    # -- proxy --------------------------------------------------------------
    say(f"[5/8] Starting Python proxy on 127.0.0.1:{PROXY_PORT}")
    try:
        proxy_pid = start_proxy()
    except Exception as exc:
        say(f"  ERROR: cannot start the proxy: {exc}")
        return 1
    say(f"  launched ollama_proxy.py, PID {proxy_pid}")

    if not wait_http(PROXY_PORT, 30):
        say("  ERROR: proxy did not answer on 11434 within 30s.")
        log_tail(ROOT / "proxy.log")
        write_log("start failed: proxy not ready")
        return 1
    owners = port_listeners(PROXY_PORT)
    if owners != [proxy_pid]:
        say(f"  ERROR: port 11434 listener(s) are {owners}, expected only PID {proxy_pid}")
        write_log("start failed: proxy port owner mismatch")
        return 1
    say(f"  proxy ready and owns port {PROXY_PORT} (PID {proxy_pid})")

    # -- end to end ---------------------------------------------------------
    say("[6/8] Verifying proxy -> backend path")
    status, body = http_ok(PROXY_PORT, "/api/tags", timeout=8)
    if status == 200:
        try:
            count = len(json.loads(body).get("models", []))
            say(f"  OK - proxy reached the backend ({count} models through the proxy)")
        except ValueError:
            say("  OK - proxy reached the backend")
    else:
        say("  ERROR: proxy could not reach the backend.")
        log_tail(ROOT / "proxy.log")
        failures += 1

    # -- duplicates + watchdog ---------------------------------------------
    say("[7/8] Checking for duplicates and starting the watchdog")
    procs = integration_processes()
    proxy_count = sum(1 for p in procs if p["role"] == "proxy")
    backend_count = sum(1 for p in procs if p["role"] == "backend")
    if proxy_count != 1 or backend_count != 1:
        say(f"  ERROR: expected 1 proxy and 1 backend, found {proxy_count} proxy / {backend_count} backend")
        failures += 1
    else:
        say("  OK - exactly one proxy and one backend")

    watchdog_pid = start_watchdog()
    if watchdog_pid:
        say(f"  watchdog started, PID {watchdog_pid}")
    elif any(p["role"] == "watchdog" for p in procs):
        say("  watchdog already running")
    else:
        say("  WARNING: watchdog could not be started (integration still works, no auto-recovery).")

    # -- Desktop ------------------------------------------------------------
    say("[8/8] Launching Ollama Desktop")
    desktop = desktop_exe()
    if not desktop:
        say('  WARNING: "ollama app.exe" not found. Open Ollama Desktop manually.')
    else:
        os.startfile(desktop)  # ShellExecute: detached from this console
        deadline = time.monotonic() + 25
        connected = False
        while time.monotonic() < deadline:
            if established_pids(PROXY_PORT):
                connected = True
                break
            time.sleep(0.5)
        if connected:
            say("  OK - Ollama Desktop is connected to the proxy on 11434")
        else:
            say("  WARNING: no Desktop connection to 11434 yet. Give it a few seconds,")
            say("           or select the model and send a message to trigger the connection.")

    write_log(f"start proxy={proxy_pid} backend={backend_pid} failures={failures}")
    say()
    say("Integration is UP." if failures == 0 else "Integration started with problems - see ERROR lines above.")
    say()
    say(' In Ollama Desktop select "air-quality-slm-improved:latest" and type:')
    say("   Hey")
    say("   When is Diwali in 2004?")
    say("   What is the absolute humidity at T = 16.7 C and RH = 54.7%?")
    say("   What was the NOx concentration on 08/12/2004 at 15:00?")
    say()
    say(" Access log:  proxy_access.log")
    say(" Logs:        proxy.log  backend.log  watchdog.log  recovery.log")
    say(" Health check: check_ollama_proxy.bat")
    say(" Stop:         stop_ollama_proxy.bat")
    return failures


# ---------------------------------------------------------------------------
# status / diagnostics
# ---------------------------------------------------------------------------
def cmd_status(args):
    checks = []          # (name, status, detail, fix)

    def add(name, status, detail, fix=""):
        checks.append((name, status, detail, fix))

    say("=" * 76)
    say(" Air Quality SLM integration - health check")
    say(f" {datetime.now():%Y-%m-%d %H:%M:%S}")
    say("=" * 76)
    say()

    procs = integration_processes()
    proxies = [p for p in procs if p["role"] == "proxy"]
    backends = [p for p in procs if p["role"] == "backend"]
    watchdogs = [p for p in procs if p["role"] == "watchdog"]
    desktops = [p for p in procs if p["role"] == "desktop"]

    listen34 = port_listeners(PROXY_PORT)
    listen35 = port_listeners(BACKEND_PORT)

    # 1 backend process
    if len(backends) == 1:
        add("1. Backend process (ollama serve)", "PASS", f"PID {backends[0]['pid']}")
    elif not backends:
        add("1. Backend process (ollama serve)", "FAIL", "not running",
            "Run start_ollama_proxy.bat")
    else:
        ids = ",".join(str(p["pid"]) for p in backends)
        add("1. Backend process (ollama serve)", "WARN",
            f"{len(backends)} instances: {ids} (extra copies are transient)",
            "Run start_ollama_proxy.bat to clean up")

    # 2 proxy process
    if len(proxies) == 1:
        add("2. Proxy process (ollama_proxy.py)", "PASS", f"PID {proxies[0]['pid']}")
    elif not proxies:
        add("2. Proxy process (ollama_proxy.py)", "FAIL", "not running",
            "Run start_ollama_proxy.bat")
    else:
        ids = ",".join(str(p["pid"]) for p in proxies)
        add("2. Proxy process (ollama_proxy.py)", "FAIL",
            f"{len(proxies)} duplicate instances: {ids}",
            "Run stop_ollama_proxy.bat, then start_ollama_proxy.bat")

    # 3 watchdog
    if watchdogs:
        add("3. Watchdog (proxy_watchdog.py)", "PASS", f"PID {watchdogs[0]['pid']}")
    else:
        add("3. Watchdog (proxy_watchdog.py)", "WARN", "not running - no automatic recovery",
            "Run start_ollama_proxy.bat to start it")

    # 4 port 11434
    if len(listen34) == 1 and proxies and listen34[0] == proxies[0]["pid"]:
        add("4. Port 11434 owner", "PASS", f"1 listener, proxy PID {listen34[0]}")
    elif not listen34:
        add("4. Port 11434 owner", "FAIL", "nothing listening", "Run start_ollama_proxy.bat")
    elif len(listen34) > 1:
        add("4. Port 11434 owner", "FAIL",
            f"{len(listen34)} listeners: {','.join(map(str, listen34))} (duplicate binds)",
            "Run stop_ollama_proxy.bat, then start_ollama_proxy.bat")
    else:
        add("4. Port 11434 owner", "FAIL",
            f"held by PID {listen34[0]} ({name_of(listen34[0])})",
            "Run stop_ollama_proxy.bat, then start_ollama_proxy.bat")

    # 5 port 11435
    if len(listen35) == 1 and backends and listen35[0] == backends[0]["pid"]:
        add("5. Port 11435 owner", "PASS", f"1 listener, backend PID {listen35[0]}")
    elif not listen35:
        add("5. Port 11435 owner", "FAIL", "nothing listening", "Run start_ollama_proxy.bat")
    else:
        add("5. Port 11435 owner", "FAIL", f"listeners {','.join(map(str, listen35))}",
            "Run stop_ollama_proxy.bat, then start_ollama_proxy.bat")

    # 6 backend HTTP
    status, body = http_ok(BACKEND_PORT, timeout=4)
    if status == 200:
        add("6. Backend HTTP /api/version", "PASS", f"200 {body.strip()}")
    else:
        add("6. Backend HTTP /api/version", "FAIL",
            f"HTTP {status}" if status else "no response", "Run start_ollama_proxy.bat")

    # 7 proxy HTTP (end to end, and proof that it is really the proxy answering)
    status, headers, _ = pw.http_headers(PROXY_PORT, timeout=6)
    if status == 200 and pw.is_our_proxy(headers):
        add("7. Proxy HTTP /api/version", "PASS", "200 via proxy (proxy reached backend)")
    elif status == 200:
        add("7. Proxy HTTP /api/version", "FAIL",
            "11434 answers but not through the proxy (Desktop took the port)",
            "Run start_ollama_proxy.bat")
    elif status == 502:
        add("7. Proxy HTTP /api/version", "FAIL",
            "HTTP 502 - proxy is up but cannot reach backend", "Run start_ollama_proxy.bat")
    else:
        add("7. Proxy HTTP /api/version", "FAIL",
            f"HTTP {status}" if status else "no response", "Run start_ollama_proxy.bat")

    # 8 model list through the proxy
    status, body = http_ok(PROXY_PORT, "/api/tags", timeout=8)
    if status == 200:
        try:
            names = [m.get("name", "") for m in json.loads(body).get("models", [])]
            aq = [n for n in names if n.startswith("air-quality-slm")]
            if aq:
                add("8. Model list via proxy", "PASS", f"{len(names)} models, {', '.join(aq)}")
            else:
                add("8. Model list via proxy", "WARN",
                    f"{len(names)} models but no air-quality-slm*", "Run: ollama list")
        except ValueError:
            add("8. Model list via proxy", "WARN", "response was not valid JSON", "Check proxy.log")
    else:
        add("8. Model list via proxy", "FAIL", "no response", "Run start_ollama_proxy.bat")

    # 9 access log
    last_line = None
    if ACCESS_LOG.exists():
        try:
            lines = [ln for ln in ACCESS_LOG.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            last_line = lines[-1] if lines else None
        except OSError:
            last_line = None
    if last_line:
        stamp = last_line[:19]
        try:
            age = (datetime.now() - datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
            if age <= 30:
                add("9. Access log activity", "PASS", f"last request {int(age)} min ago")
            else:
                add("9. Access log activity", "WARN", f"last request {int(age)} min ago",
                    'Send "Hey" from Ollama Desktop to confirm routing')
        except ValueError:
            add("9. Access log activity", "WARN", "unexpected log format")
    else:
        add("9. Access log activity", "WARN", "no requests logged yet",
            'Send "Hey" from Ollama Desktop to confirm routing')

    # 10 desktop
    peers = established_pids(PROXY_PORT)
    if desktops and peers:
        add("10. Ollama Desktop connection", "PASS",
            f"Desktop PID {desktops[0]['pid']} connected to 11434")
    elif desktops:
        add("10. Ollama Desktop connection", "WARN",
            "Desktop running but no ESTABLISHED connection to 11434",
            "Send a message in Desktop; if it errors run start_ollama_proxy.bat")
    else:
        add("10. Ollama Desktop connection", "WARN", "Desktop not running", "Open Ollama Desktop")

    # 11 duplicates
    if len(proxies) <= 1 and len(listen34) <= 1 and len(listen35) <= 1:
        add("11. Duplicate processes", "PASS", "one proxy, one listener per port")
    else:
        add("11. Duplicate processes", "FAIL",
            f"proxy={len(proxies)} listeners34={len(listen34)} listeners35={len(listen35)}",
            "Run stop_ollama_proxy.bat, then start_ollama_proxy.bat")

    for name, state, detail, _fix in checks:
        say(f"  {name:<38} {state:<5} {detail}")

    failed = [c for c in checks if c[1] == "FAIL"]
    warned = [c for c in checks if c[1] == "WARN"]
    passed = [c for c in checks if c[1] == "PASS"]
    say()
    say(f"Result: {len(passed)} passed, {len(warned)} warning, {len(failed)} failed")

    if failed:
        say()
        say("Suggested recovery:")
        for name, _s, _d, fix in failed:
            if fix:
                say(f"  - {name}: {fix}")
        say("  - Full reset: stop_ollama_proxy.bat, then start_ollama_proxy.bat")
    elif warned:
        say()
        say("Warnings:")
        for name, _s, _d, fix in warned:
            if fix:
                say(f"  - {name}: {fix}")

    say()
    if WATCHDOG_LOG.exists():
        try:
            recent = WATCHDOG_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
        except OSError:
            recent = []
        if recent:
            say("Recent watchdog activity:")
            for line in recent:
                say("  " + line)
            say()
    if last_line:
        say("Last access-log entry:")
        say("  " + last_line)
        say()

    return 1 if failed else 0


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Air Quality SLM integration control")
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--keep-desktop", action="store_true",
                        help="stop services but leave Ollama Desktop open")
    args = parser.parse_args()

    if args.action == "start":
        return cmd_start(args)
    if args.action == "stop":
        return cmd_stop(args)
    return cmd_status(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # never fail silently
        print(f"FATAL: {exc}", flush=True)
        sys.exit(1)
