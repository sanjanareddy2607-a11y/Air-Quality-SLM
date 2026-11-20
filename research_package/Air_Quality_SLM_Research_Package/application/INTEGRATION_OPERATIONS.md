# Air Quality SLM - Ollama Desktop integration: operations guide

## Architecture (unchanged)

```
Ollama Desktop  ->  127.0.0.1:11434   Python proxy  (ollama_proxy.py)
Python proxy    ->  127.0.0.1:11435   real backend  (ollama serve)
proxy_watchdog.py supervises both
```

Everything binds to localhost only. Model weights, tokenizer files, dataset files
and the humidity/retrieval logic are untouched.

## Files

| File | Purpose |
| --- | --- |
| `start_ollama_proxy.bat` | Start everything (backend -> proxy -> watchdog -> Desktop) |
| `stop_ollama_proxy.bat` | Stop everything (watchdog -> Desktop -> proxy -> backend) |
| `check_ollama_proxy.bat` | Read-only health check, PASS/WARN/FAIL + suggested fixes |
| `scripts/integration_ctl.py` | Implementation behind the three batch files |
| `proxy_watchdog.py` | Automatic failure recovery |
| `ollama_proxy.py` | The proxy itself (unchanged behaviour, plus an `X-AQ-Proxy: 1` response header used for identity checks) |

Logs: `proxy_access.log` (every intercepted chat), `proxy.log`, `backend.log`,
`watchdog.log` (checks and recoveries), `recovery.log` (start/stop history).

## Daily use

1. Double-click `start_ollama_proxy.bat`.
2. In Ollama Desktop pick `air-quality-slm-improved:latest` and type, e.g. `Hey`.
3. When finished, `stop_ollama_proxy.bat`.

The start script is idempotent: running it while the integration is already up
stops the old processes and brings up exactly one proxy and one backend.

`start_ollama_proxy.bat /quiet` runs without the final pause (used by Task Scheduler).
`stop_ollama_proxy.bat /keepdesktop` stops the services but leaves Desktop open.

## What the start script checks

1. Stops only processes belonging to this integration (matched by command line):
   the watchdog, `ollama app.exe`, `ollama_proxy.py`, `ollama.exe serve`.
   Unrelated Python programs are never touched.
2. Confirms 11434 and 11435 are free, or reports exactly which PID holds them.
3. Starts `ollama serve` on 11435 and waits for `/api/version` to answer.
4. Lists the models and reports whether the air-quality tags are present.
5. Starts one proxy on 11434 and waits for it to answer.
6. Verifies each port is owned by the process it just started.
7. Verifies proxy -> backend forwarding through `/api/tags`.
8. Rejects duplicates, starts the watchdog, launches Desktop and waits for it to
   connect to 11434.

Services are started detached (`DETACHED_PROCESS`), so closing the launcher
window does not kill them.

## Automatic recovery (proxy_watchdog.py)

Every 5 seconds it verifies:

* backend answers on 11435,
* the proxy answers on 11434 **and** identifies itself with the `X-AQ-Proxy: 1`
  header (a plain Ollama server answering on 11434 does not count as healthy),
* only one listener exists per port.

Recovery rules:

| Situation | Action |
| --- | --- |
| backend down / not answering | start `ollama serve` on 11435 only |
| proxy crashed | free 11434, start one proxy, verify it forwards to the backend |
| Ollama Desktop grabbed 11434 with its own server | restart Desktop, stop that server, start the proxy, relaunch Desktop |
| duplicate proxy listeners | consolidate to one |
| anything else owns 11434 | only logged as CRITICAL, never killed |
| sleep/resume detected | re-verify both services twice and repair what answers no longer |

Safety:

* a working component is never restarted;
* restart budget: 6 restarts per service per 10 minutes, then a 10 minute pause
  with a `restart budget exhausted ... run start_ollama_proxy.bat` message;
* every action is appended to `watchdog.log`.

Why Desktop sometimes has to be restarted: as soon as the proxy disappears,
Ollama Desktop starts its own `ollama serve` on 11434 and keeps respawning it, so
the port can only be taken back by letting Desktop reconnect through a running
proxy. The watchdog does this automatically and relaunches Desktop; it is the
only case where Desktop is touched.

## Sleep and resume

Processes survive sleep, but sockets may be stale after resume. The watchdog
compares wall-clock time with its monotonic clock; a large jump is treated as a
resume, both services are re-verified twice, and anything that no longer answers
is restarted. The proxy opens a fresh connection to the backend for every
request, so a resumed session normally continues without action.

Verified: the resume code path (simulated 5-minute gap -> re-verification ->
`post-resume check OK`). **Not verified: an actual lid-close sleep/resume**, since
that cannot be automated safely on this machine. If Desktop shows an error after
waking, send the message again; if that fails, run `start_ollama_proxy.bat`.

## Optional: Windows Task Scheduler

The watchdog already recovers failures while it is running, so these tasks are
optional. Run these in PowerShell (no administrator rights needed for a task that
runs as you):

Logon task (start the integration after a restart):

```powershell
$root   = 'C:\Users\sanja\OneDrive\Documents\JAIN_Academics\SEM3\DT\SLM'
$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
          -Argument ('/c ""{0}\start_ollama_proxy.bat"" /quiet' -f $root) `
          -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName 'Air Quality SLM - start integration' `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description 'Proxy on 11434, Ollama backend on 11435, watchdog, Ollama Desktop'
```

Optional resume task (also recover after wake). It replaces the task's triggers,
so include the logon trigger again:

```powershell
$sub = '<QueryList><Query Id="0" Path="System"><Select Path="System">' +
       '*[System[Provider[@Name=''Microsoft-Windows-Power-Troubleshooter''] and EventID=1]]' +
       '</Select></Query></QueryList>'
$resume = New-CimInstance -Namespace root/Microsoft/Windows/TaskScheduler `
          -ClassName MSFT_TaskEventTrigger `
          -Property @{ Enabled = $true; Subscription = $sub } -ClientOnly
Register-ScheduledTask -TaskName 'Air Quality SLM - resume integration' -Force `
    -Action $action -Trigger @($trigger, $resume) -Settings $settings `
    -Description 'Re-verify the integration after resume'
```

Remove with `Unregister-ScheduledTask -TaskName 'Air Quality SLM - start integration' -Confirm:$false`.

Sleep settings are not changed by anything in this project.

## Emergency recovery

1. Run `check_ollama_proxy.bat` and read the FAIL lines.
2. Run `stop_ollama_proxy.bat`.
3. Run `start_ollama_proxy.bat`.
4. In Ollama Desktop select `air-quality-slm-improved:latest` and send `Hey`.
5. Run `check_ollama_proxy.bat` again: item 9 must show that request and item 10
   must show the Desktop connection.

If start reports that a port is held by a process the script does not own, close
that application (it is another Ollama server or an unrelated program) and retry.

## Rollback to plain Ollama

1. `stop_ollama_proxy.bat` (stops the watchdog, Desktop, proxy and backend).
2. Start Ollama Desktop. It launches its own server on 11434 exactly as before
   this integration.

To return to the proxy setup later, run `start_ollama_proxy.bat`.
