"""Wave 11, top-up: re-run the top level of each sweep that ran past the server's slot cap
(the slot is capped at the model's training context; see wave11b.sweep), then Nemotron to 1M.
Waits for wave11b's shallow and -ub backfill to finish, then takes over from it."""
import subprocess, sys, time
import wave10 as w
import wave11 as x
import wave11b as b

log = w.log
C_LOG = w.ROOT + r"\logs\wave11c.log"
F16 = b.F16

def handoff():
    log("handoff: waiting for 'STEP bub done' in wave11c.log")
    while "STEP bub done" not in open(C_LOG, encoding="utf-8", errors="replace").read():
        time.sleep(15)
    ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "wave11b.py" in line:
            pid = line.split("|", 1)[0].strip()
            if pid.isdigit():
                subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
                log(f"  stopped wave11b.py (pid {pid})")
    subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
    time.sleep(10)

def topups():
    # tensor split: the 256k level of the five models whose sweeps stopped at the cap
    for stem in [w.name_of(w.Q36), w.name_of(w.NEX), x.ORNITH, w.name_of(w.GEMMA26), x.Q35_9B_Q8]:
        b.sweep(f"tensor-top-{stem}", x.P(stem), [261120], F16, "tensor")
    # tensor split: Qwen3.8-27B crashed at 264k before measuring anything
    b.sweep("tensor-Qwen3.8-27B-Q4_0-131k", x.P("Qwen3.8-27B-Q4_0"), b.LV[:5], F16, "tensor")
    # layer backfill: the configurations whose deepest cell hit the cap
    groups, _ = b.backfill_groups()
    for (model, cfg), depths in sorted(groups.items(), key=lambda kv: max(kv[1])):
        top = max(depths)
        if model.startswith("NVIDIA-Nemotron") and top > 262144:
            continue                                   # the 1M sweep below covers these
        capped = (top in (261120, 130048, 131072) and ("gpt-oss" in model or top == 261120)) \
                 or model.startswith("allenai_Olmo")
        if capped and not ("-b" in cfg and cfg[cfg.index("-b") + 1] == "4096"):
            b.sweep(f"backfill-top-{model}-{'_'.join(cfg)}", x.P(model), [top], list(cfg))

if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    wanted = sys.argv[1:] or ["handoff", "topups", "bnemo"]
    steps = {"handoff": handoff, "topups": topups, "bnemo": lambda: b.bsweep("nemo")}
    log(f"WAVE 11d start: {wanted}")
    for s in wanted:
        t0 = time.time()
        try:
            steps[s]()
        except Exception as e:
            log(f"STEP {s} CRASHED: {e}")
        log(f"STEP {s} done in {(time.time()-t0)/3600:.2f} h")
    log("WAVE 11d COMPLETE")
