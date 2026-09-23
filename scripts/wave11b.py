"""Wave 11, second half - the same measurements with one context fill per model.

llama-bench refills the cache from zero for every test and every repetition, so pp512, pp2048
and tg128 at 256k cost several 256k fills (and several 1M fills for Nemotron's deepest cell).
Here a llama-server fills the context once and climbs: at each level it reads 512 new tokens
(pp512 at that depth), 2,048 more (pp2048), then writes 128 (tg). The next level extends the
same token sequence, so the cache is never rebuilt. Prompts are sent as token ids, so every
segment is exactly the size asked for.

Steps (default: all, in this order):
  handoff   wait for wave11.py to finish its current parallel run, then stop it
  validate  Nemotron at 32k and 64k, to compare with the llama-bench figures already published
  p         the parallel-sequence runs wave11.py had not reached
  tsweep    tensor split at depth, one fill per model
  s         slot save / restore (cases the 14k smoke test had not already settled)
  bsweep    backfill of missing prefill cells deeper than 32k, one fill per configuration
  bbench    backfill of the shallow cells (<= 32k) with llama-bench, where refills are cheap
  bub       backfill at the -ub test settings (four servers per model)
  bnemo     Nemotron f16 to 1M, last because it is the longest single fill

Rows go to results/wave11.jsonl with exp "depth_sweep" (llama-bench rows to raw.jsonl).
"""
import collections, json, os, subprocess, sys, time, urllib.request
import wave10 as w
import wave11 as x

log, record, post = w.log, w.record, w.post
WAVE11_LOG = w.ROOT + r"\logs\wave11.log"
SEG = (512, 2048)          # new tokens read at each level: pp512, then pp2048
GEN = 128
HEAD = SEG[0] + SEG[1] + GEN + 256   # context needed above the top level


# ---------------------------------------------------------------- handoff
def handoff():
    """wave11.py is mid-way through step P. Let its current llama-batched-bench run finish,
    then stop it: its remaining steps are replaced by the ones below."""
    marker = "P qwen35-9b-64k"
    log(f"handoff: waiting for '{marker}' in wave11.log")
    while marker not in open(WAVE11_LOG, encoding="utf-8", errors="replace").read():
        time.sleep(10)
    ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "wave11.py" in line and "wave11b.py" not in line:
            pid = line.split("|", 1)[0].strip()
            if pid.isdigit():
                subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
                log(f"  stopped wave11.py (pid {pid}) and its children")
    subprocess.run(["taskkill", "/IM", "llama-batched-bench.exe", "/F"], capture_output=True)
    time.sleep(10)


# ---------------------------------------------------------------- the sweep
_TOKS = {}
def tokens_for(model_key, n):
    """First n token ids of WikiText train under this server's tokenizer."""
    have = _TOKS.get(model_key)
    if have and len(have) >= n:
        return have
    text = w.wiki()
    sample = text[:400_000]
    cpt = len(sample) / len(post("/tokenize", {"content": sample}, 600)["tokens"])
    nchars = int(n * cpt * 1.05) + 20_000
    toks = post("/tokenize", {"content": text[:nchars]}, 3600)["tokens"]
    if len(toks) < n:
        raise RuntimeError(f"WikiText too short: {len(toks)} tokens < {n}")
    _TOKS[model_key] = toks
    return toks

def slot_ctx():
    try:
        with urllib.request.urlopen(w.BASE + "/slots", timeout=30) as r:
            return int(json.loads(r.read())[0]["n_ctx"])
    except Exception:
        try:
            with urllib.request.urlopen(w.BASE + "/props", timeout=30) as r:
                return int(json.loads(r.read())["default_generation_settings"]["n_ctx"])
        except Exception:
            return None

def req(toks, n_predict):
    body = {"prompt": toks, "n_predict": n_predict, "cache_prompt": True, "temperature": 0}
    if n_predict:
        body.update({"ignore_eos": True, "return_tokens": True})
    return post("/completion", body, 8 * 3600)

# No context checkpoints and no host-RAM prompt cache: the climb never rolls back, so neither is
# needed, and both cost time on every request (and the second copies the cache to system RAM).
NO_ROLLBACK = ["-ctxcp", "0", "--cache-ram", "0"]

def sweep(label, model, levels, args, split="layer", tops=None):
    """One server, one climbing fill. levels: depths to measure at. tops: fallback top levels
    to try if the context does not allocate (largest first).

    The sequence only ever grows. At each level: fill up to the level, read 512 new tokens
    (pp512 at that depth), then read 2,048 more and write 128 in the same request (pp2048 and
    tg). The written tokens stay in the sequence, so the next level extends it and nothing has
    to be rolled back - which a recurrent or sliding-window cache cannot do without checkpoints."""
    levels = sorted(set(levels))
    tops = tops or [levels[-1]]
    csvp = os.path.join(w.LOGD, f"wave11-sweep.{label}.{time.strftime('%Y%m%d-%H%M%S')}.gpu.csv")
    base = {"exp": "depth_sweep", "label": label, "model": w.name_of(model), "split": split,
            "server_args": args, "method": "server-climb-v2"}
    srv = None
    for top in tops:
        ctx = top + HEAD
        s = w.Server(model, ctx, ["-np", "1", "-sm", split] + NO_ROLLBACK + args, f"sweep-{split}-{top}")
        try:
            s.start(); srv = s; break
        except Exception as e:
            record({**base, "ctx_try": ctx, "loaded": False, "error": str(e)[-300:]})
            log(f"  ctx {ctx}: did not load"); s.stop()
    if not srv:
        log(f"SWEEP {label}: nothing loaded"); return
    lv = [l for l in levels if l <= top]
    # llama-server caps a slot at the model's training context even when -c asks for more,
    # so a level whose measurements would run past the slot is moved down to the last level
    # that fits (llama-bench has no such cap, which is where 261,120 and 1,047,552 come from)
    n_ctx = slot_ctx()
    if n_ctx:
        fit = n_ctx - (SEG[0] + SEG[1] + GEN + 64)
        lv = sorted({min(l, fit) for l in lv})
        base["slot_ctx"] = n_ctx
    log(f"SWEEP {label} ({split}): levels {lv}, ctx {ctx}, slot {n_ctx}")
    samp = w.Sampler(csvp)
    base["ctx"] = ctx
    try:
        doc = tokens_for(w.name_of(model), lv[-1] + SEG[0] + SEG[1] + 8)
        seq, ptr = [], 0
        def grow(n):
            nonlocal ptr
            seq.extend(doc[ptr:ptr + n]); ptr += n
        for L in lv:
            need = L - len(seq)
            if need < 0:
                log(f"  {L}: below the tokens already in context ({len(seq)}), skipped")
                record({**base, "level": L, "skipped": "overlaps previous level"}); continue
            t0 = time.time()
            fill = {"prompt_n": 0, "prompt_ms": 0}
            if need:
                grow(need); fill = req(seq, 0)["timings"]
            d512 = len(seq); grow(SEG[0])
            r1 = req(seq, 0)["timings"]
            d2048 = len(seq); grow(SEG[1])
            r = req(seq, GEN)
            r2 = r["timings"]
            seq.extend(r.get("tokens") or [])
            row = {**base, "level": L,
                   "fill_tokens": fill["prompt_n"], "fill_s": round(fill["prompt_ms"] / 1000, 1),
                   "pp512_depth": d512, "pp512_n": r1["prompt_n"], "pp512_tps": round(r1["prompt_per_second"], 2),
                   "pp2048_depth": d2048, "pp2048_n": r2["prompt_n"],
                   "pp2048_tps": round(r2["prompt_per_second"], 2),
                   "tg_depth": d2048 + SEG[1], "tg_n": r2["predicted_n"],
                   "tg_tps": round(r2["predicted_per_second"], 3),
                   "wall_s": round(time.time() - t0, 1)}
            # more tokens read than were new means the cache was rebuilt: not a rate at this depth
            row["clean"] = (r1["prompt_n"] <= SEG[0] + 1 and r2["prompt_n"] <= SEG[1] + 1
                            and fill["prompt_n"] <= need + 1)
            record(row)
            log(f"  {L:>8}: pp512 {row['pp512_tps']:8.1f}  pp2048 {row['pp2048_tps']:8.1f}  "
                f"tg {row['tg_tps']:7.2f}  (fill {fill['prompt_n']} tok in {row['fill_s']}s)"
                + ("" if row["clean"] else f"  RE-READ {r1['prompt_n']}/{r2['prompt_n']}/{fill['prompt_n']}"))
    except Exception as e:
        record({**base, "error": str(e)[:500], "server_tail": srv.tail()})
        log(f"  SWEEP FAILED: {e} :: {srv.tail()}")
    finally:
        srv.stop(); samp.stop()
        g = w.summarize_gpu(csvp)
        record({**base, "summary": True, "gpu": g})
        log(f"  vram peak: {', '.join(f'card{k} {v['mem_max_mib']} MiB' for k, v in sorted(g.items()))}")


# ---------------------------------------------------------------- steps
F16 = ["-fa", "on", "-ctk", "f16", "-ctv", "f16", "-b", "2048", "-ub", "512"]
LV = [4096, 16384, 32768, 65536, 131072, 261120]

def validate():
    # llama-bench, same settings: pp512 329.77 / 265.57, tg128 64.75 / 59.03 at 32k / 64k
    sweep("validate-nemotron", w.NEMO, [32768, 65536], F16)

def p():
    x.batched("qwen35-9b-64k", x.P(x.Q35_9B_Q8), 65536, [1, 2, 4])
    x.batched("gptoss-32k", w.GPTOSS, 32768, [1, 2, 4, 8])
    x.batched("qwen38-27b-16k", w.Q38_27B, 16384, [1, 2, 4])
    x.batched("nemotron-256k", w.NEMO, x.D256, [4])

def tsweep():
    # Nemotron is not here: it fails to load with -sm tensor in this build.
    for stem in [w.name_of(w.Q36), w.name_of(w.NEX), x.ORNITH, w.name_of(w.GEMMA26), x.Q35_9B_Q8]:
        sweep(f"tensor-{stem}", x.P(stem), LV, F16, "tensor")
    for stem in ["gpt-oss-20b-MXFP4", "Meta-Llama-3.1-8B-Instruct-Q8_0"]:
        sweep(f"tensor-{stem}", x.P(stem), LV[:4] + [126976], F16, "tensor")
    for stem in ["Qwen3-Coder-Next-UD-Q2_K_XL", "Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K",
                 "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", "Qwen3.5-122B-A10B.i1-IQ1_M",
                 "Qwen3.8-27B-Q4_0", "gemma-4-31B_q4_0-it", "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL",
                 "mistralai_Magistral-Small-2509-Q6_K", "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M"]:
        sweep(f"tensor-{stem}", x.P(stem), LV, F16, "tensor", tops=[261120, 131072, 65536, 32768])

def s():
    x.save_restore("llama31-8b-128k", x.P("Meta-Llama-3.1-8B-Instruct-Q8_0"), 131072)
    x.save_restore("gptoss-128k-swafull", w.GPTOSS, 131072, ["--swa-full"])
    x.save_restore("coder30b-128k-q8", x.P("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"), 131072,
                   ["-ctk", "q8_0", "-ctv", "q8_0"])
    x.save_restore("gemma26-128k-swafull", w.GEMMA26, 131072, ["--swa-full"])
    x.save_restore("nemotron-128k", w.NEMO, 131072)

def backfill_groups():
    """Deep (> 32k) missing cells from wave11.backfill_jobs, grouped by configuration so one
    fill covers every depth that configuration needs."""
    groups = collections.defaultdict(set)
    shallow = []
    for job in x.backfill_jobs():
        _, model, pp, reps, extra = job
        i = extra.index("-d")
        depths = [int(v) for v in extra[i + 1].split(",")]
        if max(depths) <= 32768:
            shallow.append(job); continue
        groups[(model, tuple(extra[:i]))].update(depths)
    return groups, shallow

def to_server_args(cfg):
    a = list(cfg)
    if "-dev" in a:                       # llama-bench and llama-server spell the device list alike
        pass
    return a

def bsweep(which="main"):
    groups, _ = backfill_groups()
    items = sorted(groups.items(), key=lambda kv: max(kv[1]))
    for (model, cfg), depths in items:
        is_ub = "4096" in cfg and "-b" in cfg and cfg[cfg.index("-b") + 1] == "4096"
        is_nemo_f16 = model.startswith("NVIDIA-Nemotron") and max(depths) > 262144
        kind = "ub" if is_ub else "nemo" if is_nemo_f16 else "main"
        if kind != which:
            continue
        depths = sorted(depths)
        # levels closer together than one measurement's worth of tokens cannot share a climb
        keep, extra_jobs, last = [], [], -10**9
        for d in depths:
            if d - last < SEG[0] + SEG[1] + GEN:
                extra_jobs.append(d)
            else:
                keep.append(d); last = d
        sweep(f"backfill-{model}-{'_'.join(cfg)}", x.P(model), keep, to_server_args(cfg))
        for d in extra_jobs:
            for pp in (512, 2048):
                w.run_bench(x.P(model), "wave11-backfill", 1, list(cfg) + ["-d", str(d)], suite=f"pp{pp}")

def bbench():
    _, shallow = backfill_groups()
    log(f"B shallow backfill with llama-bench: {len(shallow)} invocations")
    for i, (_, model, pp, reps, extra) in enumerate(shallow, 1):
        log(f"  [{i}/{len(shallow)}] {model} pp{pp}")
        w.run_bench(x.P(model), "wave11-backfill", reps, extra, suite=f"pp{pp}")

STEPS = {"handoff": handoff, "validate": validate, "p": p, "tsweep": tsweep, "s": s,
         "bsweep": lambda: bsweep("main"), "bbench": bbench, "bub": lambda: bsweep("ub"),
         "bnemo": lambda: bsweep("nemo")}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["handoff", "validate", "p", "tsweep", "s", "bsweep", "bbench", "bub", "bnemo"]
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    log(f"WAVE 11b start: {wanted}")
    for step in wanted:
        t0 = time.time()
        try:
            STEPS[step]()
        except Exception as ex:
            log(f"STEP {step} CRASHED: {ex}")
        log(f"STEP {step} done in {(time.time()-t0)/3600:.2f} h")
    log("WAVE 11b COMPLETE")
