"""Wave 11 - finish the speed results. See report/test-plans/wave11.md.

  T  tensor split mode (-sm tensor) against layer, empty cache and at depth
  P  several sequences at once at long context (llama-batched-bench)
  S  save / restore a long context through llama-server slots
  E  MoE experts in system RAM (-ncmoe) to reach contexts that ran out of VRAM
  B  backfill every prefill cell missing from the master table

Usage:  python wave11.py [t1 t2 p s e t3 b ...]      (no argument = everything, in that order)

Reuses the wave 10 plumbing (server, sampler, bench wrapper). llama-bench rows land in
results/raw.jsonl through scripts/bench.ps1; everything else in results/wave11.jsonl.
"""
import collections, json, os, subprocess, sys, time, urllib.request
import wave10 as w

w.OUT = w.ROOT + r"\results\wave11.jsonl"
log, record, post = w.log, w.record, w.post
RAW = w.ROOT + r"\results\raw.jsonl"
BATCHED = w.TOOLS + r"\llama-batched-bench.exe"
SLOTDIR = w.ROOT + r"\tmp-slots"
D256 = 261120


# ---------------------------------------------------------------- model files
def index_models():
    idx = {}
    for d, _, files in os.walk(w.M):
        for f in files:
            if f.endswith(".gguf") and not f.startswith(("mmproj", "mtp-")):
                idx.setdefault(f[:-5], os.path.join(d, f))
    return idx

IDX = index_models()

def P(stem):
    if stem not in IDX:
        raise KeyError(f"model file not found: {stem}")
    return IDX[stem]

def raw_rows():
    out = []
    for line in open(RAW, encoding="utf-8-sig"):
        line = line.strip().lstrip("﻿")
        if line:
            r = json.loads(line)
            if r.get("ok") and not r["tag"].startswith(("harness-test", "wave10-smoke")):
                out.append(r)
    return out

NEMO, Q36, NEX = w.NEMO, w.Q36, w.NEX
ORNITH = "Ornith-1.5-35B-Q4_K_M"
Q35_9B_Q8 = "Qwen3.5-9B-Q8_0"
FAST_LONG = [name for name in (w.name_of(NEMO), w.name_of(Q36), w.name_of(NEX), ORNITH,
             w.name_of(w.GEMMA26), Q35_9B_Q8)]


# ---------------------------------------------------------------- T: tensor split
def t1():
    """Every model file with a two-card result: layer and tensor back to back, empty cache."""
    seen = []
    for r in raw_rows():
        if r["devices"] == "auto" and r["n_gen"] > 0 and r["n_depth"] == 0 and r["model"] not in seen:
            seen.append(r["model"])
    log(f"T1 tensor vs layer, empty cache: {len(seen)} model files")
    for i, m in enumerate(seen, 1):
        if m not in IDX:
            log(f"  [{i}/{len(seen)}] {m}: file missing, skipped"); continue
        log(f"  [{i}/{len(seen)}] {m}")
        w.run_bench(IDX[m], "wave11-tensor-d0", 3, ["-sm", "layer,tensor"], suite="standard")

def tensor_at(stems, depth, reps=2, tag="wave11-tensor-depth", kv=("f16", "f16")):
    for i, m in enumerate(stems, 1):
        d = depth(m) if callable(depth) else depth
        log(f"  [{i}/{len(stems)}] {m} @ {d}")
        try:
            path = P(m)
        except KeyError as e:
            log(f"  {e}"); continue
        w.run_bench(path, tag, reps, ["-sm", "tensor", "-fa", "on", "-ctk", kv[0], "-ctv", kv[1],
                                      "-d", str(d)], suite="quick")

def t2():
    fast128 = FAST_LONG + ["gpt-oss-20b-MXFP4", "Qwen3-Coder-Next-UD-Q2_K_XL",
                           "Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K"]
    log("T2a tensor at 128k, fast long-context models")
    tensor_at(fast128, 131072)
    shallow = {"Qwen3-32B-Q6_K": 16384, "Seed-OSS-36B-Instruct-Q6_K": 8192,
               "Llama-3.3-70B-Instruct-UD-Q2_K_XL": 8192}
    dense32 = ["Qwen3.8-27B-Q4_0", "gemma-4-31B_q4_0-it", "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL",
               "mistralai_Magistral-Small-2509-Q6_K", "allenai_Olmo-3.1-32B-Think-Q5_K_M",
               "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M", "phi-4-Q8_0", "Meta-Llama-3.1-8B-Instruct-Q8_0",
               "zai-org_GLM-4.7-Flash-Q5_K_M", "Mistral-Small-4-119B-2603.i1-IQ1_M",
               "Qwen3.5-122B-A10B.i1-IQ1_M", "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"] + list(shallow)
    log("T2b tensor at 32k (or the model's own ceiling), dense and slow models")
    tensor_at(dense32, lambda m: shallow.get(m, 32768))

def t3():
    log("T3 tensor at 256k")
    tensor_at(FAST_LONG, D256)


# ---------------------------------------------------------------- P: parallel sequences
def batched(label, model, npp, npl, extra=()):
    c = max(npl) * (npp + 128)
    args = [BATCHED, "-m", model, "-c", str(c), "-b", "2048", "-ub", "512", "-ngl", "999",
            "-fa", "on", "-npp", str(npp), "-ntg", "128", "-npl", ",".join(map(str, npl)),
            "--output-format", "jsonl"] + list(extra)
    log(f"P {label}: npp {npp}, parallel {npl}, ctx {c}")
    csvp = os.path.join(w.LOGD, f"wave11-batched.{label}.{time.strftime('%Y%m%d-%H%M%S')}.gpu.csv")
    errp = csvp.replace(".gpu.csv", ".stderr.txt")
    samp = w.Sampler(csvp)
    t0 = time.time()
    with open(errp, "w", encoding="utf-8", errors="replace") as ef:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=ef, text=True, errors="replace")
    samp.stop()
    base = {"exp": "parallel", "label": label, "model": w.name_of(model), "npp": npp, "ctx": c,
            "wall_s": round(time.time() - t0, 1), "gpu": w.summarize_gpu(csvp)}
    got = 0
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            record({**base, **row}); got += 1
            log(f"  pl {row.get('pl')}: prefill {row.get('speed_pp', 0):8.1f} t/s total, "
                f"gen {row.get('speed_tg', 0):7.2f} t/s total")
    if not got:
        tail = open(errp, encoding="utf-8", errors="replace").read().splitlines()[-4:]
        record({**base, "error": " | ".join(tail)[-400:]}); log(f"  FAILED: {' | '.join(tail)}")

def p():
    batched("nemotron-32k", NEMO, 32768, [1, 2, 4, 8])
    batched("qwen36-64k", Q36, 65536, [1, 2, 4])
    batched("qwen35-9b-64k", P(Q35_9B_Q8), 65536, [1, 2, 4])
    batched("gptoss-32k", w.GPTOSS, 32768, [1, 2, 4, 8])
    batched("qwen38-27b-16k", w.Q38_27B, 16384, [1, 2, 4])
    batched("nemotron-256k", NEMO, D256, [2, 4])


# ---------------------------------------------------------------- S: slot save / restore
def slot(action, filename, timeout=3600):
    return post(f"/slots/0?action={action}", {"filename": filename} if filename else {}, timeout)

def save_restore(label, model, ctx, more=()):
    os.makedirs(SLOTDIR, exist_ok=True)
    fn = f"{label}.bin"
    fpath = os.path.join(SLOTDIR, fn)
    base = {"exp": "slots", "label": label, "model": w.name_of(model), "ctx": ctx, "server_args": list(more)}
    extra = ["-np", "1", "-fa", "on", "--slot-save-path", SLOTDIR] + list(more)
    log(f"S {label}: ctx {ctx}")
    srv = w.Server(model, ctx, extra, f"slots-{ctx}")
    try:
        srv.start()
        doc = w.build_document(ctx - 2048, needles=False)[0]
        t0 = time.time()
        r = post("/completion", {"prompt": doc, "n_predict": 1, "temperature": 0, "cache_prompt": True}, 6 * 3600)
        fill_s = time.time() - t0
        n = r["timings"]["prompt_n"]
        log(f"  filled {n} tokens in {fill_s:.1f}s")
        rs = slot("save", fn)
        size = os.path.getsize(fpath)
        log(f"  saved {rs.get('n_saved')} tokens, {size/2**30:.2f} GiB in {rs['timings']['save_ms']/1000:.1f}s")
        slot("erase", None)
        rr = slot("restore", fn)
        log(f"  restored {rr.get('n_restored')} tokens in {rr['timings']['restore_ms']/1000:.1f}s")
        t1 = time.time()
        f1 = post("/completion", {"prompt": doc + "\n\nIn one sentence, what is this text about?",
                                  "n_predict": 32, "temperature": 0, "cache_prompt": True}, 6 * 3600)
        follow_s = time.time() - t1
        log(f"  follow-up after restore: re-read {f1['timings']['prompt_n']} tokens, "
            f"{follow_s:.1f}s to finish")
        row = {**base, "context_tokens": n, "fill_s": round(fill_s, 1), "file_bytes": size,
               "save_s": round(rs["timings"]["save_ms"] / 1000, 2),
               "restore_s": round(rr["timings"]["restore_ms"] / 1000, 2),
               "followup_reread_tokens": f1["timings"]["prompt_n"], "followup_s": round(follow_s, 2)}
        if f1["timings"]["prompt_n"] > n // 2:
            # the restore bought nothing (hybrid / SWA cache); a restart would only repeat the fill
            record(row); return
        srv.stop()
        # new process, same file
        srv = w.Server(model, ctx, extra, f"slots-{ctx}-restart")
        t2 = time.time(); srv.start(); load_s = time.time() - t2
        rr2 = slot("restore", fn)
        t3 = time.time()
        f2 = post("/completion", {"prompt": doc + "\n\nIn one sentence, what is this text about?",
                                  "n_predict": 32, "temperature": 0, "cache_prompt": True}, 6 * 3600)
        follow2_s = time.time() - t3
        log(f"  after restart: load {load_s:.0f}s, restore {rr2['timings']['restore_ms']/1000:.1f}s, "
            f"re-read {f2['timings']['prompt_n']} tokens, follow-up {follow2_s:.1f}s")
        record({**row, "restart_load_s": round(load_s, 1),
                "restart_restore_s": round(rr2["timings"]["restore_ms"] / 1000, 2),
                "restart_reread_tokens": f2["timings"]["prompt_n"], "restart_followup_s": round(follow2_s, 2)})
    except Exception as e:
        record({**base, "error": str(e)[:400], "server_tail": srv.tail()})
        log(f"  FAILED: {e} :: {srv.tail()}")
    finally:
        srv.stop()
        try: os.remove(fpath)
        except OSError: pass

def s():
    # full attention restores cleanly; SWA needs --swa-full; hybrid recurrent re-reads (smoke, 14k)
    save_restore("llama31-8b-128k", P("Meta-Llama-3.1-8B-Instruct-Q8_0"), 131072)
    save_restore("gptoss-128k-swafull", w.GPTOSS, 131072, ["--swa-full"])
    save_restore("gptoss-128k", w.GPTOSS, 131072)
    save_restore("coder30b-128k-q8", P("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"), 131072,
                 ["-ctk", "q8_0", "-ctv", "q8_0"])
    save_restore("gemma26-128k-swafull", w.GEMMA26, 131072, ["--swa-full"])
    save_restore("nemotron-128k", NEMO, 131072)
    save_restore("qwen36-128k", Q36, 131072)


# ---------------------------------------------------------------- E: experts in RAM
def offload(stem, depth, tries, kv=("f16", "f16")):
    path = P(stem)
    log(f"E {stem} @ {depth}: trying -ncmoe {tries}")
    for n in tries:
        ok = w.run_bench(path, "wave11-ncmoe", 1 if depth > 32768 else 2,
                         ["-fa", "on", "-ctk", kv[0], "-ctv", kv[1], "-ncmoe", str(n), "-d", str(depth)],
                         suite="quick")
        if ok:
            log(f"  fits with {n} expert layers on the CPU; empty-cache cost of that offload:")
            w.run_bench(path, "wave11-ncmoe", 3, ["-fa", "on", "-ncmoe", str(n)], suite="quick")
            return n
    log("  no setting fit"); return None

def e():
    offload("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", 131072, [8, 12, 16, 20, 24, 32, 48])
    offload("Qwen3.6-35B-A3B-UD-Q6_K", D256, [4, 8, 12, 16, 24, 40])
    offload("Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K", D256, [2, 4, 8, 12, 16])
    offload("Qwen3-Coder-Next-UD-Q2_K_XL", D256, [2, 4, 8, 12, 16])
    offload("Qwen3.5-122B-A10B.i1-IQ1_M", D256, [2, 4, 8, 12, 16])
    offload("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", D256, [24, 32, 40, 48])


# ---------------------------------------------------------------- B: backfill prefill cells
def backfill_jobs():
    rows = raw_rows()
    have = set()
    for r in rows:
        if r["n_prompt"] > 0:
            have.add((r["ts"], r["n_depth"], r["type_k"], r["devices"], r["n_ubatch"], r["n_prompt"]))
    need = collections.defaultdict(set)
    for r in rows:
        if r["n_gen"] <= 0 or r["tag"].startswith("wave11"):
            continue
        for pp in (512, 2048):
            if (r["ts"], r["n_depth"], r["type_k"], r["devices"], r["n_ubatch"], pp) not in have:
                key = (r["model"], r["type_k"], r["type_v"], int(r["flash_attn"]), r["devices"],
                       r["n_ubatch"], r["n_batch"], r["model_size_gb"])
                need[(key, pp)].add(r["n_depth"])
    jobs = []
    for (key, pp), depths in need.items():
        model, tk, tv, fa, dev, ub, b, size = key
        extra = ["-ctk", tk, "-ctv", tv, "-fa", {1: "on", 0: "off"}.get(fa, "auto"),
                 "-ub", str(ub), "-b", str(b)]
        if dev != "auto":
            extra += ["-dev", dev.replace(",", "/")]
        shallow = sorted(d for d in depths if d <= 32768)
        if shallow:
            jobs.append((sum(shallow) * 2 * size, model, pp, 2, extra + ["-d", ",".join(map(str, shallow))]))
        for d in sorted(d for d in depths if d > 32768):
            jobs.append((d * size, model, pp, 1, extra + ["-d", str(d)]))
    jobs.sort(key=lambda j: j[0])
    return jobs

def b():
    jobs = backfill_jobs()
    log(f"B backfill: {len(jobs)} invocations")
    for i, (_, model, pp, reps, extra) in enumerate(jobs, 1):
        if model not in IDX:
            log(f"  [{i}/{len(jobs)}] {model}: file missing, skipped"); continue
        log(f"  [{i}/{len(jobs)}] {model} pp{pp}")
        w.run_bench(IDX[model], "wave11-backfill", reps, extra, suite=f"pp{pp}")


STEPS = {"t1": t1, "t2": t2, "p": p, "s": s, "e": e, "t3": t3, "b": b}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["t1", "t2", "p", "s", "e", "t3", "b"]
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    log(f"WAVE 11 start: {wanted}")
    for step in wanted:
        t0 = time.time()
        try:
            STEPS[step]()
        except Exception as ex:
            log(f"STEP {step} CRASHED: {ex}")
        log(f"STEP {step} done in {(time.time()-t0)/3600:.2f} h")
    log("WAVE 11 COMPLETE")
