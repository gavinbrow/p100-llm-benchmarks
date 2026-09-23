"""Wave 10 - does the long-context result survive real use? See report/test-plans/wave10.md.

Drives llama-server directly (llama-bench cannot do multi-turn sessions, speculative
decoding or sustained load) and appends one JSON row per measurement to
results/wave10.jsonl. The llama-bench parts go through scripts/bench.ps1 so their rows
land in results/raw.jsonl with the same schema as every other wave.

Usage:  python wave10.py [experiment ...]      e.g.  python wave10.py e0 e2a
        python wave10.py all
        python wave10.py smoke                  quick self-test on a small model

No third-party dependencies: urllib for HTTP, subprocess for the server and nvidia-smi.
"""
import json, os, random, subprocess, sys, time, urllib.error, urllib.request

ROOT    = r"C:\Projects\p100 testing"
TOOLS   = ROOT + r"\tools\llamacpp"
SERVER  = TOOLS + r"\llama-server.exe"
TOKENIZE = TOOLS + r"\llama-tokenize.exe"
BENCH   = ROOT + r"\scripts\bench.ps1"
LOGD    = ROOT + r"\logs"
OUT     = ROOT + r"\results\wave10.jsonl"
WIKI_TRAIN = ROOT + r"\data\wikitext-2-raw\wiki.train.raw"
WIKI_TEST  = ROOT + r"\data\wikitext-2-raw\wiki.test.raw"
M = r"C:\Users\PC\.lmstudio\models"
PORT = 8110
BASE = f"http://127.0.0.1:{PORT}"

NEMO   = M + r"\ggml-org\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
NEMO_MTP = M + r"\ggml-org\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF\mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
Q36    = M + r"\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
NEX    = M + r"\bartowski\nex-agi_Nex-N2.5-mini-GGUF\nex-agi_Nex-N2.5-mini-Q4_K_M.gguf"
GEMMA26 = M + r"\lmstudio-community\gemma-4-26B-A4B-it-QAT-GGUF\gemma-4-26B-A4B-it-QAT-Q4_0.gguf"
Q35_9B = M + r"\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf"
CODERNEXT = M + r"\unsloth\Qwen3-Coder-Next-GGUF\Qwen3-Coder-Next-UD-Q2_K_XL.gguf"
NEXT80 = M + r"\bartowski\Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF\Qwen_Qwen3-Next-80B-A3B-Instruct-Q2_K.gguf"
GPTOSS = M + r"\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
Q38_27B = M + r"\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_0.gguf"
Q38_2B = M + r"\unsloth\Qwen3.8-2B-GGUF\Qwen3.8-2B-Q8_0.gguf"

STAMP = time.strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------- plumbing
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def record(row):
    row = {"ts": time.strftime("%Y%m%d-%H%M%S"), **row}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def post(path, body, timeout):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def name_of(path):
    return os.path.splitext(os.path.basename(path))[0]

def gpu_temps():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]

def wait_cool(target=45, cap_s=900):
    t0 = time.time()
    while True:
        t = gpu_temps()
        if t and max(t) <= target:
            log(f"  cards at {t} C after {time.time()-t0:.0f}s"); return t
        if time.time() - t0 > cap_s:
            log(f"  cooldown cap reached, cards at {t} C"); return t
        time.sleep(15)


class Sampler:
    """nvidia-smi at 1 Hz straight to CSV, including the clock-event (throttle) bitmask."""
    FIELDS = ("timestamp,index,temperature.gpu,power.draw,utilization.gpu,memory.used,"
              "clocks.current.graphics,clocks.current.memory,pstate,clocks_event_reasons.active")
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.f.write(self.FIELDS + "\n"); self.f.flush()
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=" + self.FIELDS,
                                   "--format=csv,noheader,nounits", "-l", "1"], stdout=self.f)
    def stop(self):
        self.p.terminate()
        try: self.p.wait(10)
        except Exception: self.p.kill()
        self.f.close()

def summarize_gpu(path, t_from=None):
    """Per card: max temp, mean power, mean graphics clock, and the share of samples in which
    a throttle reason other than 'GPU idle' (0x1) was active."""
    per = {}
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) < 10: continue
        try:
            idx, temp, pwr, util, mem, clk = p[1], int(p[2]), float(p[3]), int(p[4]), int(p[5]), float(p[6])
            reasons = int(p[9], 16)
        except ValueError:
            continue
        d = per.setdefault(idx, {"n": 0, "tmax": 0, "pw": 0.0, "clk": 0.0, "thr": 0, "reasons": 0, "mem": 0})
        d["mem"] = max(d["mem"], mem)
        if util < 50: continue          # only loaded samples below here
        d["n"] += 1; d["tmax"] = max(d["tmax"], temp); d["pw"] += pwr; d["clk"] += clk
        if reasons & ~0x1: d["thr"] += 1
        d["reasons"] |= reasons
    return {k: {"loaded_samples": v["n"], "temp_max": v["tmax"], "mem_max_mib": v["mem"],
                "power_mean": round(v["pw"]/v["n"], 1) if v["n"] else None,
                "gfx_clock_mean": round(v["clk"]/v["n"]) if v["n"] else None,
                "throttled_pct": round(100*v["thr"]/v["n"], 1) if v["n"] else None,
                "reasons_seen": hex(v["reasons"])}
            for k, v in per.items()}


class Server:
    def __init__(self, model, ctx, extra, tag):
        self.model, self.ctx, self.extra, self.tag = model, ctx, list(extra), tag
        self.logpath = os.path.join(LOGD, f"server.wave10.{name_of(model)}.{tag}.{time.strftime('%Y%m%d-%H%M%S')}.log")
        self.p = None
    def start(self, timeout=1200):
        args = [SERVER, "-m", self.model, "--host", "127.0.0.1", "--port", str(PORT),
                "-ngl", "999", "-c", str(self.ctx), "--no-webui"] + self.extra
        log(f"  server: {' '.join(a if ' ' not in a else repr(a) for a in args[1:])}")
        self.lf = open(self.logpath, "w", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen(args, stdout=self.lf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited ({self.p.returncode}): " + self.tail())
            try:
                with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        log(f"  server ready in {time.time()-t0:.0f}s"); return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("server did not become ready: " + self.tail())
    def tail(self, n=6):
        try:
            self.lf.flush()
            lines = open(self.logpath, encoding="utf-8", errors="replace").read().splitlines()
            return " | ".join(l.strip() for l in lines[-n:])
        except Exception as e:
            return str(e)
    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try: self.p.wait(30)
            except Exception: self.p.kill(); self.p.wait(10)
        try: self.lf.close()
        except Exception: pass
        time.sleep(5)


def run_bench(model, tag, reps, extra, suite="quick"):
    arr = ",".join("'" + a + "'" for a in extra)
    cmd = (f"& '{BENCH}' -Model '{model}' -Suite {suite} -Tag '{tag}' -Reps {reps} -Extra @({arr})")
    log(f"  bench: {tag} {' '.join(extra)}")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "t/s" in line or "FAILED" in line: log("   " + line.strip())
    return r.returncode == 0


# ---------------------------------------------------------------- text
_WIKI = None
def wiki():
    global _WIKI
    if _WIKI is None:
        _WIKI = open(WIKI_TRAIN, encoding="utf-8").read()
    return _WIKI

def ntokens(text):
    return len(post("/tokenize", {"content": text}, 600)["tokens"])

NAMES = ["Heron", "Basalt", "Juniper", "Quarry", "Lantern", "Meridian", "Thistle", "Cobalt"]
DEPTHS = [0.05, 0.18, 0.31, 0.44, 0.57, 0.70, 0.83, 0.95]

def build_document(target_tokens, needles=True, seed=7):
    """WikiText cut to target_tokens with the server's own tokenizer, optionally with eight
    planted facts at fixed depths. Returns (doc, codes, chars_used, chars_per_token)."""
    text = wiki()
    sample = text[:400_000]
    cpt = len(sample) / ntokens(sample)
    rng = random.Random(seed)
    codes = {n: str(rng.randint(100000, 999999)) for n in NAMES}
    nchars = int(target_tokens * cpt * 0.97)
    for attempt in range(4):
        body = text[:nchars]
        cut = body.rfind("\n"); body = body[:cut] if cut > 0 else body
        doc = body
        if needles:
            parts, last = [], 0
            for n, d in zip(NAMES, DEPTHS):
                pos = body.find("\n", int(len(body) * d))
                pos = len(body) if pos < 0 else pos
                parts.append(body[last:pos])
                parts.append(f"\nArchive note: the access code for the {n} vault is {codes[n]}.\n")
                last = pos
            parts.append(body[last:])
            doc = "".join(parts)
        n = ntokens(doc)
        if n <= target_tokens:
            return doc, codes, len(body), cpt, n
        nchars = int(nchars * target_tokens / n * 0.99)
    raise RuntimeError("could not size document")


# ---------------------------------------------------------------- E0 cold canary
def e0():
    log("E0: cold canary, Qwen3.8-27B Q4_0 on each card alone")
    wait_cool(45, 600)
    for dev in ("CUDA0", "CUDA1"):
        run_bench(Q38_27B, f"wave10-card{dev[-1]}-cold", 3, ["-dev", dev, "-ctk", "f16", "-ctv", "f16", "-fa", "on"])


# ---------------------------------------------------------------- E2 agent sessions
SYSTEM = ("You are a careful research assistant. Answer questions using only the documents and "
          "tool results in this conversation. Be brief.")

def agent_session(label, model, ctx, dev_args, reasoning, turns=8, max_tokens=64, extra=()):
    tag = f"agent-{label}"
    log(f"E2 {label}: {name_of(model)} ctx={ctx} reasoning={reasoning}")
    srv = Server(model, ctx, list(dev_args) + ["-np", "1", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
                                                "--reasoning", reasoning] + list(extra), tag)
    base = {"exp": "agent", "label": label, "model": name_of(model), "ctx": ctx,
            "reasoning": reasoning, "cards": 1 if dev_args else 2, "extra": " ".join(extra)}
    try:
        srv.start()
        doc, codes, used, cpt, doc_tokens = build_document(ctx - 24000)
        log(f"  document {doc_tokens} tokens ({cpt:.2f} chars/token)")
        chunk_chars = int(2000 * cpt)
        off = used
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"<document>\n{doc}\n</document>\n\n"
                     f"What is the access code for the {NAMES[0]} vault? Reply with only the code."}]
        for turn in range(turns):
            t0 = time.time()
            try:
                resp = post("/v1/chat/completions", {"messages": messages, "max_tokens": max_tokens,
                                                     "temperature": 0, "cache_prompt": True}, 4 * 3600)
            except Exception as e:
                record({**base, "turn": turn, "error": str(e)[:300], "server_tail": srv.tail()})
                log(f"  turn {turn} FAILED: {e}"); break
            wall = time.time() - t0
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
            rc = msg.get("reasoning_content") or ""
            tim = resp.get("timings", {}) or {}
            usage = resp.get("usage", {}) or {}
            total = usage.get("prompt_tokens")
            pn = tim.get("prompt_n")
            ok = codes[NAMES[turn]] in content
            row = {**base, "turn": turn, "needle": NAMES[turn], "needle_depth": DEPTHS[turn],
                   "doc_tokens": doc_tokens, "context_tokens": total, "prompt_n": pn,
                   "cache_n": tim.get("cache_n"), "prompt_ms": tim.get("prompt_ms"),
                   "prompt_tps": tim.get("prompt_per_second"), "predicted_n": tim.get("predicted_n"),
                   "predicted_tps": tim.get("predicted_per_second"), "wall_s": round(wall, 1),
                   "correct": ok, "answer": content.strip()[:80], "reasoning_chars": len(rc),
                   "finish": resp["choices"][0].get("finish_reason")}
            record(row)
            log(f"  turn {turn}: ctx={total} processed={pn} cached={tim.get('cache_n')} "
                f"ttft={ (tim.get('prompt_ms') or 0)/1000:.1f}s gen={tim.get('predicted_per_second') or 0:.1f}t/s "
                f"wall={wall:.0f}s {'OK' if ok else 'MISS'} [{content.strip()[:30]!r}]")
            if turn > 0 and pn and total and pn > 0.5 * total:
                log("  -> turn re-processed over half the context; recording and stopping this session")
                record({**base, "turn": turn, "note": "full_reprocess_guard"}); break
            if turn + 1 < turns:
                messages.append({"role": "assistant", "content": content})
                chunk = wiki()[off: off + chunk_chars]; off += chunk_chars
                messages.append({"role": "user", "content":
                                 f"Tool result from search_archive (page {turn+1}):\n{chunk}\n\n"
                                 f"What is the access code for the {NAMES[turn+1]} vault? Reply with only the code."})
    except Exception as e:
        record({**base, "error": str(e)[:500]})
        log(f"  SESSION FAILED: {e}")
    finally:
        srv.stop()

def e2a():
    agent_session("nemotron-256k", NEMO, 262144, [], "off")
    agent_session("qwen36-256k", Q36, 262144, [], "off")
    agent_session("nex-256k", NEX, 262144, [], "off")
    agent_session("gemma26-256k", GEMMA26, 262144, [], "off")
    agent_session("qwen35-9b-1card-256k", Q35_9B, 262144, ["-dev", "CUDA0"], "off")
    agent_session("codernext-128k", CODERNEXT, 131072, [], "off")
    agent_session("next80-128k", NEXT80, 131072, [], "off")
    agent_session("gptoss-128k", GPTOSS, 131072, [], "auto", max_tokens=1024)

def e2b():
    agent_session("nemotron-64k-think", NEMO, 65536, [], "on", turns=6, max_tokens=1024)
    agent_session("nemotron-64k-think-preserve", NEMO, 65536, [], "on", turns=6, max_tokens=1024,
                  extra=["--reasoning-preserve"])
    agent_session("qwen36-64k-think", Q36, 65536, [], "on", turns=6, max_tokens=1024)
    agent_session("qwen36-64k-think-preserve", Q36, 65536, [], "on", turns=6, max_tokens=1024,
                  extra=["--reasoning-preserve"])
    agent_session("gemma26-64k-think", GEMMA26, 65536, [], "on", turns=6, max_tokens=1024)


# ---------------------------------------------------------------- E1 sustained load
SOAK_PROMPT = "Write a detailed, multi-part history of the telescope, from Galileo to space observatories."

def soak(label, model, dev_args, minutes):
    log(f"E1 soak {label}: {name_of(model)} for {minutes} min")
    start_temps = wait_cool(45, 900)
    csvp = os.path.join(LOGD, f"wave10-soak.{label}.{time.strftime('%Y%m%d-%H%M%S')}.gpu.csv")
    samp = Sampler(csvp)
    srv = Server(model, 4096, list(dev_args) + ["-np", "1", "-fa", "on"], f"soak-{label}")
    base = {"exp": "soak", "label": label, "model": name_of(model), "cards": 1 if dev_args else 2,
            "start_temps": start_temps, "gpu_csv": os.path.basename(csvp)}
    rates = []
    try:
        srv.start()
        t0 = time.time(); i = 0
        while time.time() - t0 < minutes * 60:
            r = post("/completion", {"prompt": SOAK_PROMPT, "n_predict": 512, "ignore_eos": True,
                                     "temperature": 0, "cache_prompt": False}, 900)
            t = r["timings"]
            rates.append(t["predicted_per_second"])
            record({**base, "i": i, "t_s": round(time.time() - t0, 1), "clock": time.strftime("%H:%M:%S"),
                    "predicted_tps": round(t["predicted_per_second"], 3), "predicted_n": t["predicted_n"]})
            i += 1
    except Exception as e:
        record({**base, "error": str(e)[:500]}); log(f"  SOAK FAILED: {e}")
    finally:
        srv.stop(); samp.stop()
    if rates:
        first = sum(rates[:3]) / len(rates[:3]); last = sum(rates[-5:]) / len(rates[-5:])
        g = summarize_gpu(csvp)
        record({**base, "summary": True, "requests": len(rates), "first3_tps": round(first, 3),
                "last5_tps": round(last, 3), "change_pct": round(100 * (last - first) / first, 1), "gpu": g})
        log(f"  {label}: first {first:.2f} -> last {last:.2f} t/s ({100*(last-first)/first:+.1f}%)  gpu={g}")

def e1():
    soak("27b-card0", Q38_27B, ["-dev", "CUDA0"], 25)
    soak("27b-card1", Q38_27B, ["-dev", "CUDA1"], 25)
    soak("27b-both", Q38_27B, [], 25)
    soak("nemotron-both", NEMO, [], 20)


# ---------------------------------------------------------------- E3 MTP at depth
SPEC_PROMPTS = [
    ("code", "Write a complete Python implementation of a red-black tree with insert, delete, and in-order traversal. Include type hints and docstrings."),
    ("prose", "Write a vivid, original short story about a lighthouse keeper who discovers something unexpected in the fog. Do not use cliches."),
    ("fact", "List the planets of the solar system in order from the sun. For each, give its diameter in km, orbital period, and number of moons, formatted as a markdown table."),
]

def mtp_depth(ctx, spec):
    label = f"mtp-{ctx//1024}k-{'n2' if spec else 'base'}"
    log(f"E3 {label}")
    extra = ["-np", "1", "-fa", "on", "-ctk", "f16", "-ctv", "f16"]
    if spec:
        extra += ["--spec-type", "draft-mtp", "-md", NEMO_MTP, "--spec-draft-n-max", "2", "-ngld", "999"]
    srv = Server(NEMO, ctx, extra, label)
    base = {"exp": "mtp_depth", "label": label, "model": name_of(NEMO), "ctx": ctx, "spec": spec}
    try:
        srv.start()
        doc, _, _, _, doc_tokens = build_document(ctx - 4096, needles=False)
        for kind, text in SPEC_PROMPTS:
            t0 = time.time()
            r = post("/completion", {"prompt": doc + "\n\n" + text, "n_predict": 256, "temperature": 0,
                                     "top_k": 1, "cache_prompt": True}, 4 * 3600)
            t = r["timings"]
            acc = (round(100 * t["draft_n_accepted"] / t["draft_n"], 1)
                   if t.get("draft_n") else None)
            record({**base, "kind": kind, "doc_tokens": doc_tokens, "prompt_n": t.get("prompt_n"),
                    "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
                    "predicted_n": t.get("predicted_n"), "predicted_tps": t.get("predicted_per_second"),
                    "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                    "accept_pct": acc, "wall_s": round(time.time() - t0, 1)})
            log(f"  {kind}: {t.get('predicted_per_second'):.2f} t/s  processed={t.get('prompt_n')} accept={acc}")
    except Exception as e:
        record({**base, "error": str(e)[:500]}); log(f"  MTP FAILED: {e}")
    finally:
        srv.stop()

def e3():
    for ctx in (131072, 262144):
        mtp_depth(ctx, False)
        mtp_depth(ctx, True)


# ---------------------------------------------------------------- E4 micro-batch at depth
def e4():
    log("E4: -ub sweep at 64k")
    for model in (NEMO, Q36):
        for ub in (512, 1024, 2048, 4096):
            run_bench(model, "wave10-ubatch", 2, ["-d", "65536", "-b", "4096", "-ub", str(ub),
                                                  "-ctk", "f16", "-ctv", "f16", "-fa", "on"], suite="agent")


# ---------------------------------------------------------------- E5 tokenizer efficiency
def e5():
    log("E5: tokens per 1,000 characters")
    eng = os.path.join(LOGD, "wave10-sample-english.txt")
    code = os.path.join(LOGD, "wave10-sample-code.txt")
    with open(eng, "w", encoding="utf-8") as f:
        f.write(open(WIKI_TEST, encoding="utf-8").read()[:500_000])
    with open(code, "w", encoding="utf-8") as f:
        for fn in sorted(os.listdir(ROOT + r"\scripts")):
            if fn.endswith((".ps1", ".py")):
                f.write(open(os.path.join(ROOT, "scripts", fn), encoding="utf-8", errors="replace").read() + "\n")
    samples = {"english": eng, "code": code}
    seen = set()
    for dp, _, fns in os.walk(M):
        for fn in sorted(fns):
            if not fn.endswith(".gguf") or fn.startswith(("mmproj", "mtp-", "eagle3")) or "DFlash" in fn:
                continue
            fam = fn.split(".gguf")[0]
            key = dp  # one file per repository folder is one tokenizer
            if key in seen: continue
            seen.add(key)
            path = os.path.join(dp, fn)
            row = {"exp": "tokenizer", "model": fam}
            for kind, sp in samples.items():
                nchar = len(open(sp, encoding="utf-8").read())
                t0 = time.time()
                r = subprocess.run([TOKENIZE, "-m", path, "-f", sp, "--show-count", "--log-disable", "--ids"],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                cnt = None
                for line in r.stdout.splitlines()[::-1]:
                    if "Total number of tokens" in line:
                        cnt = int(line.split(":")[-1].strip()); break
                row[f"{kind}_tokens"] = cnt
                row[f"{kind}_chars"] = nchar
                row[f"{kind}_tok_per_1k_chars"] = round(1000 * cnt / nchar, 1) if cnt else None
                row[f"{kind}_s"] = round(time.time() - t0, 1)
            record(row)
            log(f"  {fam}: english {row['english_tok_per_1k_chars']}  code {row['code_tok_per_1k_chars']} tok/1k chars")


# ---------------------------------------------------------------- E6 full-window sweep
# Every model, filled to its own maximum context: the trained window, or the largest one
# that allocates in 32 GB if that is smaller. llama-bench's -d fill is untimed, so it cannot
# say how long a full prompt takes to read. This drives llama-server instead and measures,
# at each fill level on the way up:
#   - time to first token for the whole prompt so far (cold-equivalent, see below)
#   - average prefill speed over the whole prompt, and the marginal speed of the last segment
#   - generation speed with that much context
# One fill per model: each level's prompt extends the previous one, so the server reuses the
# cached prefix and only reads the new segment. Cold-equivalent TTFT is the sum of the segment
# prefill times, which is the same work a single cold prompt does (checked against the E2
# turn-0 cold fills). Prompts are cut at newlines so the token boundary never moves.
W = [
    # tier 1 - long windows, the agent-relevant models
    ("nemotron-q4_0", NEMO, [1048576]),
    ("qwen36-35b-ud-q4_k_m", Q36, [262144]),
    ("nex-n2.5-mini", NEX, [262144]),
    ("gemma4-26b-a4b", GEMMA26, [262144]),
    ("qwen35-9b-q8_0", M + r"\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q8_0.gguf", [262144]),
    ("qwen3-coder-next", CODERNEXT, [262144, 196608, 131072]),
    ("qwen3-next-80b-instruct", NEXT80, [262144, 196608, 131072]),
    ("gpt-oss-20b", GPTOSS, [131072]),
    ("qwen38-2b-q8_0", Q38_2B, [262144]),
    ("gemma4-12b", M + r"\unsloth\gemma-4-12B-it-qat-GGUF\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", [262144, 196608, 131072]),
    ("llama31-8b-q8_0", M + r"\bartowski\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", [131072, 98304]),
    ("lfm25-2.6b-q8_0", M + r"\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-Q8_0.gguf", [131072]),
    ("ornith-1.5-35b", M + r"\ornith-ai\Ornith-1.5-35B-A3B-GGUF\Ornith-1.5-35B-Q4_K_M.gguf", [262144]),
    ("qwen3-coder-30b-f16kv", M + r"\unsloth\Qwen3-Coder-30B-A3B-Instruct-GGUF\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", [262144, 196608, 131072, 98304, 65536]),
    ("qwen36-35b-q6_k", M + r"\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q6_K.gguf", [262144, 196608, 131072, 98304]),
    ("qwen35-122b-iq1_m", M + r"\mradermacher\Qwen3.5-122B-A10B-i1-GGUF\Qwen3.5-122B-A10B.i1-IQ1_M.gguf", [262144, 196608, 131072, 98304]),
    # tier 2 - dense and 131k-trained models
    ("mistral-small-3.2-24b", M + r"\unsloth\Mistral-Small-3.2-24B-Instruct-2506-GGUF\Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf", [131072, 98304, 65536]),
    ("magistral-small-2509-q6_k", M + r"\bartowski\mistralai_Magistral-Small-2509-GGUF\mistralai_Magistral-Small-2509-Q6_K.gguf", [131072, 98304, 65536]),
    ("qwen38-27b-q4_0", Q38_27B, [262144, 196608, 131072]),
    ("qwen36-27b-q4_0", M + r"\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q4_0.gguf", [262144, 196608, 131072]),
    ("gemma4-31b-q4_0", M + r"\google\gemma-4-31B-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf", [262144, 196608, 131072]),
    ("olmo-3.1-32b-think", M + r"\bartowski\allenai_Olmo-3.1-32B-Think-GGUF\allenai_Olmo-3.1-32B-Think-Q5_K_M.gguf", [65536]),
    ("deepseek-r1-distill-32b", M + r"\bartowski\DeepSeek-R1-Distill-Qwen-32B-GGUF\DeepSeek-R1-Distill-Qwen-32B-Q5_K_M.gguf", [131072, 98304, 65536, 49152, 32768]),
    ("phi-4-q8_0", M + r"\unsloth\phi-4-GGUF\phi-4-Q8_0.gguf", [16384]),
    ("qwen3-32b-q6_k", M + r"\unsloth\Qwen3-32B-GGUF\Qwen3-32B-Q6_K.gguf", [40960, 32768, 24576, 16384]),
    ("seed-oss-36b-q6_k", M + r"\lmstudio-community\Seed-OSS-36B-Instruct-GGUF\Seed-OSS-36B-Instruct-Q6_K.gguf", [16384, 12288, 8192]),
    ("llama33-70b-q2_k_xl", M + r"\unsloth\Llama-3.3-70B-Instruct-GGUF\Llama-3.3-70B-Instruct-UD-Q2_K_XL.gguf", [16384, 12288, 8192]),
    # tier 3 - MLA models whose prefill collapses with depth (hours per window)
    ("glm-4.7-flash-q5_k_m", M + r"\bartowski\zai-org_GLM-4.7-Flash-GGUF\zai-org_GLM-4.7-Flash-Q5_K_M.gguf", [131072]),
    ("mistral-small-4-119b-iq1_m", M + r"\mradermacher\Mistral-Small-4-119B-2603-i1-GGUF\Mistral-Small-4-119B-2603.i1-IQ1_M.gguf", [131072]),
]
LEVELS = [4096, 16384, 32768, 65536, 131072, 262144, 524288]

def window_sweep(label, model, candidates):
    log(f"E6 {label}: candidates {candidates}")
    csvp = os.path.join(LOGD, f"wave10-window.{label}.{time.strftime('%Y%m%d-%H%M%S')}.gpu.csv")
    base = {"exp": "window", "label": label, "model": name_of(model)}
    srv, ctx = None, None
    # largest context that allocates
    for c in candidates:
        s = Server(model, c, ["-np", "1", "-fa", "on", "-ctk", "f16", "-ctv", "f16"], f"window-{c}")
        try:
            s.start(); srv, ctx = s, c; break
        except Exception as e:
            record({**base, "ctx_try": c, "loaded": False, "error": str(e)[-300:]})
            log(f"  ctx {c}: did not load"); s.stop()
    if not srv:
        log("  no candidate context loaded"); return
    samp = Sampler(csvp)
    base["ctx"] = ctx
    try:
        text = wiki()
        sample = text[:400_000]
        cpt = len(sample) / ntokens(sample)
        levels = [l for l in LEVELS if l <= ctx - 1024]
        top = ctx - 1024
        if not levels or top - levels[-1] > 0.05 * top:
            levels.append(top)
        cum_ms, prev_chars = 0.0, 0
        for lvl in levels:
            nchars = max(int(lvl * cpt * 0.98), prev_chars + 1000)
            for attempt in range(4):
                cut = text.rfind("\n", 0, nchars)
                prompt = text[:cut]
                n = ntokens(prompt)
                if n <= lvl or attempt == 3: break
                nchars = int(nchars * lvl / n * 0.995)
            prev_chars = len(prompt)
            t0 = time.time()
            try:
                r = post("/completion", {"prompt": prompt, "n_predict": 128, "ignore_eos": True,
                                         "temperature": 0, "cache_prompt": True}, 6 * 3600)
            except Exception as e:
                record({**base, "level": lvl, "error": str(e)[:300], "server_tail": srv.tail()})
                log(f"  level {lvl}: FAILED {e}"); break
            t = r["timings"]
            cum_ms += t["prompt_ms"]
            ctx_tokens = r.get("tokens_evaluated") or n
            row = {**base, "level": lvl, "context_tokens": ctx_tokens,
                   "segment_tokens": t["prompt_n"], "cache_n": t.get("cache_n"),
                   "segment_prefill_s": round(t["prompt_ms"] / 1000, 2),
                   "segment_prefill_tps": round(t["prompt_per_second"], 2),
                   "ttft_cold_s": round(cum_ms / 1000, 1),
                   "avg_prefill_tps": round(ctx_tokens / (cum_ms / 1000), 2),
                   "gen_tps": round(t["predicted_per_second"], 3), "gen_n": t["predicted_n"],
                   "wall_s": round(time.time() - t0, 1)}
            record(row)
            log(f"  {ctx_tokens:>8} tok: TTFT {cum_ms/1000:8.1f}s  prefill avg {row['avg_prefill_tps']:8.1f} "
                f"(last segment {row['segment_prefill_tps']:.1f}) t/s  gen {row['gen_tps']:6.2f} t/s")
    except Exception as e:
        record({**base, "error": str(e)[:500]}); log(f"  SWEEP FAILED: {e}")
    finally:
        srv.stop(); samp.stop()
        g = summarize_gpu(csvp)
        record({**base, "summary": True, "gpu": g})
        log(f"  vram peak: {', '.join(f'card{k} {v['mem_max_mib']} MiB' for k, v in sorted(g.items()))}")

def e6():
    for label, model, cands in W:
        if os.path.exists(model):
            window_sweep(label, model, cands)
        else:
            log(f"E6 {label}: missing file, skipped")


# ---------------------------------------------------------------- smoke test
def smoke():
    global OUT
    OUT = ROOT + r"\results\wave10-smoke.jsonl"
    log("SMOKE: agent session, small model, 16k")
    agent_session("smoke", Q38_2B, 32768, ["-dev", "CUDA1"], "off", turns=3)
    log("SMOKE: soak 1 min")
    soak("smoke", Q38_2B, ["-dev", "CUDA1"], 1)


def smoke_window():
    global OUT, LEVELS
    OUT = ROOT + r"\results\wave10-smoke.jsonl"
    LEVELS = [4096, 16384]
    window_sweep("smoke", Q38_2B, [65536, 32768])

EXPERIMENTS = {"e0": e0, "e2a": e2a, "e2b": e2b, "e1": e1, "e3": e3, "e4": e4, "e5": e5, "e6": e6,
               "smoke": smoke, "smoke_window": smoke_window}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["all"]
    if wanted == ["all"]:
        wanted = ["e0", "e2a", "e2b", "e1", "e3", "e4", "e5"]
    # Hold off idle sleep for the life of this process only (ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
    # it is released automatically when the process exits and changes no power settings.
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    log(f"WAVE 10 start: {wanted}")
    for w in wanted:
        try:
            EXPERIMENTS[w]()
        except Exception as e:
            log(f"EXPERIMENT {w} CRASHED: {e}")
    log("WAVE 10 COMPLETE")
