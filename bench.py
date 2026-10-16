#!/usr/bin/env python3
"""prefill-vs-decode: every number in the Ring Zero video "Prefill vs Decode Explained" from one script.

Talks to Ollama's native API (it reports reading and writing time separately), standard library only.

    python3 bench.py all --model qwen3.5:9b            # every test below, ~15-25 min on an M4 Pro
    python3 bench.py hook  --model qwen3.5:9b          # one test at a time: hook | depth | quiz | cache | load

Fair-test rules (same as race.py): plugged in, Low Power Mode off, other apps closed, thinking OFF,
one untimed warm-up, a unique first line per run (defeats the prompt cache), median of N runs, 5 s rest.
Results: results/<test>.csv, results/summary.json, and a TEST block to paste into the video's episode.py.
"""
import argparse, csv, json, os, random, statistics, sys, time, urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
NS = 1e9
REST = float(os.environ.get("BENCH_REST", 5))   # seconds between runs

# ---------------------------------------------------------------- test document (deterministic, offline)
SUBJ = ["The supplier contract", "The warehouse lease", "Our cloud budget", "The hiring plan", "The pricing model",
        "Customer churn", "The support backlog", "The Q3 launch", "The data pipeline", "The security audit",
        "The partner programme", "Inventory turnover", "The mobile release", "Vendor onboarding", "The training budget"]
VERB = ["depends on", "has drifted from", "is tied to", "was renegotiated against", "still assumes", "puts pressure on",
        "was signed before", "quietly increases", "is reviewed alongside", "runs ahead of"]
OBJ = ["the original forecast", "a three-year renewal", "last year's volumes", "the regional targets", "a fixed exchange rate",
       "the new compliance rules", "the board's spending cap", "two key accounts", "the shared services team",
       "an automatic price review", "the delayed migration", "the holiday peak"]
TAIL = ["and nobody has owned the follow-up.", "which section four explains in detail.", "so the numbers need a second look.",
        "according to the finance team's notes.", "unless the March review changes it.", "though the risk is still small.",
        "and the next renewal is automatic.", "which matters more than it looks.", "as flagged in the last audit."]


def make_doc(words, seed=7):
    rnd, out, n, para = random.Random(seed), [], 0, []
    while n < words:
        s = f"{rnd.choice(SUBJ)} {rnd.choice(VERB)} {rnd.choice(OBJ)}, {rnd.choice(TAIL)}"
        para.append(s); n += len(s.split())
        if len(para) == 6:
            out.append(" ".join(para)); para = []
    out.append(" ".join(para))
    return "\n\n".join(p for p in out if p)


QUESTION = "What is the main risk in this report? Answer in about 200 words."
WRITE_TASK = "Write a 300-word story about a lighthouse keeper who finds a message in a bottle. Plain prose."


# ---------------------------------------------------------------- one request, timed two ways
def chat(model, messages, num_predict, num_ctx, keep_alive="30m"):
    """Streams /api/chat. Returns Ollama's own timings (s) plus a wall-clock time to first visible token."""
    body = {"model": model, "messages": messages, "stream": True, "think": False, "keep_alive": keep_alive,
            "options": {"temperature": 0, "seed": 42, "num_predict": num_predict, "num_ctx": num_ctx}}
    req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, t_first, text, thinking, final = time.perf_counter(), None, [], 0, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.strip():
                continue
            m = json.loads(raw)
            if "error" in m:
                raise RuntimeError(m["error"])
            msg = m.get("message", {})
            if msg.get("thinking"):
                thinking += 1
            if msg.get("content"):
                if t_first is None:
                    t_first = time.perf_counter()
                text.append(msg["content"])
            if m.get("done"):
                final = m
    t_end = time.perf_counter()
    g = lambda k: final.get(k, 0) / NS
    res = {"wall_ttft_s": (t_first or t_end) - t0, "wall_total_s": t_end - t0,
           "load_s": g("load_duration"), "read_s": g("prompt_eval_duration"), "write_s": g("eval_duration"),
           "in_tok": final.get("prompt_eval_count", 0), "out_tok": final.get("eval_count", 0), "thinking_chunks": thinking}
    res["read_tps"] = res["in_tok"] / res["read_s"] if res["read_s"] else 0.0
    res["write_tps"] = res["out_tok"] / res["write_s"] if res["write_s"] else 0.0
    res["answer"] = "".join(text)
    return res


def runs(label, n, fn):
    rows = []
    for i in range(1, n + 1):
        r = fn(i)
        r.update(test=label, run=i)
        rows.append(r)
        print(f"  {label} run {i}: wait {r['wall_ttft_s']:.2f}s | read {r['in_tok']} tok in {r['read_s']:.2f}s "
              f"({r['read_tps']:.0f}/s) | write {r['out_tok']} tok in {r['write_s']:.2f}s ({r['write_tps']:.1f}/s)"
              + ("  !! THINKING DETECTED" if r["thinking_chunks"] else ""), flush=True)
        time.sleep(REST)
    return rows


def med(rows, k):
    return statistics.median(r[k] for r in rows)


def save(name, rows):
    os.makedirs(OUT, exist_ok=True)
    keys = [k for k in rows[0] if k != "answer"]
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def warm(a):
    print(f"warm-up ({a.model}) ...", flush=True)
    chat(a.model, [{"role": "user", "content": "Say OK."}], 4, a.num_ctx)


def u(i, tag):  # unique first line so every run pays the full read
    return f"[{tag} run {i} · {time.time_ns()}]\n"


# ---------------------------------------------------------------- the tests
def t_hook(a, S):
    """Sections 1, 3, 4, 10: 5,000-word paste + one question."""
    doc = make_doc(a.doc_words)
    rows = runs("hook", a.runs, lambda i: chat(a.model, [{"role": "user", "content": u(i, "hook") + doc + "\n\n" + QUESTION}],
                                                a.answer_tokens, a.num_ctx))
    save("hook", rows)
    S["hook"] = {k: med(rows, k) for k in ("wall_ttft_s", "read_s", "write_s", "in_tok", "out_tok", "read_tps", "write_tps")}
    S["hook"]["doc_words"] = a.doc_words


def t_depth(a, S):
    """Section 5: reading speed vs prompt length."""
    S["depth"] = {}
    allrows = []
    for words in (400, 1500, 6000, 12000):
        doc = make_doc(words, seed=words)
        rows = runs(f"depth_{words}w", max(3, a.runs - 2),
                    lambda i: chat(a.model, [{"role": "user", "content": u(i, "depth") + doc + "\n\nReply with one word: OK."}], 8, a.num_ctx))
        allrows += rows
        S["depth"][str(words)] = {"in_tok": med(rows, "in_tok"), "read_tps": med(rows, "read_tps")}
    save("depth", allrows)


def t_quiz(a, S):
    """Section 8 + Short B: read ~2,000 words vs write ~200 words."""
    doc = make_doc(2000, seed=2000)
    r_rows = runs("quiz_read", a.runs, lambda i: chat(a.model, [{"role": "user", "content": u(i, "quizr") + doc + "\n\nReply with one word: OK."}], 1, a.num_ctx))
    w_rows = runs("quiz_write", a.runs, lambda i: chat(a.model, [{"role": "user", "content": u(i, "quizw") + WRITE_TASK}], 270, a.num_ctx))
    save("quiz", r_rows + w_rows)
    S["quiz"] = {"read_tok": med(r_rows, "in_tok"), "read_s": med(r_rows, "read_s"),
                 "write_tok": med(w_rows, "out_tok"), "write_s": med(w_rows, "write_s")}
    if S["quiz"]["write_tok"] < 250:
        print("  note: the model stopped early; writing time is for fewer than 270 tokens", flush=True)


def t_cache(a, S):
    """Sections 9-10: same document, second question (cache hit), then a changed first line (cache miss)."""
    doc = make_doc(a.doc_words)
    rows = []
    for i in range(1, a.runs + 1):
        first = u(i, "cache") + doc + "\n\n" + QUESTION
        r1 = chat(a.model, [{"role": "user", "content": first}], a.answer_tokens, a.num_ctx)
        hist = [{"role": "user", "content": first}, {"role": "assistant", "content": r1["answer"]},
                {"role": "user", "content": "And what should we do about it?"}]
        r2 = chat(a.model, hist, a.answer_tokens, a.num_ctx)
        hist[0] = {"role": "user", "content": "[edited] " + first}
        r3 = chat(a.model, hist, a.answer_tokens, a.num_ctx)
        for tag, r in (("first", r1), ("followup", r2), ("changed_start", r3)):
            r.update(test=f"cache_{tag}", run=i); rows.append(r)
        print(f"  cache run {i}: first {r1['wall_ttft_s']:.2f}s | follow-up {r2['wall_ttft_s']:.2f}s "
              f"({r2['in_tok']} tok read) | changed start {r3['wall_ttft_s']:.2f}s", flush=True)
        time.sleep(REST)
    save("cache", rows)
    pick = lambda t, k: statistics.median(r[k] for r in rows if r["test"] == t)
    S["cache"] = {"first_s": pick("cache_first", "wall_ttft_s"), "followup_s": pick("cache_followup", "wall_ttft_s"),
                  "followup_read_s": pick("cache_followup", "read_s"), "followup_in_tok": pick("cache_followup", "in_tok"),
                  "changed_start_s": pick("cache_changed_start", "wall_ttft_s")}


def t_load(a, S):
    """Section 5 'three things in the pause': cold load vs warm."""
    msg = [{"role": "user", "content": "Say OK."}]
    chat(a.model, msg, 1, a.num_ctx, keep_alive=0)              # unload
    time.sleep(3)
    cold = chat(a.model, msg, 4, a.num_ctx)
    warm_ = chat(a.model, msg, 4, a.num_ctx)
    save("load", [dict(cold, test="cold", run=1), dict(warm_, test="warm", run=1)])
    S["load"] = {"cold_load_s": cold["load_s"], "warm_load_s": warm_["load_s"]}


TESTS = {"hook": t_hook, "depth": t_depth, "quiz": t_quiz, "cache": t_cache, "load": t_load}


# ---------------------------------------------------------------- TEST block for episode.py
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def say(n):
    """Spoken form for Kokoro: 6700 -> 'six thousand seven hundred', 0.4 -> 'zero point four'."""
    if isinstance(n, float) and n != int(n):
        i, d = f"{n:.1f}".split(".")
        return f"{say(int(i))} point {ONES[int(d)]}"
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + say(n % 100))
    th, rest = divmod(n, 1000)
    return say(th) + " thousand" + ("" if rest == 0 else (" and " if rest < 100 else " ") + say(rest))


def rnd(x, step):
    return int(round(x / step) * step)


def test_block(S, a):
    h, q, d, c = S.get("hook"), S.get("quiz"), S.get("depth"), S.get("cache")
    L = ["TEST = dict(", "    placeholder=False,"]
    if h:
        rs, ws = round(h["wall_ttft_s"]), round(h["write_s"])
        ratio = round(h["read_tps"] / h["write_tps"]) if h["write_tps"] else 0
        doc, ans = rnd(h["in_tok"], 100), rnd(h["out_tok"], 10)
        L += [f"    read_s={rs}, read_spoken={say(rs)!r},",
              f"    write_s={ws}, write_spoken={say(ws)!r},",
              f"    chip={a.chip!r},", f"    model={a.label or a.model!r},", f"    paste=\"{a.doc_words:,}-word paste\",",
              f"    read_tps={rnd(h['read_tps'], 10)}, read_tps_spoken={say(rnd(h['read_tps'], 10))!r},",
              f"    write_tps={round(h['write_tps'])}, write_tps_spoken={say(round(h['write_tps']))!r},",
              f"    ratio={ratio}, ratio_spoken={say(ratio)!r},",
              f"    doc_tok={doc}, doc_tok_spoken={say(doc)!r},",
              f"    ans_tok={ans}, ans_tok_spoken={say(ans)!r},",
              f"    inout={round(doc / ans) if ans else 0}, inout_spoken={say(round(doc / ans) if ans else 0)!r},",
              f"    load_s={S.get('load', {}).get('warm_load_s', 0):.2f}, prompt_eval_s={h['read_s']:.1f}, eval_s={h['write_s']:.1f},"]
    if d:
        keys = sorted(d, key=int)
        short, long_ = d[keys[0]], d[keys[-1]]
        lt = rnd(long_["in_tok"], 1000)
        L += [f"    short_tps={rnd(short['read_tps'], 10)}, short_tps_spoken={say(rnd(short['read_tps'], 10))!r},",
              f"    long_tok={lt}, long_tok_spoken={say(lt)!r},",
              f"    long_tps={rnd(long_['read_tps'], 10)}, long_tps_spoken={say(rnd(long_['read_tps'], 10))!r},"]
    if q:
        qr, qw = round(q["read_s"]), round(q["write_s"])
        if qr == qw:
            print("note: quiz times round to the same second; the video will call it for", "A" if q["read_s"] > q["write_s"] else "B")
        L += [f"    quiz_read_s={qr if qr != qw else round(q['read_s'], 1)}, quiz_read_spoken={say(qr)!r},",
              f"    quiz_write_s={qw if qr != qw else round(q['write_s'], 1)}, quiz_write_spoken={say(qw)!r},"]
    if c:
        cs = round(c["followup_s"], 1)
        L += [f"    cache_s={cs}, cache_spoken={say(cs)!r},"]
    L.append(")")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["all"] + list(TESTS))
    ap.add_argument("--model", default="qwen3.5:9b", help="Ollama tag, e.g. qwen3.5:9b (GGUF) or an -nvfp4 / -mlx tag")
    ap.add_argument("--label", default="Qwen3.5-9B", help="how the model is named on screen")
    ap.add_argument("--chip", default="M4 Pro", help="how the Mac is named on screen")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--doc-words", type=int, default=5000)
    ap.add_argument("--answer-tokens", type=int, default=270)
    ap.add_argument("--num-ctx", type=int, default=32768, help="must fit the longest prompt (~16K tokens)")
    a = ap.parse_args()
    path = os.path.join(OUT, "summary.json")
    S = json.load(open(path)) if os.path.exists(path) else {}
    S["meta"] = {"model": a.model, "chip": a.chip, "host": HOST, "runs": a.runs, "date": time.strftime("%Y-%m-%d %H:%M")}
    try:
        S["meta"]["ollama"] = json.load(urllib.request.urlopen(HOST + "/api/version", timeout=5)).get("version")
    except Exception as e:
        sys.exit(f"Can't reach Ollama at {HOST} ({e}). Start the Ollama app first.")
    warm(a)
    for name in (TESTS if a.test == "all" else [a.test]):
        print(f"\n== {name}: {TESTS[name].__doc__}", flush=True)
        TESTS[name](a, S)
        os.makedirs(OUT, exist_ok=True)
        json.dump(S, open(path, "w"), indent=1)
    block = test_block(S, a)
    open(os.path.join(OUT, "TEST_block.py"), "w").write(block + "\n")
    print("\n" + "=" * 70 + "\nPaste into episode.py (also saved to results/TEST_block.py):\n\n" + block)


if __name__ == "__main__":
    sys.exit(main())
