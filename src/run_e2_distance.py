"""run_e2_distance.py — E2: recall vs context length (state-lossiness probe, RULER-style)
and whether a streaming write gate compensates.

Design (validated learnable at L=64/256; L=512 gated by a diag):
  block MQAR, P=16 pairs in the first 32 tokens, queries in the tail; only answers scored.
  contexts L in {64, 256, 512}; arms gla-base | gla-sg2 (+ gd variants at L=256, 1 seed).

Rationale: with P fixed, the average fact-query distance grows with L, so accuracy-vs-L is a
direct probe of how much the fixed-size state loses with distance.

Note (negative result, recorded): a gap-controlled variant (per-pair filler gaps) failed to learn
at gaps >= 16 at this scale (acc ~chance), so context length is used as the distance proxy.

Pre-registered break conditions:
  * state lossy if base accuracy falls with L;
  * gate helps if it raises long-context accuracy without hurting short-context accuracy.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PY = sys.executable
OUT = os.path.join(RES, "e2_distance.jsonl")

# Fixed TOKEN budget ~7.4M per run (avoids the tokens-vs-steps confound across lengths):
#   steps = 7.4M / (batch * L)
CTX = [(64, 3600, 32), (256, 900, 32), (512, 600, 24)]

def done():
    tags = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: tags.add(json.loads(l)["tag"])
            except Exception: pass
    return tags

def cfg(L, steps, batch):
    return ["--task", "mqar", "--pairs", "16", "--mv", "256",
            "--width", "64", "--layers", "2", "--heads", "4", "--chunk", "16",
            "--L", str(L), "--steps", str(steps), "--batch", str(batch),
            "--lr", "0.06", "--elr", "0.04"]

def run(tag, extra):
    cmd = [PY, "-I", os.path.join(HERE, "memory_lm.py"), "--out", OUT, "--tag", tag, "--dev", "cuda"] + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = False; s = {}
    if r.returncode == 0:
        try:
            s = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1]); ok = True
        except Exception: pass
    print(("OK  " if ok else "ERR ") + f"{tag}  {json.dumps(s.get('scores', s))[:220]}  {time.time()-t0:5.0f}s", flush=True)
    if not ok: print(r.stderr[-600:], flush=True)

def main():
    jobs = []
    for L, steps, batch in CTX:
        for seed in (0, 1):
            jobs.append((f"E2-L{L}-gla-base-s{seed}", cfg(L, steps, batch) + ["--mixer", "gla", "--gate", "none", "--seed", str(seed)]))
            jobs.append((f"E2-L{L}-gla-sg2-s{seed}", cfg(L, steps, batch) + ["--mixer", "gla", "--gate", "sg2", "--seed", str(seed)]))
    for arm, extra in (("gd-base", []), ("gd-sgkE", ["--gate", "sgk", "--gate_erase", "1"])):
        jobs.append((f"E2-L256-{arm}-s0", cfg(256, 900, 32) + ["--mixer", "gd", "--seed", "0"] + extra))
    d = done()
    todo = [(t, e) for t, e in jobs if t not in d]
    print(f"E2: {len(todo)}/{len(jobs)} runs to do", flush=True)
    for t, e in todo:
        run(t, e)

if __name__ == "__main__":
    main()