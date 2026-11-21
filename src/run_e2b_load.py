"""run_e2b_load.py — E2b: recall vs associative load at fixed context (capacity-limit probe).

Motivation: E2 showed accuracy is nearly flat in context length at P=16 (0.982 at L=64 vs 0.978
at L=256). That suggests the lossy-compressor failure is driven by LOAD (associations per unit of
state capacity), not distance. This probe varies P at fixed L=256.

Arms: gla-base at P in {16, 32, 64}, seed 0 (+1 seed at P=32), 7.4M tokens each.

Pre-registered break conditions:
  * capacity limit supported if accuracy falls with P and approaches chance at the largest P
    (chance = 1/(mv/2) = 1/128 = 0.0078).
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PY = sys.executable
OUT = os.path.join(RES, "e2b_load.jsonl")

def done():
    tags = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: tags.add(json.loads(l)["tag"])
            except Exception: pass
    return tags

def cfg(P):
    return ["--task", "mqar", "--pairs", str(P), "--mv", "256", "--width", "64", "--layers", "2",
            "--heads", "4", "--chunk", "16", "--L", "256", "--steps", "900", "--batch", "32",
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
    print(("OK  " if ok else "ERR ") + f"{tag}  {json.dumps(s.get('scores', s))[:200]}  {time.time()-t0:5.0f}s", flush=True)
    if not ok: print(r.stderr[-500:], flush=True)

def main():
    jobs = [(f"E2b-P{P}-s0", cfg(P) + ["--mixer", "gla", "--gate", "none", "--seed", "0"]) for P in (16, 32, 64)]
    jobs.append(("E2b-P32-s1", cfg(32) + ["--mixer", "gla", "--gate", "none", "--seed", "1"]))
    d = done()
    todo = [(t, e) for t, e in jobs if t not in d]
    print(f"E2b: {len(todo)}/{len(jobs)} runs", flush=True)
    for t, e in todo:
        run(t, e)

if __name__ == "__main__":
    main()
