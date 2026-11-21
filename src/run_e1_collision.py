"""run_e1_collision.py — E1: hash-collision sweep (main experiment of the line).

Question: under strong hash collision, is the word-vs-n-gram table crossover governed by
per-key sightings (low-collision regime) or by per-row update count (realistic regime)?

Design (text LM, FineWeb-16k, 25M tokens, d=128, 2 layers, 1630 steps):
  arms: plain | word table (V rows) | bigram table with rows N in {8192, 32768, 131072, 262144}
  The bigram table maps (prev,cur) -> row via hash; rows/unique-pairs ratio sets the collision rate.
  Capacity instrumentation records updates-per-row for each N.

Pre-registered break conditions:
  * H3 supported if the word-lead on rare buckets (pc0/pc1/pc10) rises toward positive as N grows
    (i.e. bigram loses its advantage once updates/row falls below the training threshold).
  * H3 falsified in range if word-lead stays negative for every N (then report the collision-induced
    advantage as robust across 800-3200 updates/row).

Usage: python run_e1_collision.py [seed_extra 1|0]
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
PY = sys.executable
OUT = os.path.join(RES, "e1_collision.jsonl")

def done():
    tags = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: tags.add(json.loads(l)["tag"])
            except Exception: pass
    return tags

def run(tag, extra):
    cmd = [PY, "-I", os.path.join(HERE, "memory_lm.py"), "--out", OUT, "--tag", tag, "--dev", "cuda"] + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = False; s = {}
    if r.returncode == 0:
        try:
            s = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
            ok = True
        except Exception: pass
    print(("OK  " if ok else "ERR ") + f"{tag}  {json.dumps(s.get('scores', s))[:200]}  cap={s.get('capacity')}  {time.time()-t0:5.0f}s", flush=True)
    if not ok: print(r.stderr[-600:], flush=True)
    return ok

BASE = ["--task", "text",
        "--data", os.path.join(DATA, "tok16k_train.bin"),
        "--val_data", os.path.join(DATA, "tok16k_val.bin"),
        "--width", "128", "--layers", "2", "--heads", "4", "--L", "256", "--batch", "32",
        "--mixer", "gla", "--lr", "0.04", "--elr", "0.006", "--warm", "150", "--cool", "0.2",
        "--steps", "1630"]
TAB = ["--ve", "0,1", "--tlr", "0.02"]

def main():
    extra_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    jobs = []
    jobs.append(("E1-plain-s0", BASE[:]))
    jobs.append(("E1-word-s0", BASE + TAB))
    for N in (8192, 32768, 131072, 262144):
        jobs.append((f"E1-bigram{N}-s0", BASE + TAB + ["--vebig", str(N)]))
    if extra_seeds:
        for seed in (1, 2):
            jobs.append((f"E1-word-s{seed}", BASE + TAB + ["--seed", str(seed)]))
            jobs.append((f"E1-bigram32768-s{seed}", BASE + TAB + ["--vebig", "32768", "--seed", str(seed)]))
    d = done()
    todo = [(t, e) for t, e in jobs if t not in d]
    print(f"E1: {len(todo)}/{len(jobs)} runs to do", flush=True)
    for t, e in todo:
        if "--seed" not in e: e = e + ["--seed", "0"]
        run(t, e)

if __name__ == "__main__":
    main()