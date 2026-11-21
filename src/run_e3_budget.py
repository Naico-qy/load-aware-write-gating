"""run_e3_budget.py — E3: fixed-budget allocation between state width and table capacity.

Total parameters held roughly constant (~8.5-9.7M). Configurations sweep the split:
  (d=96,  N=65536)  (d=128, N=32768)  (d=160, N=16384)  (d=192, N=8192)  (d=224, plain)
i.e. as the state (width) grows, the table shrinks and its collision rate rises.

Pre-registered break conditions:
  * A U-shape (interior optimum) supports the trade-off claim.
  * Monotone improvement toward either end falsifies the simple capacity-vs-training story
    at this budget — report the curve as-is.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
PY = sys.executable
OUT = os.path.join(RES, "e3_budget.jsonl")

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
    print(("OK  " if ok else "ERR ") + f"{tag}  {json.dumps(s.get('scores', s))[:200]}  par={s.get('params')} body={s.get('body_params')} tab={s.get('table_params')} cap={s.get('capacity')}  {time.time()-t0:5.0f}s", flush=True)
    if not ok: print(r.stderr[-600:], flush=True)

COMMON = ["--task", "text",
          "--data", os.path.join(DATA, "tok16k_train.bin"),
          "--val_data", os.path.join(DATA, "tok16k_val.bin"),
          "--L", "256", "--batch", "32", "--mixer", "gla",
          "--lr", "0.04", "--elr", "0.006", "--warm", "150", "--cool", "0.2", "--steps", "1630",
          "--seed", "0"]

def main():
    # Fixed total budget ~8.5-9.7M params (body = emb/head 32768d + blocks 24d^2; table = 2 layers * N * d).
    # As the state width grows the table shrinks, so updates-per-row rises 406 -> 3248.
    jobs = [
        ("E3-d96-N32768",  COMMON + ["--width", "96",  "--layers", "2", "--heads", "4", "--ve", "0,1", "--vebig", "32768",  "--tlr", "0.02"]),
        ("E3-d128-N16384", COMMON + ["--width", "128", "--layers", "2", "--heads", "4", "--ve", "0,1", "--vebig", "16384",  "--tlr", "0.02"]),
        ("E3-d160-N8192",  COMMON + ["--width", "160", "--layers", "2", "--heads", "4", "--ve", "0,1", "--vebig", "8192",   "--tlr", "0.02"]),
        ("E3-d192-N4096",  COMMON + ["--width", "192", "--layers", "2", "--heads", "4", "--ve", "0,1", "--vebig", "4096",   "--tlr", "0.02"]),
        ("E3-d224-plain",  COMMON + ["--width", "224", "--layers", "2", "--heads", "4"]),
    ]
    d = done()
    todo = [(t, e) for t, e in jobs if t not in d]
    print(f"E3: {len(todo)}/{len(jobs)} runs to do", flush=True)
    for t, e in todo:
        run(t, e)

if __name__ == "__main__":
    main()