"""run_e4_load.py — E4: are delta-rule write gates load-blind? (diagnosis + our fix)

Motivation (measured): E2b shows recall collapses with associative load (P=16/32/64 ->
0.978/0.867/0.444) while distance barely matters. GDN-2-style gates are token-conditioned only
("memory-blind" per CARVE) — their policy cannot shift with how full the memory is. Our method
conditions the gate logits on two FREE per-token statistics of the read-back:
    cs = cos(S_t^pre k_t, v_t)   (read-back confidence / collision indicator)
    oc = ||S_t^pre k_t|| / ||v_t|| (occupancy along the write direction)
both computable from quantities delta-rule kernels already produce. Zero extra parameters beyond
4 scalars per head; zero-init => bit-identical to the unconditioned gate at init.

Arms (delta family, L=256, block MQAR):
  gd         scalar-tie baseline (erase == write)
  gdn2       decoupled erase/write, token-conditioned (GDN-2 style; per-head scalar gates here)
  gdn2+stat  + read-back stats in gate logits (ours)
  gdn2+sgkE+stat + sgk multiplicative gate + erase sync (ours, full)
Loads P in {16, 32, 48, 64}; seeds 0,1 (all arms at all loads for the headline; gate_probe on).

Usage: python run_e4_load.py [quick]   (quick = P in {16, 64} only, 16 runs)
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PY = sys.executable
OUT = os.path.join(RES, "e4_load.jsonl")

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
            s = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1]); ok = True
        except Exception: pass
    print(("OK  " if ok else "ERR ") + f"{tag}  {json.dumps(s.get('scores', s))[:160]}  probe={json.dumps(s.get('gate_probe', {}))[:180]}  {time.time()-t0:5.0f}s", flush=True)
    if not ok: print(r.stderr[-600:], flush=True)

CFG = ["--task", "mqar", "--mv", "256", "--width", "64", "--layers", "2", "--heads", "4",
       "--chunk", "16", "--L", "256", "--steps", "1200", "--batch", "32",
       "--lr", "0.06", "--elr", "0.04", "--gate_probe", "1"]

def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    loads = (16, 64) if quick else (16, 32, 48, 64)
    arms = [
        ("gd",         "--mixer gd --gate none"),
        ("gdn2",       "--mixer gdn2 --gate none"),
        ("gdn2-stat",  "--mixer gdn2 --gate none --gate_stat 1"),
        ("gdn2-sgkE-stat", "--mixer gdn2 --gate sgk --gate_erase 1 --gate_stat 1"),
    ]
    seeds = (0, 1)
    jobs = []
    for P in loads:
        for name, flags in arms:
            for seed in seeds:
                jobs.append((f"E4-P{P}-{name}-s{seed}",
                             CFG + ["--pairs", str(P), "--seed", str(seed)] + flags.split()))
    d = done()
    todo = [(t, e) for t, e in jobs if t not in d]
    print(f"E4: {len(todo)}/{len(jobs)} runs to do", flush=True)
    for t, e in todo:
        run(t, e)

if __name__ == "__main__":
    main()