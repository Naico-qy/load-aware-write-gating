"""analyze.py — readout for E1/E2/E3.

E1: word-lead per bucket across bigram table sizes (does the crossover move with collision rate?)
E2: recall accuracy per distance bucket per arm (is the state lossy? does the gate help?)
E3: fixed-budget allocation curve (val vs body/table split)
"""
import json, os, sys, re
sys.stdout.reconfigure(encoding="utf-8")

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
B = ("pc0", "pc1", "pc10", "pc100", "pc1000")

def load(name):
    p = os.path.join(RES, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def mean(xs): return sum(xs) / len(xs) if xs else None

def e1():
    rows = load("e1_collision.jsonl")
    if not rows:
        print("[E1] no data yet"); return
    print("\n===== E1 : hash-collision sweep (text LM, 25M tokens) =====")
    print(f"{'tag':20s} {'params':>10s} {'body':>10s} {'tab':>10s} {'rows':>7s} {'upd/row':>8s}  " + "  ".join(f"{b:>7s}" for b in B) + "  val_known")
    for r in sorted(rows, key=lambda x: x["tag"]):
        s = r["scores"]; c = r.get("capacity") or {}
        vals = "  ".join(f"{s.get(b, float('nan')):7.4f}" for b in B)
        print(f"{r['tag']:20s} {r['params']:>10d} {r.get('body_params',0):>10d} {r.get('table_params',0):>10d} "
              f"{str(c.get('rows')):>7s} {str(c.get('updates_per_row')):>8s}  {vals}  {s.get('val_known')}")
    # word-lead per bucket, bigram vs word, same seed
    word = {r["tag"].rsplit("-s",1)[0]: r for r in rows if r["tag"].startswith("E1-word")}
    for r in rows:
        if not r["tag"].startswith("E1-bigram"): continue
        N = re.search(r"bigram(\d+)", r["tag"]).group(1)
        key_w = "E1-word"
        if key_w not in word: continue
        w = word[key_w]["scores"]; b = r["scores"]
        lead = "  ".join(f"{w.get(x,0)-b.get(x,0):+.4f}" for x in B)
        print(f"  word-lead vs N={N:>6s}: {lead}  (overall {w.get('val_known',0)-b.get('val_known',0):+.4f})")

def e2():
    rows = load("e2_distance.jsonl")
    if not rows:
        print("\n[E2] no data yet"); return
    print("\n===== E2 : distance-controlled recall (state lossiness) =====")
    keys = sorted({k for r in rows for k in r["scores"] if k.startswith("acc_g")}, key=lambda z: int(z[5:]))
    print(f"{'tag':18s} {'loss':>7s} {'acc':>6s}  " + "  ".join(f"{k:>7s}" for k in keys))
    for r in sorted(rows, key=lambda x: x["tag"]):
        s = r["scores"]
        print(f"{r['tag']:18s} {s.get('loss',float('nan')):7.3f} {s.get('acc',float('nan')):6.3f}  " +
              "  ".join(f"{s.get(k, float('nan')):7.3f}" for k in keys))

def e3():
    rows = load("e3_budget.jsonl")
    if not rows:
        print("\n[E3] no data yet"); return
    print("\n===== E3 : fixed-budget allocation (state width vs table rows) =====")
    print(f"{'tag':16s} {'params':>9s} {'body':>9s} {'tab':>9s} {'rows':>6s} {'upd/row':>8s} {'val':>7s} {'val_known':>9s}")
    for r in sorted(rows, key=lambda x: x.get("body_params", 0)):
        s = r["scores"]; c = r.get("capacity") or {}
        print(f"{r['tag']:16s} {r['params']:>9d} {r.get('body_params',0):>9d} {r.get('table_params',0):>9d} "
              f"{str(c.get('rows')):>6s} {str(c.get('updates_per_row')):>8s} {s.get('val',float('nan')):7.4f} {s.get('val_known',float('nan')):9.4f}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "e1"): e1()
    if which in ("all", "e2"): e2()
    if which in ("all", "e3"): e3()