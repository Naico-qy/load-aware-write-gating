"""make_figures.py — publication figures from the raw experiment records (e4_load, e2b_load, e4_probe_fill, e1_collision, e2_distance).
Outputs PNG (300 dpi) into repo_root/assets/."""
import json, os, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ASSETS = os.path.join(HERE, "..", "assets")
os.makedirs(ASSETS, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300})

def load(f):
    p = os.path.join(RES, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def dedupe(rows):
    seen = {}
    for r in rows: seen[r["tag"]] = r
    return seen

# ---------------- Figure 1: load-collapse ladder ----------------
def fig_ladder():
    rows = dedupe(load("e4_load.jsonl"))
    g = {}
    for t, r in rows.items():
        import re
        m = re.match(r"E4-P(\d+)-([a-z2EA+-]+?)-s\d", t)
        if m: g.setdefault((int(m.group(1)), m.group(2)), []).append(r["scores"]["acc"])
    loads = sorted({P for P, _ in g})
    arms = [("gd", "Scalar-tied gates (pre-GDN-2)", "#9aa5b1", "o"),
            ("gdn2", "GDN-2 style: decoupled, token-conditioned", "#4c78a8", "s"),
            ("gdn2-stat", "+ read-back stats (ours)", "#e45756", "^"),
            ("gdn2-sgkE-stat", "+ erase-synced streaming erase gate (ours, full)", "#54a24b", "D")]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key, label, color, marker in arms:
        xs, ys = [], []
        for P in loads:
            vals = g.get((P, key))
            if vals:
                xs.append(P); ys.append(st.mean(vals))
        ax.plot(xs, ys, marker=marker, color=color, label=label, markersize=6, linewidth=1.6)
        for x, y in zip(xs, ys):
            vs = g.get((x, key))
            if vs and len(vs) > 1:
                ax.errorbar(x, y, yerr=st.stdev(vs), color=color, capsize=3, linewidth=0)
    ax.axhline(0.0078, color="gray", linestyle=":", linewidth=1)
    ax.text(60, 0.05, "chance (1/128)", color="gray", fontsize=9)
    ax.set_xlabel("Associative load P (number of key-value pairs in context)")
    ax.set_ylabel("Recall accuracy")
    ax.set_xscale("log", base=2); ax.set_xticks(loads); ax.set_xticklabels([str(x) for x in loads])
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Write gates must know how full the memory is", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_ladder.png"), bbox_inches="tight")
    print("fig_ladder.png done")

# ---------------- Figure 2: gate-profile contrast at P=48 ----------------
def fig_profile():
    rows = dedupe(load("e4_load.jsonl"))
    prof = {}
    for t, r in rows.items():
        if "P48" not in t or not r.get("gate_probe"): continue
        key = "gd" if "-gd-" in t else ("stat" if "stat" in t else "gdn2")
        for lay, classes in r["gate_probe"].items():
            for cls, v in classes.items():
                if not cls.startswith("e"):
                    prof.setdefault((key, cls), []).append(v)
    classes = ["0", "1", "2", "3"]
    labels = ["pair\nkey", "pair\nvalue", "query\nkey", "answer\n(re-write)"]
    arms = [("gd", "Scalar-tied gates", "#9aa5b1"),
            ("gdn2", "GDN-2 style (token-only)", "#4c78a8"),
            ("stat", "+ read-back stats (ours)", "#e45756")]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    w = 0.26
    for i, (key, label, color) in enumerate(arms):
        if (key, "0") not in prof: continue
        vals = [st.mean(prof[(key, c)]) for c in classes]
        xs = [j + (i - 1) * w for j in range(4)]
        ax.bar(xs, vals, width=w, color=color, label=label)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean write-gate value $\\beta$")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Learned gate profile at load P=48\n(ours keeps facts, suppresses re-writes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_profile.png"), bbox_inches="tight")
    print("fig_profile.png done")

# ---------------- Figure 3: redundancy census ----------------
def fig_census():
    rows = dedupe(load("phase1_capacity_survey.jsonl")) if False else {}
    # pull from archived phase-1 capacity survey jsonl
    p = os.path.join(RES, "raw", "phase1_capacity_survey.jsonl")
    if not os.path.exists(p):
        p = os.path.join(RES, "phase1_capacity_survey.jsonl")
    census = None
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r["tag"] == "B-census-s0":
                census = r.get("census"); break
    if not census:
        print("census data not found"); return
    layers = sorted({int(k.replace("layer", "")) for k in census})
    rates = [census[f"layer{i}"]["rate>0.5"] for i in layers]
    spikes = [census[f"layer{i}"]["rate>0.95"] for i in layers]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar([str(i) for i in layers], [r * 100 for r in rates], color="#4c78a8", label="redundant writes (cos>0.5)")
    ax.plot([str(i) for i in layers], [s * 100 for s in spikes], "o-", color="#e45756", label="spikes (cos>0.95)", markersize=5)
    ax.set_xlabel("layer"); ax.set_ylabel("% of writes")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Write redundancy in a real-text LM\n(25M tokens: diffuse, no spikes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_census.png"), bbox_inches="tight")
    print("fig_census.png done")

# ---------------- Figure 4: collision crossover ----------------
def fig_collision():
    rows = dedupe(load("e1_collision.jsonl"))
    B = ["pc0", "pc1", "pc10", "pc100", "pc1000"]
    word = {}
    for r in rows.values():
        if r["tag"].startswith("E1-word"):
            for b in B: word.setdefault(b, []).append(r["scores"][b])
    wm = {b: st.mean(v) for b, v in word.items()}
    Ns = [(8192, 1623.6), (32768, 405.9), (131072, 101.5), (262144, 50.7)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    colors = ["#4c78a8", "#54a24b", "#e45756", "#9aa5b1", "#72b7b2"]
    for (N, ups), color in zip(Ns, colors):
        r = rows[f"E1-bigram{N}-s0"]["scores"]
        leads = [wm[b] - r[b] for b in B]
        ax.plot(range(5), leads, "o-", color=color, label=f"rows={N//1024}K ({ups:.0f} upd/row)", markersize=5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["never", "1-9", "10-99", "100-999", "1000+"], fontsize=9)
    ax.set_ylabel("word-lead  (word − bigram)")
    ax.set_xlabel("training sightings of the bigram")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Hash-collision crossover is governed by\nupdates per row, not per-key sightings", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_collision.png"), bbox_inches="tight")
    print("fig_collision.png done")

if __name__ == "__main__":
    fig_ladder(); fig_profile(); fig_census(); fig_collision()