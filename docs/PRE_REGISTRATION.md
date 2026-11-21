# Pre-registered hypotheses and falsification conditions

This project follows a pre-registration discipline: hypotheses, experiment matrices, and
falsification conditions are written **before** the corresponding runs, and negative results
are recorded verbatim. This file is the English pre-registration for the load-aware gating
study (E4); the full experiment log with all runs is in `results/*.jsonl`.

---

## H1 — write gates are region-dependent (pilot, 110+ runs)

**Claim.** The optimal write-gate read signal depends on the geometry between a fact and its
query: a chunk-start read signal is optimal when facts and queries live in different chunks,
and **mechanically dead** when they share a chunk (the state it reads is empty).

**Falsification conditions (written before the sweep):**
1. If the chunk-start gate shows no gap vs. the streaming gates in the cross-chunk regime,
   the region-dependence claim dies.
2. If the chunk-start gate's learned values differ strongly across loads *in the same-chunk
   regime* (i.e. it is not dead), the "mechanical death" claim dies.

**Result.** Confirmed. Cross-chunk: chunk-start gate best (3.4x over baseline, 2 seeds).
Same-chunk: output bit-identical to the ungated baseline, gate values pinned at init
(gamma = 1.0, zero gradient). Mechanism-level evidence, not a score comparison.

## H2 — real-text write redundancy is diffuse (3 runs, 25M tokens)

**Claim.** In a real-text LM, redundant state writes are diffuse (no cos>0.95 spikes) and
rise with depth; therefore hard "block redundant writes" oracles transfer poorly and any
write gate's on-text gain is bounded by the diffuse redundancy.

**Falsification conditions:**
1. If >5% of writes at any layer are spikes (cos>0.95), the "no spikes" claim dies.
2. If a hard oracle on the census achieves large on-text gains, the "diffuse limits gain"
   claim dies.

**Result.** Confirmed: per-layer redundancy 5.0/11.4/9.2/10.3/14.1/10.5 % (cos>0.5),
spikes (cos>0.95) < 0.1% everywhere. Read-back-conditioned gate on text: −0.017 nats
(directional, single seed — labeled as such everywhere).

## H3 — hashed-table crossover is governed by updates per row (13 runs)

**Claim.** For a collision-pooled hashed table, the crossover against a per-word table is
governed by **updates per row**, not per-key sightings: as rows grow (updates/row falls),
the hashed table's rare-bucket advantage shrinks monotonically and crosses over around
~100 updates/row.

**Falsification conditions:**
1. If the hashed table's rare-bucket advantage does not shrink as rows grow by 32x,
   H3 dies.
2. If the crossover position is unrelated to the ~100-updates/row threshold, the
   "updates per row" reading dies and a per-key reading must be restored.

**Result.** Confirmed (see README, E1 section). Known confound, recorded honestly: changing
the row count also changes row-mate composition. A fixed-load permutation control
(vary who shares a row at constant N and constant load) is required to isolate the
composition effect and is planned as follow-up work.

## H4 — delta-rule write gates are load-blind (main study, 32 runs)

**Claim.** Token-conditioned erase/write gates (GDN-2 style) cannot express the optimal
write policy under changing associative load, because their input carries no information
about how full the memory is or whether the incoming write collides with stored content.
Conditioning the same gates on two free read-back statistics (collision confidence and
occupancy) closes the gap.

**Falsification conditions (pre-registered):**
1. If the token-conditioned gate's per-position-class values already differ strongly
   across loads (|Δβ| > 0.15 between P=16 and P=64 on any class), "load-blindness" dies.
2. If adding the read-back statistics does not improve accuracy by ≥2 points at load 32,
   "our method closes the gap" dies.
3. If the stat-conditioned variant is *worse* than the token-conditioned variant at low
   load (P=16), the "no regression" condition dies and the method must be revised.

**Result.** Confirmed. Token-conditioned gate profiles are structurally load-invariant
(pair-value 0.60 vs 0.58; answer 0.61 vs 0.58 between P=48 and P=64) while baseline
accuracy falls 0.87 → 0.58. Adding read-back stats: +12.4 points at load 32 (seed-sd
0.001), no regression at load 16 (+2.5). At extreme overload (P=64 ≫ state capacity) all
gate variants compress into a high-variance band (0.43–0.81) — **the capacity wall
dominates gate design there; gates help in the approach regime, not beyond the wall**.

## Known confounds and negative results (recorded verbatim)

1. A per-pair gap-controlled distance task (filler gaps 8/24/48/96) failed to learn at
   gaps >= 16 at this scale (accuracy ~ chance) while the single-gap version (gap 8)
   learned to 99.7% — implementation verified separately; the failure is optimization
   difficulty at longer sequences, not a math error. Context length was used as the
   distance proxy instead.
2. E3 (fixed-budget body-vs-table allocation) showed a monotone trend (wider body beats
   hash-table capacity at ~1.5 tokens/param) instead of the expected U-shape; recorded
   with a design confound (widening the model also enlarges the embedding/head).
3. The E2 streaming-query gate showed no consistent gain in the block layout — consistent
   with H1's region map (block layout's winning gate is the chunk-start read), recorded
   as a negative result rather than hidden.
4. Single-seed cells are labeled directional everywhere; headline claims use >= 2 seeds.
   One single-seed "win" (a 10-point gap at load 32) was re-measured with more seeds and
   collapsed to a tie — recorded as a cautionary example in the experiment log.