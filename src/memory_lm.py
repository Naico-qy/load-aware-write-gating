"""memory_lm.py — fixed-budget memory testbed for linear-attention LMs.

Research line: a linear-attention state is a lossy compressor, so (a) write timing/strength
matters, and (b) an external n-gram lookup table (under hash collision) acts as a second
memory store; the two compete for a fixed parameter budget.

Recurrence per LA layer (decay-first, GDN-compatible):
    S_t^pre = a_t * S_{t-1}
    S_t     = S_t^pre + (beta_t * v_t - e_t * (S_t^pre k_t)) k_t^T
    y_t     = S_t q_t                      (read includes the self write)

mixer codes:
    gla   e_t = 0                                  (gated linear attention, Mamba-2 style decay)
    delta e_t = beta_t, a_t = 1                    (delta rule)
    gd    e_t = beta_t, learned decay              (gated delta rule)
    gde   e_t = beta_t * sigmoid(w.h) channel-wise (GDN-2 style erase)
    sm    softmax attention (reference)

write gates (gamma = g0*exp(w), one w per head), gate scales the WRITE only:
    none  -
    cg    beta_t *= exp(-gamma*relu(cos(S_c   q_{t-1}, v_t)))    (paper-3 chunk-start gate)
    sg    beta_t *= exp(-gamma*relu(cos(S_t^pre q_{t-1}, v_t)))  (streaming gate, chunk-free — ours)
    sgk   beta_t *= exp(-gamma*relu(cos(S_t^pre k_t,  v_t)))     (streaming, erase read)
Gate signal is measured on an ungated twin recurrence (pass-1); the gated stream re-solves with
gated beta. With --census 1 and gate none, the twin signal is logged as a redundancy census.

Track C (text): word / hashed-bigram / hybrid value tables; per-sighting-bucket val loss.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- utils
def rope(x):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, d, 2, device=x.device).float() / d)
    ang = torch.outer(torch.arange(t, device=x.device).float(), f)
    c, s_ = ang.cos(), ang.sin(); x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1).flatten(-2)

def dconv(z, w):
    k = w.shape[-1]; out = 0.0
    for m in range(k):
        out = out + w[..., m].unsqueeze(1) * F.pad(z, (0, 0, m, 0))[..., :z.shape[-2], :]
    return out

def tri_dec(lc, inclusive=True):
    """G_ij = exp(lc_i - lc_j) for i>=j (inclusive) or i>j (exclusive)."""
    n = lc.shape[-1]
    M = torch.triu(torch.ones(n, n, device=lc.device, dtype=torch.bool), 1 if inclusive else 0)
    G = (lc[..., :, None] - lc[..., None, :]).masked_fill(M, float("-inf"))
    return G.exp()

def newton_schulz(G, steps=5):
    A, B, C = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        M = X @ X.T
        X = A * X + (B * M + C * M @ M) @ X
    return X.T if tr else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, wd=0.0):
        super().__init__(params, dict(lr=lr)); self.wd = wd
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for w in g["params"]:
                if w.grad is None: continue
                m = self.state[w].setdefault("m", torch.zeros_like(w))
                m.mul_(0.95).add_(w.grad)
                O = newton_schulz(w.grad + 0.95 * m)
                lr = g["lr"] * max(1.0, w.shape[0] / w.shape[1]) ** 0.5
                if self.wd: w.mul_(1 - self.wd * g["lr"])
                w.add_(O, alpha=-lr)

# ---------------------------------------------------------------- chunk math
def solve_chunk(St, qc, kc, vc, bc, lc, ec):
    """Exact one-chunk solve of S_t = a S_{t-1} + (beta v - e (S^pre k)) k^T.
    St: (b,H,dk,dv) state at chunk start; bc,lc,ec: (b,H,n) with lc = inclusive cumsum of log a.
    Returns y (b,H,n,dv), S_end, U (b,H,n,dv) (=u_t), A (b,H,n) inclusive decay,
    KWin (b,H,n,dv) = sum_{j<i} exp(lc_i-lc_j) (k_i.k_j) u_j  (the pre-write erase read's intra part)."""
    b, H, n, dk = kc.shape; dev = kc.device
    A = lc.exp()
    G = tri_dec(lc, inclusive=True)
    if ec is not None:
        SK0 = A[..., None] * torch.einsum("bhcv,bhtc->bhtv", St, kc)   # A_i (St k_i): chunk-start part of S_i^pre k_i
        M = (ec[..., None] * (kc @ kc.transpose(-1, -2)) * G).tril(-1)
        Y = bc[..., None] * vc - ec[..., None] * SK0
        I = torch.eye(n, device=dev, dtype=kc.dtype).expand(b, H, n, n)
        U = torch.linalg.solve_triangular(I + M, Y, upper=False)
    else:
        U = bc[..., None] * vc
    KWin = (G.tril(-1) * (kc @ kc.transpose(-1, -2))) @ U
    y = A[..., None] * torch.einsum("bhcv,bhtc->bhtv", St, qc) + ((qc @ kc.transpose(-1, -2)) * G) @ U
    w = (lc[..., -1:] - lc).exp()
    S_end = lc[..., -1].exp()[..., None, None] * St + torch.einsum("bhtv,bhtc->bhcv", w[..., None] * U, kc)
    return y, S_end, U, A, KWin

# ---------------------------------------------------------------- blocks
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); f = int(8 * d / 3) // 8 * 8
        self.gate = nn.Linear(d, f, bias=False); self.up = nn.Linear(d, f, bias=False); self.down = nn.Linear(f, d, bias=False)
    def forward(self, h): return self.down(F.silu(self.gate(h)) * self.up(h))

class SMBlock(nn.Module):
    def __init__(self, d, H):
        super().__init__(); self.H, self.dk = H, d // H; self.code = "sm"
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4)); self.mlp = MLP(d)
        self.census_report = None
    def forward(self, x, ids=None, prev=None):
        b, t, d = x.shape; h = self.n1(x); sp = lambda z: z.view(b, t, self.H, self.dk).transpose(1, 2)
        y = F.scaled_dot_product_attention(rope(sp(self.q(h))), rope(sp(self.k(h))), sp(self.v(h)), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d)); return x + self.mlp(self.n2(x))

class Table(nn.Module):
    """value-table E(ids, prev); added to attention values with a learned lam."""
    def __init__(self, V, d, kind, n_big=0, cover=0.0, rank=4, counts=None):
        super().__init__()
        self.kind = kind
        self.lam = nn.Parameter(torch.tensor(1.0))
        if kind == "word":
            self.T = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        elif kind == "bigram":
            self.T = nn.Parameter(torch.randn(n_big, d) * d ** -0.5)
        elif kind == "hybrid":
            c = np.asarray(counts, dtype=np.float64)
            top = np.argsort(-c); cum = np.cumsum(c[top]) / c.sum()
            k = int(np.searchsorted(cum, cover) + 1)
            lut = np.full(V, -1, np.int64); lut[top[:k]] = np.arange(k)
            self.register_buffer("lut", torch.tensor(lut, dtype=torch.long))
            self.full = nn.Parameter(torch.randn(k, d) * d ** -0.5)
            self.Ta = nn.Parameter(torch.randn(V, rank) * rank ** -0.5)
            self.Tb = nn.Parameter(torch.randn(rank, d) * (rank * d) ** -0.5)
        else:
            raise ValueError(kind)
    def forward(self, ids, prev):
        if self.kind == "word":
            return self.T[ids]
        if self.kind == "bigram":
            return self.T[(prev * 1000003 + ids) % self.T.shape[0]]
        j = self.lut[ids]
        has = (j >= 0).unsqueeze(-1).float()
        return self.full[j.clamp(min=0)] * has + (self.Ta[ids] @ self.Tb) * (1 - has)

class LABlock(nn.Module):
    def __init__(self, d, H, code, gate, g0, chunk, census=False, table=None, gate_erase=False, gate_stat=False, gate_probe=False):
        super().__init__()
        assert code in ("gla", "delta", "gd", "gde", "gdn2")
        self.H, self.dk, self.code, self.gate_code = H, d // H, code, gate
        self.gate_erase = bool(gate_erase); self.gate_stat = bool(gate_stat); self.gate_probe = bool(gate_probe)
        self.chunk, self.want_census, self.g0 = chunk, census, g0
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4)); self.mlp = MLP(d)
        self.cw = nn.Parameter(torch.cat([torch.ones(3, H, d // H, 1), torch.zeros(3, H, d // H, 3)], dim=-1))
        self.wb = nn.Parameter(torch.zeros(H, d)); self.bb = nn.Parameter(torch.zeros(H))
        self.wa = nn.Parameter(torch.zeros(H, d)); self.ba = nn.Parameter(torch.full((H,), -4.0))
        self.on = nn.Parameter(torch.ones(H, d // H))
        if code in ("gde", "gdn2"):
            self.lwe = nn.Parameter(torch.zeros(H, d)); self.lbe = nn.Parameter(torch.full((H,), -2.0))
        if gate != "none" or census: self.lpg = nn.Parameter(torch.zeros(H))
        if gate_stat:  # our method: per-head scalars folding read-back stats into gate logits (zero-init = identical to unconditioned)
            self.lws = nn.Parameter(torch.zeros(H)); self.lwo = nn.Parameter(torch.zeros(H))
            self.les = nn.Parameter(torch.zeros(H)); self.leo = nn.Parameter(torch.zeros(H))
        if table is not None: self.table = table
        self.reset_census()
        self.reset_probe()
    def reset_probe(self):
        self._gp = {}
    def probe_gate(self, beta_c, e_c, s0, n):
        """mean gate values per MQAR block-layout position class.
        classes: 0 pair-key, 1 pair-value, 2 query-key, 3 answer (pairs occupy [0,2P), queries [2P,L))."""
        P = a_pairs[0]
        pos = torch.arange(s0, s0 + n, device=beta_c.device)
        cls = torch.full((n,), -1, device=beta_c.device, dtype=torch.long)
        pair = pos < 2 * P
        cls[pair & (pos % 2 == 0)] = 0
        cls[pair & (pos % 2 == 1)] = 1
        q = pos >= 2 * P
        cls[q & ((pos - 2 * P) % 2 == 0)] = 2
        cls[q & ((pos - 2 * P) % 2 == 1)] = 3
        for g in (0, 1, 2, 3):
            sel = cls == g
            if int(sel.sum()) == 0: continue
            acc = self._gp.setdefault(g, [0.0, 0])
            acc[0] += float(beta_c[..., sel].mean()); acc[1] += 1
            if e_c is not None:
                acc2 = self._gp.setdefault("e" + str(g), [0.0, 0])
                acc2[0] += float(e_c[..., sel].mean()); acc2[1] += 1
    def probe_report(self):
        return {str(k): round(s / max(c, 1), 4) for k, (s, c) in sorted(self._gp.items(), key=lambda kv: str(kv[0]))} or None

    def reset_census(self):
        self._n = 0; self._sum = 0.0; self._sum2 = 0.0
        self._r5 = 0.0; self._r8 = 0.0; self._r95 = 0.0
        self._hist = torch.zeros(33)
    def update_census(self, cs):
        x = cs.detach().float().reshape(-1)
        n = x.numel()
        self._n += n
        self._sum += float(x.sum()); self._sum2 += float((x * x).sum())
        self._r5 += float((x > 0.5).sum()); self._r8 += float((x > 0.8).sum()); self._r95 += float((x > 0.95).sum())
        self._hist += torch.histc(x, bins=33, min=-1.0, max=1.0).cpu()
    def census_report(self):
        if self._n == 0: return None
        mean = self._sum / self._n; var = max(0.0, self._sum2 / self._n - mean * mean)
        return {"n": self._n, "mean": round(mean, 4), "std": round(var ** 0.5, 4),
                "rate>0.5": round(self._r5 / self._n, 4), "rate>0.8": round(self._r8 / self._n, 4),
                "rate>0.95": round(self._r95 / self._n, 4),
                "hist": [round(float(z) / self._n, 4) for z in self._hist]}
    def proj(self, h, ids=None, prev=None):
        b, t, d = h.shape; H, dk = self.H, self.dk
        sp = lambda z: z.view(b, t, H, dk).transpose(1, 2)
        v = self.v(h)
        if getattr(self, "table", None) is not None:
            v = v + self.table.lam * self.table(ids, prev)
        q = dconv(sp(self.q(h)), self.cw[0]); k = dconv(sp(self.k(h)), self.cw[1]); v = dconv(sp(v), self.cw[2])
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        zb = torch.einsum("btd,hd->bht", h, self.wb) + self.bb[None, :, None]
        beta = torch.sigmoid(zb)
        la = -F.softplus(torch.einsum("btd,hd->bht", h, self.wa) + self.ba[None, :, None])
        if self.code == "delta":
            la = torch.zeros_like(la)                                    # delta rule: no decay
        ze = torch.einsum("btd,hd->bht", h, self.lwe) + self.lbe[None, :, None] if self.code in ("gde", "gdn2") else None
        eg = torch.sigmoid(ze) if ze is not None else None
        self._zb, self._ze = zb, ze                                      # pre-sigmoid logits for --gate_stat correction
        return q, k, v, beta, la, eg
    def gam(self):
        return self.g0 * self.lpg.exp() if (self.gate_code != "none" or self.want_census) else None

    def la_forward(self, q, k, v, beta, la, eg, zb=None, ze=None):
        """chunked linear-attention core. q,k: (b,H,t,dk) normalized; v: (b,H,t,dv); beta,la: (b,H,t).
        Returns y (b,H,t,dv) pre-output-norm."""
        b, H, t, dk = q.shape; dv = v.shape[-1]; dev = q.device
        C = min(self.chunk, t); delta = self.code in ("delta", "gd", "gde", "gdn2")
        use_gate = self.gate_code != "none" or self.want_census or self.gate_stat
        gam = self.gam()
        St = torch.zeros(b, H, dk, dv, device=dev, dtype=q.dtype)
        St1 = torch.zeros_like(St) if use_gate else None
        # carried tensors of the last position of the previous chunk (for the sg2 self-write correction)
        p_kq = torch.zeros(b, H, 1, device=dev, dtype=q.dtype)
        p_u = torch.zeros(b, H, 1, dv, device=dev, dtype=q.dtype)
        outs = []
        for s0 in range(0, t, C):
            e0 = min(t, s0 + C); n = e0 - s0
            qc, kc, vc = q[..., s0:e0, :], k[..., s0:e0, :], v[..., s0:e0, :]
            bc = beta[..., s0:e0]
            lc = la[..., s0:e0].cumsum(-1)
            if self.code == "gdn2": ec = eg[..., s0:e0]
            elif self.code == "gde": ec = bc * eg[..., s0:e0]
            else: ec = bc if delta else None
            qsh = torch.cat([q[..., s0 - 1:s0, :] if s0 > 0 else torch.zeros_like(qc[:, :, :1]), qc[..., :n - 1, :]], dim=2)
            if use_gate:
                y1, S1e, U1, A1, KWin1 = solve_chunk(St1, qc, kc, vc, bc, lc, ec)   # ungated twin
                Rk = A1[..., None] * torch.einsum("bhcv,bhtc->bhtv", St1, kc) + KWin1       # S_t^pre k_t (free for delta arms)
                if self.gate_code == "sgk":
                    Rd = A1[..., None] * torch.einsum("bhcv,bhtc->bhtv", St1, kc) + KWin1
                elif self.gate_code in ("sg", "sg2"):
                    Rd = A1[..., None] * torch.einsum("bhcv,bhtc->bhtv", St1, qsh) + \
                         (tri_dec(lc, inclusive=False) * (qsh @ kc.transpose(-1, -2))) @ U1
                    if self.gate_code == "sg2":
                        # subtract the t-1 self-write: Rd then equals cos(S_{t-1}^pre q_{t-1}, v_t)
                        # (cosine is scale-free, so the residual a_t factor cancels)
                        kq = torch.einsum("bhtc,bhtc->bht", qc, kc)                   # (b,H,n) dot(k_t, q_t)
                        kq_sh = torch.cat([p_kq, kq[..., :n - 1]], dim=2)
                        u_sh = torch.cat([p_u, U1[..., :n - 1, :]], dim=2)            # twin's full write at t-1 (incl. delta correction)
                        a_cur = la[..., s0:e0].exp()[..., None]                       # exp(la_t): decay applied at t
                        self_term = a_cur * kq_sh[..., None] * u_sh
                        Rd = Rd - self_term
                elif self.gate_code == "cg":
                    Rd = torch.einsum("bhcv,bhtc->bhtv", St1, qsh)                   # chunk-start twin state only
                else:                                                                # census-only mode
                    Rd = A1[..., None] * torch.einsum("bhcv,bhtc->bhtv", St1, qsh) + \
                         (tri_dec(lc, inclusive=False) * (qsh @ kc.transpose(-1, -2))) @ U1
                St1 = S1e
                ec2s = None
                if self.gate_stat:
                    cs = F.cosine_similarity(Rk, vc, dim=-1, eps=1e-6)              # read-back confidence (b,H,n)
                    oc = Rk.norm(dim=-1) / (vc.norm(dim=-1) + 1e-6)                 # occupancy ratio along k_t
                    bc2 = torch.sigmoid(zb[..., s0:e0] + self.lws[None, :, None] * cs + self.lwo[None, :, None] * oc)
                    if delta:
                        ec2s = torch.sigmoid(ze[..., s0:e0] + self.les[None, :, None] * cs + self.leo[None, :, None] * oc)
                if self.gate_code != "none":
                    gf = torch.exp(-gam[None, :, None] * F.relu(F.cosine_similarity(Rd, vc, dim=-1, eps=1e-6)))
                    bc2 = bc2 * gf if self.gate_stat else bc * gf
                elif not self.gate_stat:
                    bc2 = bc
                if self.want_census:
                    self.update_census(F.cosine_similarity(Rd, vc, dim=-1, eps=1e-6))
            else:
                bc2 = bc
            ec2 = ec                                                               # default: erase stays ungated
            if delta:
                if self.code == "gdn2": ec2 = eg[..., s0:e0]                       # GDN-2 style: erase decoupled from write
                if self.gate_erase:
                    ec2 = bc2 if self.code != "gdn2" else (ec2s if self.gate_stat else eg[..., s0:e0])
            if self.gate_probe:
                self.probe_gate(bc2, (ec2 if delta else None), s0, n)
            y, S_end, U, A_, KWin = solve_chunk(St, qc, kc, vc, bc2, lc, ec2)       # gated main stream
            St = S_end
            # carry last-position scalars for the next chunk's sg2 correction
            p_kq = torch.einsum("bhtc,bhtc->bht", qc, kc)[..., n - 1:n]
            if use_gate:
                p_u = U1[..., n - 1:n, :]                                             # twin's last write (ungated)
            else:
                p_u = torch.zeros(b, H, 1, dv, device=dev, dtype=q.dtype)
            outs.append(y)
        return torch.cat(outs, dim=2)

    def forward(self, x, ids=None, prev=None):
        h = self.n1(x)
        q, k, v, beta, la, eg = self.proj(h, ids, prev)
        y = self.la_forward(q, k, v, beta, la, eg, getattr(self, "_zb", None), getattr(self, "_ze", None))
        y = F.rms_norm(y, (y.shape[-1],)) * self.on[None, :, None, :]
        x = x + self.o(y.transpose(1, 2).reshape(*x.shape))
        return x + self.mlp(self.n2(x))

# ---------------------------------------------------------------- model
class Model(nn.Module):
    def __init__(self, V, d, nl, H, mixers, gate, g0, chunk, census=False, tables=None, counts=None, n_big=0, cover=0.0, gate_erase=False, gate_stat=False, gate_probe=False):
        super().__init__()
        self.emb = nn.Embedding(V, d); nn.init.normal_(self.emb.weight, std=d ** -0.5)
        self.blocks = nn.ModuleList()
        for i, code in enumerate(mixers):
            if code == "sm":
                self.blocks.append(SMBlock(d, H))
            else:
                tab = None
                if tables and i in tables:
                    kind = tables[i]
                    tab = Table(V, d, kind, n_big, cover, 4, counts)
                self.blocks.append(LABlock(d, H, code, gate, g0, chunk, census, tab, gate_erase, gate_stat, gate_probe))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False); nn.init.normal_(self.head.weight, std=d ** -0.5)
    def forward(self, ids):
        x = self.emb(ids)
        prev = F.pad(ids, (1, 0), value=0)[:, :-1]
        for blk in self.blocks:
            x = blk(x, ids, prev)
        return self.head(self.norm(x))

# ---------------------------------------------------------------- data
class MQARGap:
    """Distance-controlled in-context recall (local RULER-style NIAH).
    Layout per pair i: key, value, gap_i fillers, query key, answer value.
    gap_i is drawn from --gaps (cycled with shuffle), so the fact-to-query distance is
    (gap_i + 1) tokens by construction. Scoring buckets by gap."""
    def __init__(self, rows, P, mv, seed, gaps=(24, 72, 152, 320)):
        g = np.random.default_rng(seed)
        third = mv // 3
        self.gaps = list(gaps)
        gmax = max(self.gaps)
        L = P * (4 + gmax)
        self.x = np.zeros((rows, L), np.int64); self.ans = np.zeros((rows, L), bool)
        self.bucket = -np.ones((rows, L), int)
        self.gap_idx = np.zeros((rows, P), int)
        for r in range(rows):
            keys = g.permutation(third)[:P]; vals = third + g.permutation(third)[:P]
            order = g.permutation(len(self.gaps))
            pos = 0
            for i in range(P):
                gi = int(order[i % len(self.gaps)])
                gap = self.gaps[gi]
                self.x[r, pos] = keys[i]; self.x[r, pos + 1] = vals[i]
                fill = 2 * third + g.integers(0, max(third, 1), gap)
                self.x[r, pos + 2:pos + 2 + gap] = fill
                self.x[r, pos + 2 + gap] = keys[i]; self.x[r, pos + 3 + gap] = vals[i]
                self.ans[r, pos + 3 + gap] = True
                self.bucket[r, pos + 3 + gap] = gi
                self.gap_idx[r, i] = gi
                pos += 4 + gap
            if pos < L:  # tail filler (does not change any distance)
                self.x[r, pos:] = 2 * third + g.integers(0, max(third, 1), L - pos)
    def __len__(self): return len(self.x)

class MQAR:
    def __init__(self, rows, L, P, mv, seed, interleave=False, il_gap=0):
        g = np.random.default_rng(seed)
        half = mv // 2
        self.x = np.zeros((rows, L), np.int64); self.ans = np.zeros((rows, L), bool)
        kk = np.array([g.permutation(half)[:P] for _ in range(rows)])
        vv = half + np.array([g.permutation(half)[:P] for _ in range(rows)])
        if interleave:
            # per pair: k_i v_i, il_gap filler tokens, then the repeat query k_i -> v_i
            # vocab split in thirds: keys [0,t), values [t,2t), fillers [2t,3t)
            blk = 2 + il_gap + 2
            assert L == blk * P, "L must equal P * (4 + il_gap)"
            third = mv // 3
            kk = np.array([g.permutation(third)[:P] for _ in range(rows)])
            vv = third + np.array([g.permutation(third)[:P] for _ in range(rows)])
            fill = 2 * third + g.integers(0, third, (rows, P, il_gap))
            for i in range(P):
                c0 = i * blk
                self.x[:, c0] = kk[:, i]; self.x[:, c0 + 1] = vv[:, i]
                self.x[:, c0 + 2:c0 + 2 + il_gap] = fill[:, i]
                self.x[:, c0 + 2 + il_gap] = kk[:, i]; self.x[:, c0 + 3 + il_gap] = vv[:, i]
                self.ans[:, c0 + 3 + il_gap] = True
        else:
            n_q = L - 2 * P
            assert n_q % 2 == 0 and P <= mv // 2
            self.x[:, 0:2 * P:2] = kk; self.x[:, 1:2 * P:2] = vv
            qidx = np.array([g.integers(0, P, n_q // 2) for _ in range(rows)])
            self.x[:, 2 * P::2] = np.take_along_axis(kk, qidx, 1)
            self.x[:, 2 * P + 1::2] = np.take_along_axis(vv, qidx, 1)
            self.ans[:, 2 * P + 1::2] = True
    def __len__(self): return len(self.x)

# ---------------------------------------------------------------- train / eval
a_pairs = [16]  # set by main; used by gate_probe class computation

def evaluate_mqar(model, val, dev, bs=64):
    model.eval()
    losses, correct, tot = [], 0, 0
    with torch.no_grad():
        for i in range(0, len(val), bs):
            x = torch.tensor(val.x[i:i + bs], device=dev)
            m = torch.tensor(val.ans[i:i + bs], device=dev)
            logits = model(x[:, :-1])
            tgt = x[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), reduction="none")
            mm = m[:, 1:].reshape(-1)
            losses.append(float(ce[mm].mean()))
            correct += int((logits.argmax(-1) == tgt).reshape(-1)[mm].sum()); tot += int(mm.sum())
    model.train()
    return {"loss": round(sum(losses) / len(losses), 4), "acc": round(correct / tot, 4)}

@torch.no_grad()
def text_val_ce(model, val, V, dev, bs=100):
    """per-position CE on the val set, batched to avoid one giant logits tensor."""
    model.eval()
    ces = []
    for i in range(0, len(val), bs):
        x = val[i:i + bs].to(dev)
        logits = model(x[:, :-1])
        ces.append(F.cross_entropy(logits.reshape(-1, V).float(), x[:, 1:].reshape(-1), reduction="none").reshape(len(x), -1).cpu())
    model.train()
    return torch.cat(ces, 0)

def text_val_loss(model, val, V, dev):
    return float(text_val_ce(model, val, V, dev).mean())

@torch.no_grad()
def evaluate_mqar_gap(model, val, dev, bs=32):
    """Per-gap-bucket recall accuracy (and loss) on the distance-controlled task."""
    model.eval()
    losses, correct, tot = [], 0, 0
    bcorrect = {}; btot = {}
    with torch.no_grad():
        for i in range(0, len(val), bs):
            x = torch.tensor(val.x[i:i + bs], device=dev)
            m = torch.tensor(val.ans[i:i + bs], device=dev)
            bk = torch.tensor(val.bucket[i:i + bs], device=dev)
            logits = model(x[:, :-1]); tgt = x[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), reduction="none")
            mm = m[:, 1:].reshape(-1)
            losses.append(float(ce[mm].mean()))
            pred = (logits.argmax(-1) == tgt).reshape(-1)[mm]
            lbl = bk[:, 1:].reshape(-1)[mm]
            correct += int(pred.sum()); tot += int(pred.numel())
            for g in torch.unique(lbl).tolist():
                sel = lbl == g
                bcorrect[g] = bcorrect.get(g, 0) + int(pred[sel].sum())
                btot[g] = btot.get(g, 0) + int(sel.sum())
    model.train()
    out = {"loss": round(sum(losses) / len(losses), 4), "acc": round(correct / max(tot, 1), 4)}
    for g in sorted(btot):
        out[f"acc_g{g}"] = round(bcorrect[g] / btot[g], 4) if btot[g] else None
    return out

def schedule(step, steps, warm, cool):
    ramp = min(1.0, (step + 1) / max(1, warm))
    cstart = int(steps * (1 - cool))
    cd = min(1.0, (steps - step) / max(1, steps - cstart)) if step >= cstart else 1.0
    return ramp * cd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mqar", choices=["mqar", "mqar_gap", "text"])
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--interleave", type=int, default=0, help="1: each pair followed by its repeat query after --il_gap fillers")
    ap.add_argument("--il_gap", type=int, default=0)
    ap.add_argument("--gaps", default="24,72,152,320", help="mqar_gap: fact-to-query gaps, one per bucket")
    ap.add_argument("--mv", type=int, default=256)
    ap.add_argument("--val_rows", type=int, default=512)
    ap.add_argument("--data", default=""); ap.add_argument("--val_data", default="")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--mixer", default="gla", choices=["gla", "delta", "gd", "gde", "gdn2", "sm"])
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--gate", default="none", choices=["none", "cg", "sg", "sg2", "sgk"])
    ap.add_argument("--g0", type=float, default=0.25)
    ap.add_argument("--gate_erase", type=int, default=0, help="delta family: also scale the erase strength e by the write gate (default: write only)")
    ap.add_argument("--gate_stat", type=int, default=0, help="1: condition beta/erase logits on per-token read-back stats (cos, occupancy) from the ungated twin - our method")
    ap.add_argument("--gate_probe", type=int, default=0, help="1: record mean beta/e per MQAR position class (pair-key/pair-value/query-key/answer)")
    ap.add_argument("--ve", default="", help="comma layer indices for word value tables (text)")
    ap.add_argument("--vebig", type=int, default=0, help="N>0: bigram-hash value tables with N rows")
    ap.add_argument("--vecover", type=float, default=0.0, help=">0: hybrid tables (needs --ve)")
    ap.add_argument("--tlr", type=float, default=-1.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--elr", type=float, default=0.04)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--cool", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--census", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.dev if (a.dev != "cuda" or torch.cuda.is_available()) else "cpu")

    mixers = [a.mixer] * a.layers
    tables, counts, n_big, cover = None, None, 0, 0.0
    if a.task == "text":
        tr = np.memmap(a.data, dtype=np.uint16, mode="r")
        va = np.memmap(a.val_data, dtype=np.uint16, mode="r")
        V = int(max(int(tr[:50_000_000].max()), int(va.max()))) + 1
        need = a.steps * a.batch * a.L
        assert need + a.batch * a.L <= len(tr), "not enough train tokens"
        off = min(a.seed * 1_000_000, len(tr) - need - a.batch * a.L)
        train = torch.tensor(np.asarray(tr[off:off + need]).reshape(a.steps * a.batch, a.L).astype(np.int64), dtype=torch.long)
        vrows = np.linspace(0, len(va) - a.L - 1, 1000).astype(np.int64)
        val = torch.tensor(np.stack([np.asarray(va[x:x + a.L]).astype(np.int64) for x in vrows]), dtype=torch.long)
        # pair-sighting buckets from the train window
        tp = train[:, :-1].reshape(-1) * (V + 1) + train[:, 1:].reshape(-1)
        u, c = np.unique(tp.numpy(), return_counts=True)
        vp = val[:, :-2].reshape(-1) * (V + 1) + val[:, 1:-1].reshape(-1)
        ix = np.clip(np.searchsorted(u, vp.numpy()), 0, len(u) - 1)
        pcnt = np.where(u[ix] == vp.numpy(), c[ix], 0).reshape(val.shape[0], a.L - 2)
        # align: CE column j is target position j+1; its pair is (j, j+1) -> vp index j (for j <= L-3)
        pc_full = np.zeros((val.shape[0], a.L - 1), np.int64)
        pc_full[:, :a.L - 2] = pcnt
        PF = [0, 1, 10, 100, 1000, 10 ** 12]
        pbsel = {f"pc{lo}": ((pc_full >= lo) & (pc_full < hi)) for lo, hi in zip(PF[:-1], PF[1:])}
        if a.ve:
            counts = np.bincount(train.reshape(-1).numpy(), minlength=V)
            tables = {int(i): ("bigram" if a.vebig else ("hybrid" if a.vecover else "word")) for i in a.ve.split(",") if i != ""}
            n_big = a.vebig; cover = a.vecover
    else:
        V = a.mv
        if a.task == "mqar_gap":
            gaps = tuple(int(z) for z in a.gaps.split(","))
            train_np = MQARGap(a.steps * a.batch, a.pairs, a.mv, a.seed, gaps)
            val = MQARGap(a.val_rows, a.pairs, a.mv, a.seed + 1, gaps)
            L = train_np.x.shape[1]
        else:
            a_pairs[0] = a.pairs
            train_np = MQAR(a.steps * a.batch, a.L, a.pairs, a.mv, a.seed, interleave=bool(a.interleave), il_gap=a.il_gap)
            val = MQAR(a.val_rows, a.L, a.pairs, a.mv, a.seed + 1, interleave=bool(a.interleave), il_gap=a.il_gap)
            L = a.L
        train = torch.tensor(train_np.x, dtype=torch.long)

    model = Model(V, a.width, a.layers, a.heads, mixers, a.gate, a.g0, a.chunk,
                  census=bool(a.census), tables=tables, counts=counts, n_big=n_big, cover=cover,
                  gate_erase=bool(a.gate_erase), gate_stat=bool(a.gate_stat), gate_probe=bool(a.gate_probe)).to(dev)
    nparam = sum(w.numel() for w in model.parameters())

    muon_params, adam_params, tab_params = [], [], []
    tlr = a.tlr if a.tlr > 0 else a.elr
    for name, w in model.named_parameters():
        if w.ndim == 2 and name.endswith((".q.weight", ".k.weight", ".v.weight", ".o.weight", ".mlp.gate.weight", ".mlp.up.weight", ".mlp.down.weight")):
            muon_params.append(w)                                    # Muon on the block matrices
        elif ".table." in name or name.endswith(".lam"):
            tab_params.append(w)                                     # tables at their own Adam rate
        else:
            adam_params.append(w)
    rest_params = adam_params
    groups = [dict(params=rest_params, lr=a.elr)]
    if tab_params: groups.append(dict(params=tab_params, lr=tlr))
    mu = Muon(muon_params, a.lr, wd=a.wd); ad = torch.optim.Adam(groups, betas=(0.9, 0.95))

    curve = []
    t0 = time.time()
    for step in range(a.steps):
        sc = schedule(step, a.steps, a.warm, a.cool)
        for g in mu.param_groups: g["lr"] = a.lr * sc
        for g in ad.param_groups:
            g["lr"] = (tlr if g in groups[1:] else a.elr) * sc if len(groups) > 1 else a.elr * sc
        xb = train[step * a.batch:(step + 1) * a.batch].to(dev)
        if a.task == "mqar":
            m = torch.tensor(train_np.ans[step * a.batch:(step + 1) * a.batch], device=dev)
            logits = model(xb[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, V).float(), xb[:, 1:].reshape(-1), reduction="none")
            loss = ce[m[:, 1:].reshape(-1)].mean()
        else:
            loss = F.cross_entropy(model(xb[:, :-1]).reshape(-1, V).float(), xb[:, 1:].reshape(-1))
        loss.backward()
        mu.step(); ad.step()
        model.zero_grad(set_to_none=True)
        if a.log_every and (step + 1) % a.log_every == 0:
            if a.task == "mqar": curve.append([step + 1, evaluate_mqar(model, val, dev)])
            elif a.task == "mqar_gap": curve.append([step + 1, evaluate_mqar_gap(model, val, dev)])
            else: curve.append([step + 1, round(text_val_loss(model, val, V, dev), 4)])

    # ---- final scores ----
    res_scores = {}
    if a.task == "mqar":
        res_scores = evaluate_mqar(model, val, dev)
    elif a.task == "mqar_gap":
        res_scores = evaluate_mqar_gap(model, val, dev)
    else:
        model.eval()
        ce = text_val_ce(model, val, V, dev)   # (nval, L-1) per-position CE, batched
        res_scores["val"] = round(float(ce.mean()), 4)
        unk = int(V - 1)
        known = (val[:, 1:] != unk)
        if bool(known.any()):
            res_scores["val_known"] = round(float(ce[known].mean()), 4)
            for b, sel in pbsel.items():
                m2 = torch.tensor(sel) & known                          # cpu tensors (ce is on cpu)
                if int(m2.sum()) >= 50:
                    res_scores[b] = round(float(ce[m2].mean()), 4)
    census = {}
    for i, blk in enumerate(model.blocks):
        if isinstance(blk, LABlock) and a.census:
            r = blk.census_report()
            if r: census[f"layer{i}"] = r
    gam = {f"layer{i}": [round(float(g_), 3) for g_ in blk.lpg.exp()] for i, blk in enumerate(model.blocks)
           if isinstance(blk, LABlock) and hasattr(blk, "lpg")}
    # --- capacity instrumentation ---
    tab_info = {}
    tab_params = 0
    for i, blk in enumerate(model.blocks):
        t = getattr(blk, "table", None)
        if t is None: continue
        tp = sum(x.numel() for x in t.parameters())
        tab_params += tp
        rows = None
        if getattr(t, "T", None) is not None: rows = int(t.T.shape[0])
        elif getattr(t, "full", None) is not None: rows = int(t.full.shape[0])
        tab_info[f"layer{i}"] = {"kind": t.kind, "rows": rows, "params": tp}
    cap = {}
    if a.task == "text" and tab_params:
        tp_ = train[:, :-1].reshape(-1) * (V + 1) + train[:, 1:].reshape(-1)
        uniq = int(np.unique(tp_.numpy()).size)
        total_upd = int(tp_.numel())
        cap = {"unique_pairs": uniq, "total_updates": total_upd,
               "updates_per_row": round(total_upd / max(1, tab_info.get("layer0", {}).get("rows") or 1), 1),
               "rows": tab_info.get("layer0", {}).get("rows")}
    gate_probe = {}
    for i, blk in enumerate(model.blocks):
        if isinstance(blk, LABlock) and a.gate_probe:
            gp = blk.probe_report()
            if gp: gate_probe[f"layer{i}"] = gp
    out = dict(tag=a.tag, args=vars(a), params=nparam, body_params=nparam - tab_params,
               table_params=tab_params, tables=tab_info, capacity=cap,
               scores=res_scores, curve=curve,
               census=census, gamma=gam, gate_probe=gate_probe, train_s=round(time.time() - t0, 1), dev=str(dev))
    with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")
    print(json.dumps({**{k: v for k, v in out.items() if k in ("tag", "scores", "train_s", "params")}}), flush=True)

if __name__ == "__main__":
    main()
