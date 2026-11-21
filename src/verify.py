"""verify.py — numerical verification that the chunked implementation exactly matches the
sequential recurrence, for every mixer code and gate code. This is the reliability backbone:
any experiment built on bed.py is only trusted after these pass.

Reference (sequential, per step t):
    Spre = a_t * S
    u    = beta_t * v_t - e_t * (Spre . k_t)          (e per mixer; 0 for gla)
    S    = Spre + u k_t^T
    y_t  = S . q_t                                    (read includes the self write)
Gate (cg/sg/sgk) runs an UNGATED TWIN stream S1 for the signal:
    Sp1   = a_t * S1 ;  <twin update with ungated beta> ;  S1 = Sp1 + u1 k_t^T
    Rd    = Sp1 . r_t   (r_t = q_{t-1} for cg/sg — cg uses S1 WITHOUT decay and no intra part,
                           i.e. the twin's chunk-start state; sgk uses Sp1 . k_t)
    beta_t *= exp(-gam * relu(cos(Rd, v_t)))
cg semantics: the chunked cg reads the twin's CHUNK-START state with q_{t-1}. The sequential
equivalent is therefore evaluated per chunk: signal state = twin state at chunk start (no decay
applied), read vector = q_{t-1} (may come from the previous chunk). We replicate exactly that.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import memory_lm as bed

torch.manual_seed(0)

def seq_forward(code, q, k, v, beta, la, eg, gate=None, gam=None, chunk=16):
    b, H, t, dk = q.shape; dv = v.shape[-1]
    delta = code in ("delta", "gd", "gde", "gdn2")
    S = torch.zeros(b, H, dk, dv, dtype=torch.float64)
    S1 = torch.zeros_like(S)
    S1_prevpre = torch.zeros_like(S)
    ys = []
    for i in range(t):
        ai = la[..., i].exp()[..., None, None]
        Spre = ai * S
        if gate is not None and gate != "none":
            Sp1 = ai * S1
            if gate == "sgk":
                Rd = torch.einsum("bhcv,bhc->bhv", Sp1, k[..., i, :])
            elif gate == "sg":
                rq = q[..., i - 1, :] if i > 0 else torch.zeros_like(q[..., 0, :])
                Rd = torch.einsum("bhcv,bhc->bhv", Sp1, rq)
            elif gate == "sg2":
                # read the twin's PRE-state at t-1 (before t-1 wrote itself): the genuine lookup
                rq = q[..., i - 1, :] if i > 0 else torch.zeros_like(q[..., 0, :])
                Rd = torch.einsum("bhcv,bhc->bhv", S1_prevpre, rq)
            else:  # cg: twin's chunk-start state, no decay, read q_{t-1}
                if i % chunk == 0:
                    cg_state = S1.clone()          # twin state at chunk start (before this chunk's writes)
                rq = q[..., i - 1, :] if i > 0 else torch.zeros_like(q[..., 0, :])
                Rd = torch.einsum("bhcv,bhc->bhv", cg_state, rq)
            gf = torch.exp(-gam[None, :] * F.relu(F.cosine_similarity(Rd, v[..., i, :], dim=-1, eps=1e-6)))
            b1 = beta[..., i]
            e1 = (b1 if code == "gd" else (eg[..., i] if code == "gdn2" else (b1 * eg[..., i] if code == "gde" else b1))) if delta else None
            SK1 = torch.einsum("bhcv,bhc->bhv", Sp1, k[..., i, :])
            u1 = b1[..., None] * v[..., i, :] - ((e1[..., None] * SK1) if e1 is not None else 0)
            S1 = Sp1 + (u1[..., None, :] * k[..., i, :][..., :, None])
            S1_prevpre = Sp1
            bi = beta[..., i] * gf                                   # gated main-stream beta (write only)
            bgate_free = beta[..., i]                                # erase strength stays ungated (paper-3 semantics)
        else:
            bi = beta[..., i]; bgate_free = bi
        e = (eg[..., i] if code == "gdn2" else (bgate_free * eg[..., i] if code == "gde" else bgate_free)) if delta else None
        SK = torch.einsum("bhcv,bhc->bhv", Spre, k[..., i, :])
        u = bi[..., None] * v[..., i, :] - ((e[..., None] * SK) if e is not None else 0)
        S = Spre + (u[..., None, :] * k[..., i, :][..., :, None])
        ys.append(torch.einsum("bhcv,bhc->bhv", S, q[..., i, :]))
    return torch.stack(ys, dim=2)

def check(name, a_, b_, tol=2e-4):
    err = float((a_ - b_).abs().max())
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: max|diff|={err:.2e}")
    return ok

def main():
    dev = torch.device("cpu")
    dt = torch.float64
    b, H, t, dk, dv = 2, 3, 96, 8, 12
    chunk = 16
    g = torch.Generator().manual_seed(1)
    all_ok = True
    for code in ("gla", "delta", "gd", "gde", "gdn2"):
        for gate in ("none", "cg", "sg", "sg2", "sgk"):
            q = F.normalize(torch.randn(b, H, t, dk, generator=g, dtype=dt), dim=-1)
            k = F.normalize(torch.randn(b, H, t, dk, generator=g, dtype=dt), dim=-1)
            v = torch.randn(b, H, t, dv, generator=g, dtype=dt) * 0.5
            beta = torch.rand(b, H, t, generator=g, dtype=dt)
            if code == "delta":
                la = torch.zeros(b, H, t, dtype=dt)                      # delta rule: no decay (bed zeroes in proj)
            else:
                la = -F.softplus(torch.randn(b, H, t, generator=g, dtype=dt)) - 0.1
            eg = torch.rand(b, H, t, generator=g, dtype=dt) if code in ("gde", "gdn2") else None
            gam = (0.25 * torch.exp(torch.randn(H, generator=g, dtype=dt) * 0.1)) if gate != "none" else None
            blk = bed.LABlock(d=H * dk, H=H, code=code, gate=(gate if gate != "none" else "none"),
                              g0=0.25, chunk=chunk, census=False)
            if gate != "none":
                with torch.no_grad(): blk.lpg.data.copy_((gam / 0.25).log().float())   # block gam == ref gam
            with torch.no_grad():
                y_chunk = blk.la_forward(q, k, v, beta, la, eg)          # stays float64 end to end
            with torch.no_grad():
                y_ref = seq_forward(code, q, k, v, beta, la, eg, gate=(gate if gate != "none" else None),
                                    gam=gam, chunk=chunk)
            all_ok &= check(f"{code}/{gate}", y_chunk, y_ref, tol=1e-8)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
