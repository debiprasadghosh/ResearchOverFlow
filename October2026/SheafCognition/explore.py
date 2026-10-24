"""Numerical exploration for 'Sheaf Geometry for Flexible Cognition'.
All experiments are synthetic; they probe the stated theorems and the in-silico
analogue of Hypothesis 1 / Conjecture 1. Seeds are fixed for reproducibility."""
import json, numpy as np, scipy.linalg as sla
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = {}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
C1, C2, C3, C4, GR = "#1f4e79", "#c0504d", "#4f8f3a", "#8064a2", "#7f7f7f"
pinv = np.linalg.pinv
PR = lambda S: np.trace(S) ** 2 / np.trace(S @ S)

# ---------------------------------------------------------------- E1 reproduce 9.2
D = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]], float)
def tri(b):
    b = np.asarray(b, float); k = -pinv(D) @ b; return k, np.linalg.norm(D @ k + b)
k1, r1 = tri([1, 1, -2]); k2, r2 = tri([1, 1, 1])
d = np.array([[-1, 0, 1, 0]], float); x0 = np.array([1, 2, -1, -3.])
priv = [abs(sla.expm(-20 * w * d.T @ d) @ x0 - x0)[[1, 3]].max() for w in (0.2, 1, 4)]
shared02 = (sla.expm(-20 * 0.2 * d.T @ d) @ x0)[[0, 2]]
rng = np.random.default_rng(23)
A = rng.standard_normal((7, 10)); E = 0.02 * rng.standard_normal((7, 10)); z = rng.standard_normal(10)
L, Lp, tau = A.T @ A, (A + E).T @ (A + E), 0.3
h = np.linalg.solve(np.eye(10) + tau * L, z); hp = np.linalg.solve(np.eye(10) + tau * Lp, z)
OUT["E1"] = dict(compatible_resid=r1, k_compat=k1.tolist(), obstructed_eps=r2,
                 PR_src=PR(np.diag([100., 1])), PR_tgt=PR(np.eye(2)),
                 PR_sv=(0.1 + 1) ** 2 / (0.01 + 1), private_err=priv, shared_t20_w02=shared02.tolist(),
                 resolvent_obs=np.linalg.norm(h - hp), resolvent_bound=tau * np.linalg.norm(L - Lp, 2) * np.linalg.norm(z))

# ---------------------------------------------------------------- helpers: general cellular sheaves
def coboundary(nv, edges, dv, de, R):
    """R[(e,'s')], R[(e,'t')] : maps dv->de ; (delta x)_e = R_t x_t - R_s x_s"""
    off = np.concatenate([[0], np.cumsum([dv] * nv)]); M = np.zeros((len(edges) * de, nv * dv))
    for i, (s, t) in enumerate(edges):
        M[i*de:(i+1)*de, off[t]:off[t]+dv] += R[(i, 't')]
        M[i*de:(i+1)*de, off[s]:off[s]+dv] -= R[(i, 's')]
    return M
nullity = lambda M, tol=1e-9: M.shape[1] - np.linalg.matrix_rank(M, tol)
def null(M, tol=1e-9):
    U, s, Vt = np.linalg.svd(M); r = (s > tol).sum(); return Vt[r:].T

def random_graph(rng, nv, p):
    while True:
        edges = [(i, j) for i in range(nv) for j in range(i + 1, nv) if rng.random() < p]
        # connectivity check
        adj = {i: set() for i in range(nv)}
        for s, t in edges: adj[s].add(t); adj[t].add(s)
        seen, st = {0}, [0]
        while st:
            u = st.pop()
            for w in adj[u] - seen: seen.add(w); st.append(w)
        if len(seen) == nv: return edges

def extension_sheaf(rng, nv, edges, dk, dq, ek, eq, bscale, qrank_deficit=True):
    """Triangular extension 0->K->F->Q->0 with block restrictions [[A,B],[0,C]].
    Q restrictions are made to share a common 'rule' subspace so H0(Q) is non-trivial."""
    RA, RB, RC = {}, {}, {}
    Gq = rng.standard_normal((eq, dq))               # common quotient interface (constant-like)
    for i, _ in enumerate(edges):
        for end in "st":
            RA[(i, end)] = rng.standard_normal((ek, dk))
            RB[(i, end)] = bscale * rng.standard_normal((ek, dq))
            RC[(i, end)] = Gq
    dK = coboundary(nv, edges, dk, ek, RA); dQ = coboundary(nv, edges, dq, eq, RC)
    RF = {key: np.block([[RA[key], RB[key]], [np.zeros((eq, dk)), RC[key]]]) for key in RA}
    dF = coboundary(nv, edges, dk + dq, ek + eq, RF)
    return dK, dQ, dF, RB

def connecting_map(nv, edges, dk, dq, ek, eq, dK, dQ, dF):
    """Matrix of ∂ : H0(Q) -> C1(K)/im dK, via lifts x=(0,q) and K-part of δF x, projected on coker dK."""
    HQ = null(dQ)
    # embed q into F by zero K-part (vertexwise ordering is [k_v, q_v])
    nF = nv * (dk + dq); lift = np.zeros((nF, HQ.shape[1]))
    for v in range(nv):
        lift[v*(dk+dq)+dk: (v+1)*(dk+dq), :] = HQ[v*dq:(v+1)*dq, :]
    res = dF @ lift
    Kidx = np.concatenate([np.arange(i*(ek+eq), i*(ek+eq)+ek) for i in range(len(edges))])
    b = res[Kidx]
    P = np.eye(dK.shape[0]) - dK @ pinv(dK)          # projector on ker dK^T (harmonic H1(K))
    return HQ, P @ b

# ---------------------------------------------------------------- E2 Corollary 1 at scale + lifting fraction
rng = np.random.default_rng(2026)
rows, maxerr = [], 0
for trial in range(600):
    nv = int(rng.integers(4, 9)); p = float(rng.uniform(0.3, 0.8))
    edges = random_graph(rng, nv, p); beta1 = len(edges) - nv + 1
    dk, dq = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    ek = int(rng.integers(1, dk + 2)); eq = int(rng.integers(1, dq + 1))
    bscale = float(rng.choice([0.0, 1.0]))
    dK, dQ, dF, _ = extension_sheaf(rng, nv, edges, dk, dq, ek, eq, bscale)
    HQ, Dmat = connecting_map(nv, edges, dk, dq, ek, eq, dK, dQ, dF)
    h0F, h0K, h0Q = nullity(dF), nullity(dK), HQ.shape[1]
    kerdel = h0Q - np.linalg.matrix_rank(Dmat, 1e-8) if h0Q else 0
    h1K = dK.shape[0] - np.linalg.matrix_rank(dK, 1e-9)
    maxerr = max(maxerr, abs(h0F - (h0K + kerdel)))
    rows.append(dict(generic=(kerdel == max(0, h0Q - h1K)), beta1=beta1, h0Q=h0Q, kerdel=kerdel, h1K=h1K, b=bscale, h0F=h0F, h0K=h0K))
lf = lambda rs: np.mean([r["kerdel"] / r["h0Q"] for r in rs if r["h0Q"] > 0])
OUT["E2"] = dict(n=len(rows), max_abs_violation=int(maxerr),
                 frac_with_H0Q=np.mean([r["h0Q"] > 0 for r in rows]),
                 lift_frac_B0=lf([r for r in rows if r["b"] == 0]),
                 lift_frac_B1=lf([r for r in rows if r["b"] == 1]),
                 overstate_frac=np.mean([r["kerdel"] < r["h0Q"] for r in rows if r["h0Q"] > 0 and r["b"] == 1]),
                 generic_rank_match=float(np.mean([r["generic"] for r in rows if r["b"] == 1])),
                 h1K_zero_but_obstructed=int(sum(r["h1K"] == 0 and r["kerdel"] < r["h0Q"] for r in rows)))
# lifting fraction versus H1(K) dimension for B != 0
byh = {}
for r in rows:
    if r["b"] == 1 and r["h0Q"] > 0: byh.setdefault(min(r["h1K"], 8), []).append(r["kerdel"] / r["h0Q"])
OUT["E2"]["lift_frac_by_h1K"] = {int(k): (float(np.mean(v)), len(v)) for k, v in sorted(byh.items())}

# ---------------------------------------------------------------- E3 obstruction amplitude: continuous vs discrete
# triangle family b = s*(1,1,1)+(1,1,-2): eps linear in s; dims constant
ss = np.linspace(-1, 1, 81); eps_line = [tri(s * np.ones(3) + np.array([1, 1, -2.]))[1] for s in ss]
OUT["E3"] = dict(slope=float(np.polyfit(np.abs(ss), eps_line, 1)[0]), expected=np.sqrt(3))
# random-sheaf obstruction amplitude scales linearly with ||B||
rng = np.random.default_rng(7); nv = 6; edges = random_graph(rng, nv, 0.6)
amps = []
base = extension_sheaf(rng, nv, edges, 2, 2, 2, 1, 1.0)
for sc in [0, .25, .5, 1, 2, 4]:
    rng2 = np.random.default_rng(7); random_graph(rng2, nv, 0.6)
    dK, dQ, dF, _ = extension_sheaf(rng2, nv, edges, 2, 2, 2, 1, sc)
    HQ, Dm = connecting_map(nv, edges, 2, 2, 2, 1, dK, dQ, dF)
    amps.append((sc, float(np.linalg.norm(Dm, 2))))
OUT["E3"]["random_sheaf_amp"] = amps

# ---------------------------------------------------------------- E4 exact-but-fragile lifts (Prop 1)
# K on triangle as a twisted local system: source gains g, product = holonomy h. H1(K)=0 iff h != 1
def DK_hol(h):
    g = [1.0, 1.0, h]
    return np.array([[-g[0], 1, 0], [0, -g[1], 1], [1, 0, -g[2]]])
hs = 1 - np.logspace(-6, 0, 40); b = np.array([1., 1, 1])
fr = []
for hh in hs:
    Dh = DK_hol(hh); s = np.linalg.svd(Dh, compute_uv=False); sp = s[s > 1e-12].min()
    k = -pinv(Dh) @ b; fr.append((1 - hh, sp, np.linalg.norm(k), np.linalg.norm(Dh @ k + b), np.linalg.norm(b) / sp))
fr = np.array(fr)
OUT["E4"] = dict(max_resid=float(fr[:, 3].max()), corr_at_1e6=float(fr[0, 2]), bound_at_1e6=float(fr[0, 4]),
                 bound_holds=bool((fr[:, 2] <= fr[:, 4] * (1 + 1e-9)).all()),
                 loglog_slope=float(np.polyfit(np.log(fr[:, 0]), np.log(fr[:, 2]), 1)[0]))

# ---------------------------------------------------------------- E5 leakage of approximately private contrasts (Prop 2)
# two-node example, restriction (s,p) -> s + mu p ; z = private contrast (0,1,0,-1)
leak = {}
for mu in [0.0, 1e-3, 1e-2, 1e-1]:
    dd = np.array([[-1, -mu, 1, mu]]); L = dd.T @ dd; zz = np.array([0, 1, 0, -1.])
    eps_ = np.linalg.norm(dd @ zz); nrm = np.linalg.norm(dd.T, 2)
    ts = np.logspace(-1, 4, 60)
    obs = np.array([np.linalg.norm(sla.expm(-t * L) @ zz - zz) for t in ts])
    leak[mu] = (ts, obs, ts * nrm * eps_)
OUT["E5"] = {str(mu): dict(eps=float(np.linalg.norm(np.array([[-1, -mu, 1, mu]]) @ np.array([0, 1, 0, -1.]))),
                           final_loss=float(v[1][-1]), bound_ok=bool((v[1] <= v[2] + 1e-12).all()),
                           t_half=float(v[0][np.argmax(v[1] >= 0.5 * v[1][-1])]) if v[1][-1] > 1e-9 else None)
             for mu, v in leak.items()}

# ---------------------------------------------------------------- E6 tightness of Prop 4 and Prop 5
rng = np.random.default_rng(11); ratios = []; ratios_cue = []
for _ in range(2000):
    n = 10; A = rng.standard_normal((7, n)); E = 0.02 * rng.standard_normal((7, n)); z = rng.standard_normal(n)
    L, Lp = A.T @ A, (A + E).T @ (A + E); tau = rng.uniform(0.01, 3)
    obs = np.linalg.norm(np.linalg.solve(np.eye(n) + tau * L, z) - np.linalg.solve(np.eye(n) + tau * Lp, z))
    ratios.append(obs / (tau * np.linalg.norm(L - Lp, 2) * np.linalg.norm(z)))
    # adversarial cue: top singular vector of resolvent difference
    Dd = np.linalg.inv(np.eye(n) + tau * L) - np.linalg.inv(np.eye(n) + tau * Lp)
    ratios_cue.append(np.linalg.norm(Dd, 2) / (tau * np.linalg.norm(L - Lp, 2)))
ratios, ratios_cue = np.array(ratios), np.array(ratios_cue)
prr = []
for _ in range(3000):
    n = 6; T = rng.standard_normal((n, n)); s = np.linalg.svd(T, compute_uv=False); a, bb = s.min(), s.max()
    G = rng.standard_normal((n, n)) * rng.uniform(0.1, 3, n); S = G @ G.T
    ratio = PR(T @ S @ T.T) / PR(S); prr.append((ratio, (a / bb) ** 4, (bb / a) ** 4, PR(S), PR(T @ S @ T.T)))
prr = np.array(prr)
OUT["E6"] = dict(prop4_median=float(np.median(ratios)), prop4_max=float(ratios.max()),
                 prop4_worstcue_median=float(np.median(ratios_cue)), prop4_worstcue_max=float(ratios_cue.max()),
                 prop5_violations=int(((prr[:, 0] < prr[:, 1] - 1e-12) | (prr[:, 0] > prr[:, 2] + 1e-12)).sum()),
                 prop5_frac_PR_increase=float((prr[:, 0] > 1).mean()),
                 prop5_median_logwidth=float(np.median(np.log10(prr[:, 2] / prr[:, 1]))))

# ---------------------------------------------------------------- E7 in-silico Hypothesis 1 / Conjecture 1
# 8 regions, stalk R^4 = rotated (shared rule s in R^2, private detail p_v in R^2); interfaces expose s.
def make_net(rng, nv=8):
    edges = [(i, (i + 1) % nv) for i in range(nv)] + [(0, 4), (2, 6)]
    M = [np.linalg.qr(rng.standard_normal((4, 4)))[0] for _ in range(nv)]
    return edges, M
def sheaf_L(edges, M, nv, theta=0.0, rng=None):
    R = {}
    for i, (s, t) in enumerate(edges):
        for end, v in (("s", s), ("t", t)):
            P = np.hstack([np.eye(2), np.zeros((2, 2))])
            if theta and rng is not None:   # misspecified interface: rotate exposed plane by angle theta into detail
                c, sn = np.cos(theta), np.sin(theta)
                P = np.hstack([c * np.eye(2), sn * np.linalg.qr(rng.standard_normal((2, 2)))[0]])
            R[(i, end)] = P @ M[v].T
    dF = coboundary(nv, edges, 4, 2, R); return dF.T @ dF
def scalar_L(edges, nv):
    Bi = np.zeros((len(edges), nv))
    for i, (s, t) in enumerate(edges): Bi[i, s], Bi[i, t] = -1, 1
    return np.kron(Bi.T @ Bi, np.eye(4))
RULES = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1.]])
DET = np.array([[1, 0], [0, 1], [-1, 0], [0, -1.]])
def sample(rng, M, nv, n, sigma, combos):
    """combos: list of allowed (rule, detail-of-region-0) pairs; other regions' details random."""
    X, Y, ys, yd = [], [], [], []
    for _ in range(n):
        r, d0 = combos[rng.integers(len(combos))]
        dets = [d0] + list(rng.integers(0, 4, nv - 1))
        x = np.concatenate([M[v] @ np.concatenate([RULES[r], DET[dets[v]]]) for v in range(nv)])
        X.append(x); Y.append(x + sigma * rng.standard_normal(x.size)); ys.append(r); yd.append(dets)
    return np.array(X), np.array(Y), np.array(ys), np.array(yd)
def decode(est, M, nv, ys, yd):
    """Template decoding of rule (pooled across regions, in each region's own frame) and of each region's detail."""
    sh, dt = [], []
    for v in range(nv):
        loc = est[:, v*4:(v+1)*4] @ M[v]         # back to (s,p) local frame (a calibrated decoder)
        sh.append(loc[:, :2]); dt.append(np.argmax(loc[:, 2:] @ DET.T, 1) == yd[:, v])
    s0 = sh[0]; acc_s0 = (np.argmax(s0 @ RULES.T, 1) == ys).mean()   # rule read from region 0 only
    return acc_s0, np.mean(dt), np.mean(dt[0])
rng = np.random.default_rng(5); nv = 8; edges, M = make_net(rng, nv)
Ls, Lc_raw = sheaf_L(edges, M, nv), scalar_L(edges, nv)
BM = sla.block_diag(*M); Lc = BM @ Lc_raw @ BM.T      # scalar consensus in the registered (s,p) frame
allc = [(r, d) for r in range(4) for d in range(4)]
held = [(r, d) for (r, d) in allc if (r + d) % 4 == 0]; train = [c for c in allc if c not in held]
sigma = 1.2
Xte, Yte, yste, ydte = sample(rng, M, nv, 4000, sigma, held)   # held-out combinations only
res = {"raw": decode(Yte, M, nv, yste, ydte)}
taus = np.logspace(-2, 2, 25); fr_sheaf, fr_scal, fr_raw = [], [], []
for tau in taus:
    fr_sheaf.append(decode(np.linalg.solve(np.eye(nv*4) + tau * Ls, Yte.T).T, M, nv, yste, ydte))
    fr_scal.append(decode(np.linalg.solve(np.eye(nv*4) + tau * Lc, Yte.T).T, M, nv, yste, ydte))
    fr_raw.append(decode(np.linalg.solve(np.eye(nv*4) + tau * Lc_raw, Yte.T).T, M, nv, yste, ydte))
fr_sheaf, fr_scal, fr_raw = np.array(fr_sheaf), np.array(fr_scal), np.array(fr_raw)
# misspecification sweep at tau = 3
mis = []
for th in [0, 0.1, 0.2, 0.4, 0.8, np.pi / 2]:
    rr = np.random.default_rng(99); Lm = sheaf_L(edges, M, nv, th, rr) if th else Ls
    mis.append((th, *decode(np.linalg.solve(np.eye(nv*4) + 3 * Lm, Yte.T).T, M, nv, yste, ydte)))
# Conjecture 1 analogue: unconstrained block-linear denoiser (ridge Y->X) learned from n samples (train combos)
lc, lsc = [], []
for n in [16, 32, 64, 128, 256, 512, 2048]:
    accs, lsh = [], []
    for rep in range(5):
        rr = np.random.default_rng(1000 + 10 * n + rep)
        Xtr, Ytr, ystr, _ = sample(rr, M, nv, n, sigma, train)
        # learned-interface sheaf: per region, least-squares rule decoder from the same n samples is the restriction
        R = {}
        for v in range(nv):
            Yv = Ytr[:, v*4:(v+1)*4]; Sv = RULES[ystr]
            Rv = np.linalg.lstsq(Yv, Sv, rcond=None)[0].T          # 2x4
            for i, (a_, b_) in enumerate(edges):
                if a_ == v: R[(i, "s")] = Rv
                if b_ == v: R[(i, "t")] = Rv
        dFl = coboundary(nv, edges, 4, 2, R); Ll = dFl.T @ dFl
        bl = max((decode(np.linalg.solve(np.eye(32) + t_ * Ll, Yte.T).T, M, nv, yste, ydte) for t_ in [0.3, 1, 3, 10, 30, 100]),
                 key=lambda a: a[0] + a[1])
        lsh.append(bl)
        best = None
        for lam in [1e-2, 1e-1, 1, 10, 100]:
            W = np.linalg.solve(Ytr.T @ Ytr + lam * np.eye(32), Ytr.T @ Xtr)
            a = decode(Yte @ W, M, nv, yste, ydte)
            if best is None or a[0] + a[1] > best[0] + best[1]: best = a     # oracle-tuned (favours the baseline)
        accs.append(best)
    lc.append((n, *np.mean(accs, 0))); lsc.append((n, *np.mean(lsh, 0)))
best_sheaf = fr_sheaf[np.argmax(fr_sheaf[:, :2].sum(1))]
# matched-shared comparison: scalar tau giving shared acc closest to the best sheaf shared acc
i_sc = np.argmin(np.abs(fr_scal[:, 0] - best_sheaf[0]))
OUT["E7"] = dict(sigma=sigma, raw=res["raw"], best_sheaf=best_sheaf.tolist(),
                 best_sheaf_tau=float(taus[np.argmax(fr_sheaf[:, :2].sum(1))]),
                 rawframe_scalar_max_shared=float(fr_raw[:, 0].max()),
                 scalar_max_shared=float(fr_scal[:, 0].max()),
                 scalar_detail_at_its_max_shared=float(fr_scal[np.argmax(fr_scal[:, 0]), 1]),
                 scalar_matched=fr_scal[i_sc].tolist(), misspec=mis, learning_curve=lc, learned_sheaf_curve=lsc,
                 n_params_block=32 * 32, n_params_sheaf_interface=len(edges) * 2 * 2 * 4)

json.dump(OUT, open("results.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
ax[0].plot(ss, eps_line, color=C1); ax[0].set_xlabel("cycle offset s"); ax[0].set_ylabel("ε(q)")
ax[0].set_title("(a) amplitude varies, dims fixed", fontsize=8)
ax[1].loglog(fr[:, 0], fr[:, 2], color=C1, label="‖k*‖"); ax[1].loglog(fr[:, 0], fr[:, 4], "--", color=C2, label="‖c‖/σ₊")
ax[1].set_xlabel("|1 − holonomy|"); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("(b) exact but fragile lift", fontsize=8)
for mu, col in zip([1e-3, 1e-2, 1e-1], [C1, C3, C2]):
    t, o, bd = leak[mu]; ax[2].loglog(t, o, color=col, label=f"μ={mu:g}"); ax[2].loglog(t, bd, ":", color=col)
    eps_ = 2 * mu; ax[2].axhline(eps_ / np.sqrt(2 + 2 * mu**2), color=col, lw=0.6, ls="--")
ax[2].set_ylim(1e-6, 3); ax[2].set_xlabel("time t"); ax[2].set_ylabel(r"$\|e^{-tL}z-z\|$")
ax[2].legend(frameon=False, fontsize=7); ax[2].set_title("(c) leakage vs Prop. 2 bound", fontsize=8)
fig.tight_layout(); fig.savefig("fig_obstruction.png")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(fr_scal[:, 0], fr_scal[:, 1], "o-", ms=2.5, color=C2, label="scalar consensus")
ax[0].plot(fr_sheaf[:, 0], fr_sheaf[:, 1], "o-", ms=2.5, color=C1, label="selective sheaf")
ax[0].plot(*res["raw"][:2], "k*", ms=8, label="no inference")
for th, a_s, a_d, _ in mis[1:]:
    ax[0].plot(a_s, a_d, "s", ms=3.5, color=C4)
ax[0].plot([], [], "s", color=C4, ms=3.5, label="misspecified maps (τ=3)")
ax[0].set_xlabel("held-out shared-rule accuracy (region 0)"); ax[0].set_ylabel("detail accuracy")
ax[0].legend(frameon=False, fontsize=7, loc="lower left"); ax[0].set_title("(a) τ sweep: rule vs detail frontier", fontsize=8)
lcA = np.array(lc)
ax[1].semilogx(lcA[:, 0], lcA[:, 1], "o-", color=C2, ms=3, label="block-linear: rule")
ax[1].semilogx(lcA[:, 0], lcA[:, 2], "o--", color=C2, ms=3, label="block-linear: detail (all)")
ax[1].semilogx(lcA[:, 0], lcA[:, 3], "o:", color=C4, ms=3, label="block-linear: detail (region 0)")
lsA = np.array(lsc)
ax[1].semilogx(lsA[:, 0], lsA[:, 1], "s-", color=C1, ms=3, label="learned-interface sheaf: rule")
ax[1].semilogx(lsA[:, 0], lsA[:, 2], "s--", color=C1, ms=3, label="learned-interface sheaf: detail (all)")
ax[1].semilogx(lsA[:, 0], lsA[:, 3], "s:", color=C3, ms=3, label="learned-interface sheaf: detail (region 0)")
ax[1].axhline(0.25, color=GR, lw=0.6, ls=":")
ax[1].set_xlabel("training samples (same data for both models)"); ax[1].set_ylabel("held-out accuracy")
ax[1].legend(frameon=False, fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.62)); ax[1].set_title("(b) matched-data learning curves", fontsize=8)
fig.tight_layout(); fig.savefig("fig_selective.png")
print(json.dumps(OUT, indent=1, default=float))
