"""Seeded verification harness for Sections 29-31 of 'Sheaf Based Programming Paradigm' (V7).

A  calibrated verification under Gaussian noise (Section 29)
B  group-valued transport, holonomy and certified synchronisation (Section 30)
C  restriction maps from population recordings: gauge, attenuation, holonomy test (Section 31)

Run: python3 paradigm_numerics.py [A|B|C ...]   -> paradigm_results.json, fig_*.png, golden_*.json
"""
import itertools, json, math, sys, time
from collections import deque
import numpy as np
import networkx as nx
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261021
EPS = np.finfo(float).eps
R = {}

# ----------------------------------------------------------------------------- shared helpers
def incidence(n, edges):
    D = np.zeros((len(edges), n))
    for e, (u, v) in enumerate(edges):
        D[e, u] -= 1.0; D[e, v] += 1.0                # (Ds)_uv = s_v - s_u
    return D

def triangles(G):
    return [tuple(t) for t in nx.enumerate_all_cliques(G) if len(t) == 3]

def d1_matrix(edges, tris):
    idx = {e: i for i, e in enumerate(edges)}
    E = np.zeros((len(tris), len(edges)))
    for t, (a, b, c) in enumerate(tris):              # (dg)_abc = g_bc - g_ac + g_ab
        E[t, idx[(b, c)]] += 1; E[t, idx[(a, c)]] -= 1; E[t, idx[(a, b)]] += 1
    return E

def rank_tol(A, rtol=1e-9):
    if A.size == 0: return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > rtol * s[0]).sum()) if s[0] > 0 else 0

def w_projector_range(A, W):
    """W-orthogonal projector onto im A (A may be rank deficient)."""
    if A.shape[1] == 0: return np.zeros((W.shape[0], W.shape[0]))
    Wh = np.sqrt(np.diag(W)) if np.allclose(W, np.diag(np.diag(W))) else None
    Wsq = np.diag(Wh) if Wh is not None else np.real(sqrtm_psd(W))
    Wisq = np.linalg.inv(Wsq)
    Q, s, _ = np.linalg.svd(Wsq @ A, full_matrices=False)
    r = int((s > 1e-10 * s[0]).sum()) if s.size and s[0] > 0 else 0
    Q = Q[:, :r]
    return Wisq @ (Q @ Q.T) @ Wsq

def sqrtm_psd(W):
    w, V = np.linalg.eigh(W); return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T

def null_basis(A, rtol=1e-10):
    if A.shape[0] == 0: return np.eye(A.shape[1])
    _, s, vt = np.linalg.svd(A)
    r = int((s > rtol * (s[0] if s.size else 1)).sum())
    return vt[r:].T

def connected_er(n, p, rng):
    while True:
        G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 31)))
        if nx.is_connected(G): return G

def fundamental_cycles(n, edges):
    """Signed fundamental cycle matrix Z (rows span ker D^T) from a BFS spanning forest."""
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    parent = {}; depth = {}
    for comp in nx.connected_components(G):
        r = min(comp); parent[r] = None; depth[r] = 0
        for u, v in nx.bfs_edges(G, r): parent[v] = u; depth[v] = depth[u] + 1
    tree = {tuple(sorted((v, parent[v]))) for v in parent if parent[v] is not None}
    idx = {e: i for i, e in enumerate(edges)}
    def path_to_root(v):
        out = []
        while parent[v] is not None: out.append((parent[v], v)); v = parent[v]
        return out
    Z = []
    for e in edges:
        if e in tree: continue
        u, v = e; z = np.zeros(len(edges)); z[idx[e]] = 1.0           # traverse u -> v
        # then v -> root -> u along tree:  v up to root (reverse of root->v), root down to u
        for (a, b) in path_to_root(v):   # edge a->b oriented root-ward means we go b->a
            i = idx[tuple(sorted((a, b)))]; z[i] += -1.0 if (a, b) == edges[i] else 1.0
        for (a, b) in path_to_root(u):   # we go a->b (root to u)
            i = idx[tuple(sorted((a, b)))]; z[i] += 1.0 if (a, b) == edges[i] else -1.0
        Z.append(z)
    return np.array(Z) if Z else np.zeros((0, len(edges)))

# ============================================================================= A: Section 29
def part_A():
    rng = np.random.default_rng(SEED)
    out = {}
    # A1 calibration and independence on random clique complexes ---------------------------------
    rows = []
    for (n, p) in [(12, 0.45), (20, 0.30), (40, 0.15)]:
        G = connected_er(n, p, rng)
        edges = sorted(tuple(sorted(e)) for e in G.edges()); m = len(edges)
        D = incidence(n, edges); T = triangles(G); E = d1_matrix(edges, T) if T else np.zeros((0, m))
        assert np.abs(E @ D).max() == 0
        w = rng.uniform(0.2, 5.0, m); W = np.diag(w)
        PD = w_projector_range(D, W)
        KE = null_basis(E) if E.shape[0] else np.eye(m)
        PK = w_projector_range(KE, W)
        PH, PC = PK - PD, np.eye(m) - PK
        rD, rE = rank_tol(D), rank_tol(E) if E.size else 0
        h, c = m - rD - rE, rE
        # projector sanity: idempotent, W-self-adjoint, mutually W-orthogonal
        chk = max(np.abs(PH @ PH - PH).max(), np.abs(PC @ PC - PC).max(), np.abs(PH @ PC).max(),
                  np.abs(PH.T @ W - W @ PH).max(), np.abs(PC.T @ W - W @ PC).max())
        N = 40000; sigma = 1.0
        s_true = rng.normal(0, 1, n); mu = D @ s_true
        eps = rng.normal(0, 1, (N, m)) / np.sqrt(w)            # covariance W^{-1}
        g = mu + sigma * eps
        TH = np.einsum("ij,j,ij->i", g @ PH.T, w, g @ PH.T) / sigma ** 2
        TC = np.einsum("ij,j,ij->i", g @ PC.T, w, g @ PC.T) / sigma ** 2
        a = 0.05
        rejH = TH > stats.chi2.ppf(1 - a, h); rejC = TC > stats.chi2.ppf(1 - a, c) if c > 0 else np.zeros(N, bool)
        ksH = stats.kstest(TH, stats.chi2(h).cdf).pvalue
        ksC = stats.kstest(TC, stats.chi2(c).cdf).pvalue if c > 0 else float("nan")
        rows.append(dict(n=n, p=p, m=m, triangles=len(T), h=h, c=c, projector_check=float(chk), trials=N,
                         typeI_H=float(rejH.mean()), typeI_C=float(rejC.mean()), joint=float((rejH & rejC).mean()),
                         either=float((rejH | rejC).mean()), either_theory=1 - (1 - a) * (1 - a if c > 0 else 1),
                         corr_TH_TC=float(np.corrcoef(TH, TC)[0, 1]) if c > 0 else float("nan"),
                         ks_p_H=float(ksH), ks_p_C=float(ksC), mean_TH=float(TH.mean()), mean_TC=float(TC.mean()) if c > 0 else 0.0))
    out["A1_calibration"] = rows

    # A2 power versus cyclic class on a weighted k-cycle (no 2-cells): theory = noncentral chi2(1) --
    pw = []
    k = 6; edges = [tuple(sorted((i, (i + 1) % k))) for i in range(k)]
    D = incidence(k, edges); z = np.ones(k); z[-1] = -1.0; assert np.allclose(z @ D, 0)
    w = np.array([4.0, 1.0, 4.0, 1.0, 2.0, 0.5]); W = np.diag(w); PD = w_projector_range(D, W)
    zWz = float(z @ (z / w)); crit = stats.chi2.ppf(0.95, 1)
    for kappa in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        N = 40000
        v = (z / w) / zWz                                  # any vector with z.v = 1 gives the same class
        mu = D @ rng.normal(0, 1, k) + kappa * v
        g = mu + rng.normal(0, 1, (N, k)) / np.sqrt(w)
        r = g - g @ PD.T
        TH = np.einsum("ij,j,ij->i", r, w, r)
        lam = kappa ** 2 / zWz
        pw.append(dict(kappa=kappa, lam=lam, power_mc=float((TH > crit).mean()),
                       power_theory=float(stats.ncx2.sf(crit, 1, lam)) if lam > 0 else 0.05))
    out["A2_power"] = dict(k=k, weights=w.tolist(), zWinvz=zWz, rows=pw)

    # A3 minimal detectable class versus loop length (unit weights, alpha .05, power .8) ------------
    lam_star = optimize.brentq(lambda L: stats.ncx2.sf(stats.chi2.ppf(0.95, 1), 1, L) - 0.8, 0.1, 50)
    out["A3_detectable"] = dict(lam_star=lam_star,
                                rows=[dict(k=kk, kappa_min=math.sqrt(lam_star * kk)) for kk in [3, 6, 12, 24, 48]])

    # A4 cycle form equals projector form on graphs ---------------------------------------------------
    worst = 0.0; cnt = 0
    for _ in range(200):
        n = int(rng.integers(5, 40)); G = connected_er(n, min(1.0, 4.0 / n), rng)
        edges = sorted(tuple(sorted(e)) for e in G.edges()); m = len(edges)
        if m - n + 1 < 1: continue
        D = incidence(n, edges); Z = fundamental_cycles(n, edges)
        assert Z.shape[0] == m - n + 1 and np.abs(Z @ D).max() < 1e-12
        w = rng.uniform(0.2, 5.0, m); W = np.diag(w); PD = w_projector_range(D, W)
        g = rng.normal(0, 2, m); r = g - PD @ g
        a = float(r @ (w * r)); zg = Z @ g
        b = float(zg @ np.linalg.solve(Z @ np.diag(1 / w) @ Z.T, zg))
        worst = max(worst, abs(a - b) / max(1.0, a)); cnt += 1
    out["A4_cycle_form"] = dict(graphs=cnt, max_rel_diff=worst)

    # A5 misspecified metric: analyst uses W_used = W_true * exp(tau * xi) ---------------------------
    mis = []
    G = connected_er(30, 0.18, rng); edges = sorted(tuple(sorted(e)) for e in G.edges()); m = len(edges); n = 30
    D = incidence(n, edges); h = m - n + 1
    w_true = rng.uniform(0.2, 5.0, m)
    for tau in [0.0, 0.25, 0.5, 1.0]:
        rates = []
        for rep in range(20):                               # 20 different misspecification draws
            w_used = w_true * np.exp(tau * rng.normal(0, 1, m))
            PD = w_projector_range(D, np.diag(w_used))
            g = rng.normal(0, 1, (4000, m)) / np.sqrt(w_true)   # exact mean 0, true covariance W_true^{-1}
            r = g - g @ PD.T
            TH = np.einsum("ij,j,ij->i", r, w_used, r)
            rates.append(float((TH > stats.chi2.ppf(0.95, h)).mean()))
        mis.append(dict(tau=tau, typeI_median=float(np.median(rates)), typeI_min=float(min(rates)), typeI_max=float(max(rates))))
    out["A5_misspecified"] = dict(m=m, h=h, rows=mis)

    # A6 unknown sigma: replicate F test ---------------------------------------------------------------
    k = 6; edges = [tuple(sorted((i, (i + 1) % k))) for i in range(k)]; D = incidence(k, edges)
    w = np.array([4.0, 1.0, 4.0, 1.0, 2.0, 0.5]); PD = w_projector_range(D, np.diag(w))
    Rr = 4; sigma = 0.7; Ntr = 20000; dfd = k * (Rr - 1)
    gs = sigma * rng.normal(0, 1, (Ntr, Rr, k)) / np.sqrt(w)
    gbar = gs.mean(axis=1)
    s2 = np.einsum("trj,j,trj->t", gs - gbar[:, None, :], w, gs - gbar[:, None, :]) / dfd
    r = gbar - gbar @ PD.T
    F = (Rr * np.einsum("ij,j,ij->i", r, w, r) / 1) / s2
    out["A6_F_test"] = dict(replicates=Rr, df=(1, dfd), sigma_true=sigma, typeI=float((F > stats.f.ppf(0.95, 1, dfd)).mean()),
                            ks_p=float(stats.kstest(F, stats.f(1, dfd).cdf).pvalue))
    R["A"] = out

    # figure: power curve
    rr = out["A2_power"]["rows"]; ks = np.linspace(0, 4, 200)
    plt.figure(figsize=(6.0, 3.2))
    plt.plot(ks, [stats.ncx2.sf(crit, 1, kk ** 2 / zWz) if kk > 0 else 0.05 for kk in ks], lw=1.6, label="noncentral χ²₁ theory")
    plt.plot([x["kappa"] for x in rr], [x["power_mc"] for x in rr], "o", ms=5, label="Monte Carlo, 40 000 trials")
    plt.axhline(0.05, color="0.6", lw=0.8, ls="--")
    plt.xlabel("cyclic class κ"); plt.ylabel("rejection rate of exactness (α = 0.05)")
    plt.title("Weighted 6-cycle: power is set by the repair cost κ² / zᵀW⁻¹z", fontsize=9)
    plt.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("fig_power.png", dpi=200); plt.close()

# ============================================================================= B: Section 30
def rotz(axis, ang):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

def order_of(M, maxo=12):
    P = np.eye(3)
    for o in range(1, maxo + 1):
        P = P @ M
        if np.allclose(P, np.eye(3), atol=1e-9): return o
    return None

def part_B():
    rng = np.random.default_rng(SEED + 1)
    out = {}
    # B1 icosahedral counterexample: cycle-basis holonomies trivial, assignment not gauge-trivial ------
    phi = (1 + 5 ** 0.5) / 2
    verts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    verts = np.array(verts, float)
    v0 = verts[0]
    nbrs = [v for v in verts if abs(np.linalg.norm(v - v0) - 2.0) < 1e-9]
    face_c = (v0 + nbrs[0] + [v for v in nbrs if abs(np.linalg.norm(v - nbrs[0]) - 2.0) < 1e-9][0]) / 3
    g5, g3 = rotz(v0, 2 * math.pi / 5), rotz(face_c, 2 * math.pi / 3)
    # close the group
    elems = [np.eye(3)]; frontier = [np.eye(3)]
    def known(M): return any(np.allclose(M, X, atol=1e-9) for X in elems)
    while frontier:
        nxt = []
        for X in frontier:
            for gen in (g5, g3):
                Y = X @ gen
                if not known(Y): elems.append(Y); nxt.append(Y)
        frontier = nxt
    assert len(elems) == 60
    # choose x (order 2), y (order 3) with xy of order 5  -> generate A5
    pair = None
    for X in elems:
        if order_of(X) != 2: continue
        for Y in elems:
            if order_of(Y) == 3 and order_of(X @ Y) == 5: pair = (X, Y); break
        if pair: break
    x, y = pair
    # shortest words for each element over alphabet a, A=a^-1, b, B=b^-1
    gens = {"a": x, "A": x.T, "b": y, "B": y.T}
    words = {0: ""}; mats = [np.eye(3)]
    def find(M):
        for i, X in enumerate(mats):
            if np.allclose(M, X, atol=1e-9): return i
        return None
    q = deque([0])
    while q:
        i = q.popleft()
        for ch, Gm in gens.items():
            Y = mats[i] @ Gm
            if find(Y) is None: mats.append(Y); words[len(mats) - 1] = words[i] + ch; q.append(len(mats) - 1)
    assert len(mats) == 60
    def ev(word):
        M = np.eye(3)
        for ch in word: M = M @ gens[ch]
        return M
    inv = {"a": "A", "A": "a", "b": "B", "B": "b"}
    def winv(wd): return "".join(inv[c] for c in reversed(wd))
    def commutator_word(target):
        best = None
        for i in range(60):
            for j in range(60):
                U, V = mats[i], mats[j]
                if np.allclose(U @ V @ U.T @ V.T, target, atol=1e-9):
                    wd = words[i] + words[j] + winv(words[i]) + winv(words[j])
                    if best is None or len(wd) < len(best): best = wd
        return best
    cx, cy = commutator_word(x), commutator_word(y)
    w1, w2 = "a" + winv(cx), "b" + winv(cy)
    def expsum(wd): return (wd.count("a") - wd.count("A"), wd.count("b") - wd.count("B"))
    # realise on the bowtie graph: triangles 0-1-2-0 (generator a) and 0-3-4-0 (generator b)
    # transports: identity on every edge except 2->0 carries x and 4->0 carries y
    hol_w1, hol_w2 = ev(w1), ev(w2)
    out["B1_icosahedral"] = dict(word_w1=w1, word_w2=w2, len_w1=len(w1), len_w2=len(w2),
                                 exponent_sums_w1=expsum(w1), exponent_sums_w2=expsum(w2),
                                 max_dev_hol_w1=float(np.abs(hol_w1 - np.eye(3)).max()), max_dev_hol_w2=float(np.abs(hol_w2 - np.eye(3)).max()),
                                 fundamental_a_rotation_deg=float(math.degrees(math.acos(max(-1, min(1, (np.trace(x) - 1) / 2))))),
                                 fundamental_b_rotation_deg=float(math.degrees(math.acos(max(-1, min(1, (np.trace(y) - 1) / 2))))),
                                 x=x.round(12).tolist(), y=y.round(12).tolist())

    # B2 U(1) winding: real-valued repair versus circle repair on a triangle ----------------------------
    deg = math.pi / 180
    th = np.array([170.0, 170.0, -10.0]) * deg              # edges (01,12,02); kappa = g01 + g12 - g02
    kappa_real = th[0] + th[1] - th[2]                       # 350 deg as a real number
    kappa_circ = (kappa_real + math.pi) % (2 * math.pi) - math.pi   # class on the circle, in [-pi, pi)
    zc = np.array([1.0, 1.0, -1.0])
    out["B2_winding"] = dict(measurements_deg=[170.0, 170.0, -10.0],
                             real_class_deg=math.degrees(kappa_real), circle_class_deg=math.degrees(kappa_circ),
                             winding=int(round((kappa_real - kappa_circ) / (2 * math.pi))),
                             real_repair_deg=(-math.degrees(kappa_real) * zc / 3).tolist(),
                             circle_repair_deg=(-math.degrees(kappa_circ) * zc / 3).tolist())

    # B3 angular synchronisation with certificates on random graphs ------------------------------------
    def sync_matrix(n, edges, meas, w):
        """M with f(z) = sum_e w_e |z_v - r_uv z_u|^2 = z^H M z  (r_uv measures s_u^{-1} s_v)."""
        M = np.zeros((n, n), complex)
        for (u, v), r, we in zip(edges, meas, w):
            M[u, u] += we; M[v, v] += we
            M[u, v] += -we * np.conj(r); M[v, u] += -we * r
        return M
    def gpm(M, z0, iters=2000, tol=1e-13):
        """Generalised power method for min z^H M z over |z_u| = 1: z <- proj((cI - M) z)."""
        c = np.linalg.eigvalsh(M)[-1] + 1.0
        z = z0 / np.abs(z0)
        for _ in range(iters):
            y = c * z - M @ z
            znew = y / np.abs(y)
            if np.abs(znew - z).max() < tol: z = znew; break
            z = znew
        return z
    def certificate(M, z):
        lam = np.real(np.conj(z) * (M @ z))
        S = M - np.diag(lam)
        ev = np.linalg.eigvalsh(S)
        return float(ev[0]), float(ev[1]), float(np.real(np.conj(z) @ M @ z)), float(lam.sum())
    rows = []
    n = 60
    for sigma in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
        cert = 0; rms_cert = []; rms_lin = []; trials = 40; gaps = []
        for _ in range(trials):
            G = connected_er(n, 0.15, rng); edges = sorted(tuple(sorted(e)) for e in G.edges())
            theta = rng.uniform(-math.pi, math.pi, n); ztrue = np.exp(1j * theta)
            noise = rng.normal(0, sigma, len(edges))
            meas = np.array([np.exp(1j * (theta[v] - theta[u] + e)) for (u, v), e in zip(edges, noise)])
            w = np.ones(len(edges))
            M = sync_matrix(n, edges, meas, w)
            vals, vecs = np.linalg.eigh(M)
            z = gpm(M, vecs[:, 0])
            lmin, l2, f, lsum = certificate(M, z)
            scale = np.abs(M).sum(axis=1).max()
            ok = lmin >= -1e-9 * scale
            cert += ok; gaps.append(max(0.0, -lmin) * n)
            def align_rms(zz):
                ph = np.angle(np.vdot(zz, ztrue)); e = np.angle(zz * np.exp(1j * ph) / ztrue); return float(np.sqrt(np.mean(e ** 2)))
            rms_cert.append(align_rms(z))
            # baseline: unwrap along BFS tree, then real least squares on unwrapped edge angles
            parent = {0: None}; ang = {0: 0.0}
            for a_, b_ in nx.bfs_edges(G, 0):
                i = edges.index(tuple(sorted((a_, b_))))
                d = np.angle(meas[i]) if (a_, b_) == edges[i] else -np.angle(meas[i])
                ang[b_] = ang[a_] + d
            gl = []
            for (u, v), r in zip(edges, meas):
                pred = ang[v] - ang[u]; d = np.angle(r)
                gl.append(d + 2 * math.pi * round((pred - d) / (2 * math.pi)))
            Dm = incidence(n, edges)[:, 1:]
            sol = np.linalg.lstsq(Dm, np.array(gl), rcond=None)[0]
            zl = np.exp(1j * np.concatenate([[0.0], sol]))
            rms_lin.append(align_rms(zl))
        rows.append(dict(sigma=sigma, trials=trials, certified=cert / trials, rms_certified_method=float(np.median(rms_cert)),
                         rms_unwrap_LS=float(np.median(rms_lin)), max_gap_bound=float(max(gaps))))
    out["B3_sync"] = dict(n=n, p=0.15, rows=rows)

    # B4 SE(2) loop: certified joint repair versus the linearised (abelian) prediction ------------------
    def se2_problem(nv, edges, rot, trans, kw, tw):
        """Reduced complex form. Unknowns z (unit rotations) and t (complex translations).
        f = sum kw |z_v - r_uv z_u|^2 + tw |t_v - t_u - z_u tau_uv|^2 ;  min over t is exact (gauge t_0 = 0)."""
        m = len(edges)
        Mr = sync_matrix(nv, edges, rot, kw)
        Dt = np.zeros((m, nv), complex); B = np.zeros((m, nv), complex)
        for e, (u, v) in enumerate(edges):
            Dt[e, u] = -1; Dt[e, v] = 1; B[e, u] = trans[e]
        Om = np.diag(tw.astype(complex))
        Dg = Dt[:, 1:]
        A = Dg.conj().T @ Om @ Dg
        P = Dg @ np.linalg.solve(A, Dg.conj().T @ Om)          # Omega-orthogonal projector onto im D (gauge fixed)
        Mt = B.conj().T @ Om @ (np.eye(m) - P) @ B
        M = Mr + (Mt + Mt.conj().T) / 2
        def tsolve(z):
            tt = np.linalg.solve(A, Dg.conj().T @ Om @ (B @ z)); return np.concatenate([[0], tt])
        return M, tsolve
    def certify_min(M, z):
        lam = np.real(np.conj(z) * (M @ z)); S = M - np.diag(lam)
        return float(np.linalg.eigvalsh(S)[0]), float(np.real(np.conj(z) @ M @ z))
    corners = 4; L = 4.0
    loop_edges = [(i, (i + 1) % corners) for i in range(corners)]
    rows = []
    for delta_deg in [0, 5, 15, 30, 60, 90]:
        d = math.radians(delta_deg)
        # true walk: square, each segment rotates heading by +90 deg; measured last turn is 90 + delta (heading wormhole)
        turns = [math.pi / 2] * corners; turns[-1] += d
        rot = np.array([np.exp(1j * a) for a in turns])
        trans = np.array([L + 0j] * corners)                    # displacement in the frame of the start of each segment
        kw = np.array([4.0, 1.0, 4.0, 1.0]); tw = np.array([4.0, 1.0, 4.0, 1.0])
        # edges as (u,v) with u<v orientation: (0,1),(1,2),(2,3),(3,0)-> need (0,3) with inverse measurement
        # edges in traversal orientation: odometry of segment i is expressed in the frame where it starts
        E2 = list(loop_edges); R2 = rot.copy(); T2 = trans.copy()
        M, tsolve = se2_problem(corners, E2, R2, T2, kw, tw)
        vals, vecs = np.linalg.eigh(M)
        best = None
        for start in [vecs[:, 0]] + [np.exp(1j * rng.uniform(-math.pi, math.pi, corners)) for _ in range(30)]:
            z = gpm(M, start * np.conj(start[0]) / abs(start[0]))
            lmin, f = certify_min(M, z)
            if best is None or f < best[1] - 1e-12: best = (z, f, lmin)
        z, f, lmin = best
        z = z * np.conj(z[0])                                     # gauge: z_0 = 1
        t = tsolve(z) * 1.0
        # brute-force check of the global minimum over the 3 free headings (grid + polish)
        def fobj(angles):
            zz = np.concatenate([[1.0 + 0j], np.exp(1j * angles)]); return float(np.real(np.conj(zz) @ M @ zz))
        bf = min((optimize.minimize(fobj, x0, method="BFGS").fun for x0 in rng.uniform(-math.pi, math.pi, (60, corners - 1))))
        # implied corrections: heading correction per segment and translation residual per segment
        head_corr = []; pos_corr = []
        for (u, v), r, tr in zip(E2, R2, T2):
            rel = np.conj(z[u]) * z[v]                            # fitted relative rotation
            head_corr.append(math.degrees(np.angle(rel / r)))
            fitted_tr = np.conj(z[u]) * (t[v] - t[u])
            pos_corr.append(abs(fitted_tr - tr))
        # (i) first-order (coupled, abelian) prediction: one Gauss-Newton step from the consistent square
        def residual(p, R2_, T2_):
            ph = np.concatenate([[0.0], p[:corners - 1]]); zz = np.exp(1j * ph)
            tt = np.concatenate([[0j], p[corners - 1::2] + 1j * p[corners::2]])
            res = []
            for (u, v), r_, tr_, a_, b_ in zip(E2, R2_, T2_, kw, tw):
                c1 = math.sqrt(a_) * (zz[v] - zz[u] * r_); c2 = math.sqrt(b_) * (tt[v] - tt[u] - zz[u] * tr_)
                res += [c1.real, c1.imag, c2.real, c2.imag]
            return np.array(res)
        p0 = np.array([math.pi / 2, math.pi, 3 * math.pi / 2, L, 0.0, L, L, 0.0, L])
        r0 = residual(p0, R2, T2); J = np.zeros((r0.size, p0.size)); hh = 1e-7
        for q in range(p0.size):
            e_ = np.zeros(p0.size); e_[q] = hh
            J[:, q] = (residual(p0 + e_, R2, T2) - residual(p0 - e_, R2, T2)) / (2 * hh)
        p1 = p0 - np.linalg.lstsq(J, r0, rcond=None)[0]
        z_lin = np.exp(1j * np.concatenate([[0.0], p1[:corners - 1]]))
        lin_head = [math.degrees(np.angle((np.conj(z_lin[u]) * z_lin[v]) / r_)) for (u, v), r_ in zip(E2, R2)]
        cost_lin = float(np.real(np.conj(z_lin) @ M @ z_lin))
        # (ii) decoupled two-stage heuristic: headings by Proposition 7.3 on angles, then translations
        zc = np.ones(corners)                                   # every edge traversed along its orientation
        ang_meas = np.array([np.angle(r_) for r_ in R2])
        k_ang = float(zc @ ang_meas); k_ang = (k_ang + math.pi) % (2 * math.pi) - math.pi
        ccorr = -k_ang * (zc / kw) / (zc @ (zc / kw))
        fitted = ang_meas + ccorr
        ph2 = [0.0]
        for q in range(corners - 1): ph2.append(ph2[-1] + fitted[q])          # tree 0-1-2-3
        z_two = np.exp(1j * np.array(ph2))
        cost_two = float(np.real(np.conj(z_two) @ M @ z_two))
        Erev = E2[:-1] + [(0, corners - 1)]; Rrev = np.concatenate([R2[:-1], [np.conj(R2[-1])]])
        Trev = np.concatenate([T2[:-1], [-np.conj(R2[-1]) * T2[-1]]])
        Mrev, _ = se2_problem(corners, Erev, Rrev, Trev, kw, tw)
        frev = min(float(np.real(np.conj(zz) @ Mrev @ zz)) for zz in
                   [gpm(Mrev, st) for st in [np.linalg.eigh(Mrev)[1][:, 0]] + [np.exp(1j * rng.uniform(-math.pi, math.pi, corners)) for _ in range(30)]])
        rows.append(dict(delta_deg=delta_deg, cost=f, bruteforce_min=bf, cert_lambda_min=lmin, cost_reversed_D=frev,
                         heading_corr_deg=head_corr, firstorder_heading_corr_deg=lin_head,
                         max_heading_diff_firstorder_deg=float(max(abs(a - b) for a, b in zip(head_corr, lin_head))),
                         cost_firstorder=cost_lin, cost_two_stage=cost_two,
                         twostage_heading_corr_deg=[math.degrees(c) for c in ccorr],
                         translation_residual_m=pos_corr))
    out["B4_se2_loop"] = dict(segment_length_m=L, weights=[4.0, 1.0, 4.0, 1.0], rows=rows)

    # B5 source dependence: where the heading discrepancy enters versus where the certified repair puts it
    src = []
    kw = np.array([4.0, 1.0, 4.0, 1.0]); tw = np.array([4.0, 1.0, 4.0, 1.0])
    abelian_share = (1 / kw) / np.sum(1 / kw)
    for where in range(corners):
        for delta_deg in [15, 60]:
            turns = [math.pi / 2] * corners; turns[where] += math.radians(delta_deg)
            E2 = list(loop_edges); R2 = np.array([np.exp(1j * a) for a in turns]); T2 = np.array([L + 0j] * corners)
            M, _ = se2_problem(corners, E2, R2, T2, kw, tw)
            best = None
            for start in [np.linalg.eigh(M)[1][:, 0]] + [np.exp(1j * rng.uniform(-math.pi, math.pi, corners)) for _ in range(40)]:
                z = gpm(M, start)
                lmin_, f_ = certify_min(M, z)
                if best is None or f_ < best[1] - 1e-12: best = (z, f_, lmin_)
            z, f_, lmin_ = best
            corr = [math.degrees(np.angle((np.conj(z[u]) * z[v]) / r_)) for (u, v), r_ in zip(E2, R2)]
            src.append(dict(source_edge="ABCD"[where], delta_deg=delta_deg, cert_lambda_min=lmin_,
                            certified_share=[-c / delta_deg for c in corr], abelian_share=abelian_share.tolist()))
    out["B5_source_dependence"] = src
    R["B"] = out

# ============================================================================= C: Section 31
def random_gl(k, rng, spread=0.4):
    Q1, _ = np.linalg.qr(rng.normal(size=(k, k))); Q2, _ = np.linalg.qr(rng.normal(size=(k, k)))
    return Q1 @ np.diag(np.exp(rng.normal(0, spread, k))) @ Q2

def simulate_population_data(rng, k, ntr, nneur, noise, model, phi=0.0, shear=0.0):
    """Three contexts 1,2,3; overlaps (1,2),(2,3),(3,1). Each overlap trial has a shared state o ~ N(0,I_k)
    presented twice (repeats a,b). Population i encodes its latent x_i = S_i o (frame S_i) except that, in the
    obstructed models, population 1 uses S_1 X on overlap (3,1) with X a rotation (phi) or a shear."""
    S = [random_gl(k, rng) for _ in range(3)]
    A = [rng.normal(0, 1, (nneur, k)) / math.sqrt(k) for _ in range(3)]
    X = np.eye(k)
    if model == "rotation":
        X = np.eye(k); c, s_ = math.cos(phi), math.sin(phi); X[0, 0] = c; X[0, 1] = -s_; X[1, 0] = s_; X[1, 1] = c
    elif model == "shear":
        X = np.eye(k); X[0, 1] = shear
    data = {}
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        o = rng.normal(0, 1, (k, ntr))
        Fi = S[i] @ (X if (model != "null" and (i, j) == (2, 0) and i == 0) else np.eye(k))
        Fj = S[j] @ (X if (model != "null" and (i, j) == (2, 0) and j == 0) else np.eye(k))
        reps = {}
        for rp in ("a", "b"):
            reps[rp] = (A[i] @ (Fi @ o) + noise * rng.normal(size=(nneur, ntr)),
                        A[j] @ (Fj @ o) + noise * rng.normal(size=(nneur, ntr)))
        data[(i, j)] = reps
    return data

def pca_gauges(data, k):
    """PCA basis per population from all of its trials (both overlaps, both repeats). Fixed thereafter."""
    pops = {0: [], 1: [], 2: []}
    for (i, j), reps in data.items():
        for rp in ("a", "b"):
            pops[i].append(reps[rp][0]); pops[j].append(reps[rp][1])
    Pi = {}
    for p, mats in pops.items():
        Y = np.concatenate(mats, axis=1); Y = Y - Y.mean(axis=1, keepdims=True)
        U, _, _ = np.linalg.svd(Y, full_matrices=False); Pi[p] = U[:, :k].T
    return Pi

def estimate_holonomy(data, Pi, method, idx=None):
    """Transitions T_ij on each overlap in the fixed PCA gauges; holonomy 0 -> 1 -> 2 -> 0."""
    T = {}
    for (i, j), reps in data.items():
        sel = slice(None) if idx is None else idx[(i, j)]
        Xi_a, Xj_a = Pi[i] @ reps["a"][0][:, sel], Pi[j] @ reps["a"][1][:, sel]
        Xi_b = Pi[i] @ reps["b"][0][:, sel]
        if method == "ols":
            T[(i, j)] = Xj_a @ Xi_a.T @ np.linalg.inv(Xi_a @ Xi_a.T)
        else:   # instrumental variable: repeat b of population i instruments repeat a
            T[(i, j)] = Xj_a @ Xi_b.T @ np.linalg.inv(Xi_a @ Xi_b.T)
    return T[(2, 0)] @ T[(1, 2)] @ T[(0, 1)]

def trace_invariants(H):
    k = H.shape[0]; P = np.eye(k); out = []
    for _ in range(k):
        P = P @ H; out.append(np.trace(P) - k)
    return np.array(out)

def wald_F(d, boots):
    p = d.size; Bn = boots.shape[0]
    C = np.cov(boots.T)
    W = float(d @ np.linalg.solve(C, d))
    Fs = W * (Bn - p) / (p * (Bn - 1))          # Hotelling-type correction for an estimated covariance
    return W, float(stats.f.sf(Fs, p, Bn - p))

def holonomy_test(data, k, method, rng, B=200):
    Pi = pca_gauges(data, k)
    H = estimate_holonomy(data, Pi, method)
    ntr = next(iter(data.values()))["a"][0].shape[1]
    boots = []
    for _ in range(B):
        idx = {key: rng.integers(0, ntr, ntr) for key in data}
        boots.append(estimate_holonomy(data, Pi, method, idx))
    boots = np.array(boots)
    W, p = wald_F((H - np.eye(k)).ravel(), boots.reshape(B, -1))
    Wt, pt = wald_F(trace_invariants(H), np.array([trace_invariants(Hb) for Hb in boots]))
    return dict(W=W, p=p, fro=float(np.linalg.norm(H - np.eye(k))), W_spec=Wt, p_spec=pt)

def part_C():
    rng = np.random.default_rng(SEED + 2)
    out = {}
    k, nneur = 3, 40
    conds = []
    # calibration under the gauge-trivial model: naive OLS versus repeat-instrument IV
    for ntr in [100, 400]:
        for noise in [0.5, 1.0]:
            for method in ["ols", "iv"]:
                reps = 150; rej = 0; fro = []
                for _ in range(reps):
                    data = simulate_population_data(rng, k, ntr, nneur, noise, "null")
                    res = holonomy_test(data, k, method, rng)
                    rej += res["p"] < 0.05; fro.append(res["fro"])
                conds.append(dict(model="null", ntr=ntr, noise=noise, method=method, reps=reps,
                                  reject_rate=rej / reps, median_fro=float(np.median(fro))))
    out["C1_calibration"] = conds
    # power under a rotational obstruction (IV), and the shear case: Wald versus spectral test
    pw = []
    for phi_deg in [0, 5, 10, 20, 40]:
        reps = 120; rej = 0
        for _ in range(reps):
            data = simulate_population_data(rng, k, 200, nneur, 1.0, "rotation", phi=math.radians(phi_deg))
            rej += holonomy_test(data, k, "iv", rng)["p"] < 0.05
        pw.append(dict(phi_deg=phi_deg, ntr=200, noise=1.0, power=rej / reps))
    out["C2_power_rotation"] = pw
    sh = []
    for shear in [0.0, 0.2, 0.4, 0.8]:
        reps = 120; rW = 0; rE = 0
        for _ in range(reps):
            data = simulate_population_data(rng, k, 200, nneur, 1.0, "shear", shear=shear)
            res = holonomy_test(data, k, "iv", rng)
            rW += res["p"] < 0.05; rE += res["p_spec"] < 0.05
        sh.append(dict(shear=shear, wald_power=rW / reps, spectral_power=rE / reps))
    out["C3_shear"] = sh
    R["C"] = out

# ============================================================================= main
if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C"]
    try:
        R.update(json.load(open("paradigm_results.json")))
    except Exception:
        pass
    for part in which:
        t0 = time.time(); {"A": part_A, "B": part_B, "C": part_C}[part](); print(part, "done", round(time.time() - t0, 1), "s", flush=True)
    R["seed"] = SEED; R["numpy"] = np.__version__; R["scipy"] = __import__("scipy").__version__
    json.dump(R, open("paradigm_results.json", "w"), indent=1, default=float)
