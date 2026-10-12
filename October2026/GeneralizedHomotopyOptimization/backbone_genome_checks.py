#!/usr/bin/env python3
"""
backbone_genome_checks.py -- checks for the section
"Learning backbone and parameter optimization" of
"Homotopy Learning for Category Level Training" (revision of 6 October 2026), D. P. Ghosh.
The labels H1-H10 match Appendix C, Table 8, and the checks cited in Section 9.

H1  LNS genome of the power-preserving group O(1,1): composition, closure, four components
H2  Diffusion Evolution's x0-estimate is a convex combination: dense genomes leave O(1,1)
H3  Gauge-aligned recombination: a law-preserving average can still erase content
H4  Non-abelian recombination: exponential-chart averages are not left-equivariant; Karcher means are
H5  Log-Kronecker genomes: log-domain averaging keeps Kronecker rank 1, raw averaging does not
H6  Vertex-potential genomes: coherence by construction; residual is chi-square with beta_1 dof
H7  Law selection by code length: the power law is selected only when the data obey it
H8  Lifted sheaf: zero fixed states on V, but the norm survives as a section of Sym^2 V
H9  Ultrametric tolerance: p-adic disk radii combine by max, real interval radii by sum
H10 A small Diffusion Evolution experiment: dense, LNS-edge and LNS-vertex genomes

Requires numpy, scipy, networkx.  Run:  python3 backbone_genome_checks.py
"""
import math
from fractions import Fraction

import numpy as np
import networkx as nx
from scipy import stats
from scipy.linalg import expm, logm
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


S = np.array([[0.0, 1.0], [1.0, 0.0]])          # effort/flow exchange
J = np.array([[0.0, 0.5], [0.5, 0.0]])          # power P(x) = x^T J x = e*f


def T(g):
    """Decode an LNS genome g = (sigma, s, eps): sign bit, log-magnitude, swap bit."""
    sig, s, eps = g
    return ((-1) ** sig) * np.diag([math.exp(s), math.exp(-s)]) @ np.linalg.matrix_power(S, eps)


def compose(g1, g2):
    (a1, s1, e1), (a2, s2, e2) = g1, g2
    return (a1 ^ a2, s1 + ((-1) ** e1) * s2, e1 ^ e2)


def law_defect(M):
    return float(np.linalg.norm(M.T @ J @ M - J))


# ---------------------------------------------------------------------------
def check_H1():
    rng = np.random.default_rng(1)
    worst_c, worst_p = 0.0, 0.0
    for _ in range(2000):
        g1 = (int(rng.integers(2)), float(rng.normal() * 2), int(rng.integers(2)))
        g2 = (int(rng.integers(2)), float(rng.normal() * 2), int(rng.integers(2)))
        worst_c = max(worst_c, np.abs(T(compose(g1, g2)) - T(g1) @ T(g2)).max())
        worst_p = max(worst_p, law_defect(T(g1)))
    report("H1: LNS composition law (sign XOR, s1 + (-1)^eps1 s2, swap XOR)", worst_c < 1e-9,
           f"max error {worst_c:.1e}")
    report("H1: every decoded transport preserves power", worst_p < 1e-9, f"max defect {worst_p:.1e}")
    # four components: det and sign of the (1,1) or (1,2) entry
    comps = {(round(np.linalg.det(T((a, 0.3, e)))), a, e) for a in (0, 1) for e in (0, 1)}
    neg_id_reachable_without_sign = any(np.allclose(T((0, s, e)), -np.eye(2))
                                        for s in np.linspace(-3, 3, 601) for e in (0, 1))
    report("H1: O(1,1) has four components; without the sign bit, -I is unreachable",
           len(comps) == 4 and not neg_id_reachable_without_sign, f"components {sorted(comps)}")
    # completeness: any M with M^T J M = J decodes to a genome
    ok = True
    for _ in range(500):
        g = (int(rng.integers(2)), float(rng.normal() * 2), int(rng.integers(2)))
        M = T(g)
        eps = 0 if abs(M[0, 1]) < 1e-12 else 1
        a = M[0, 0] if eps == 0 else M[0, 1]
        g_back = (0 if a > 0 else 1, math.log(abs(a)), eps)
        ok &= np.allclose(T(g_back), M)
    report("H1: decoding is invertible on O(1,1) (genome recovered from the matrix)", ok)


# ---------------------------------------------------------------------------
def de_weights(X, fit, alpha, temp=None):
    """Diffusion Evolution x0-estimate weights for every individual (Zhang et al., Eq. 8-9)."""
    f = np.asarray(fit, float)
    temp = temp if temp is not None else (f.std() + 1e-12)
    g = np.exp((f - f.max()) / temp)
    d2 = ((X[:, None, :] - math.sqrt(alpha) * X[None, :, :]) ** 2).sum(-1)
    logk = -d2 / (2 * max(1 - alpha, 1e-12))
    logk -= logk.max(axis=1, keepdims=True)
    W = g[None, :] * np.exp(logk)
    return W / W.sum(axis=1, keepdims=True)


def check_H2():
    rng = np.random.default_rng(2)
    N = 64
    genomes = [(0, float(rng.normal()), int(rng.integers(2))) for _ in range(N)]
    dense = np.stack([T(g).ravel() for g in genomes])
    fit = -np.array([abs(g[1] - 0.5) for g in genomes])
    Wt = de_weights(dense, fit, alpha=0.5)
    convex = np.all(Wt >= 0) and np.allclose(Wt.sum(1), 1.0)
    x0 = Wt @ dense
    defects = [law_defect(x.reshape(2, 2)) for x in x0]
    report("H2: the x0-estimate is a convex combination of the population", convex)
    report("H2: dense genomes -- x0-estimates leave O(1,1)", max(defects) > 1e-2,
           f"max law defect of x0 {max(defects):.3f}; population defect 0")
    # same population in LNS coordinates, within one swap branch
    branch = [g for g in genomes if g[2] == 0]
    sv = np.array([[g[1]] for g in branch])
    Wb = de_weights(sv, -np.abs(sv[:, 0] - 0.5), alpha=0.5)
    s0 = Wb @ sv
    d_lns = max(law_defect(T((0, float(s), 0))) for s in s0[:, 0])
    report("H2: LNS genomes -- every x0-estimate decodes inside O(1,1)", d_lns < 1e-12,
           f"max law defect {d_lns:.1e}")


# ---------------------------------------------------------------------------
def check_H3():
    a, b = (0, math.log(4.0), 0), (0, -math.log(4.0), 0)
    naive = (0, 0.5 * (a[1] + b[1]), 0)
    gain_naive = T(naive)[0, 0]
    # b is the same transport written with effort and flow exchanged in both domains: S T(b) S = T(a)
    same_transport = np.allclose(S @ T(b) @ S, T(a))
    # aligned recombination: choose the convention c in {I, S} minimizing the distance to a
    cands = [(np.eye(2), b), (S, (0, -b[1], 0))]
    best = min(cands, key=lambda cb: np.linalg.norm(T(cb[1]) - T(a)))
    aligned = (0, 0.5 * (a[1] + best[1][1]), 0)
    report("H3: the two candidates are one transport in exchanged conventions", same_transport)
    report("H3: naive log-average preserves power but erases the gain (4 -> 1)",
           law_defect(T(naive)) < 1e-12 and abs(gain_naive - 1.0) < 1e-12, f"decoded gain {gain_naive:.3f}")
    report("H3: aligned log-average preserves power and the gain", abs(T(aligned)[0, 0] - 4.0) < 1e-12,
           f"decoded gain {T(aligned)[0, 0]:.3f}")


# ---------------------------------------------------------------------------
def karcher(Rs, iters=100):
    mu = Rs[0]
    for _ in range(iters):
        v = np.mean([Rotation.from_matrix(mu.T @ R).as_rotvec() for R in Rs], axis=0)
        mu = mu @ Rotation.from_rotvec(v).as_matrix()
        if np.linalg.norm(v) < 1e-14:
            break
    return mu


def exp_chart_mean(Rs):
    return Rotation.from_rotvec(np.mean([Rotation.from_matrix(R).as_rotvec() for R in Rs], 0)).as_matrix()


def check_H4():
    rng = np.random.default_rng(4)
    worst_exp, worst_k, errs = 0.0, 0.0, []
    for _ in range(50):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        g = Rotation.from_rotvec(2.0 * axis).as_matrix()
        Rs = [g @ Rotation.from_rotvec(rng.normal(size=3) * 0.35).as_matrix() for _ in range(8)]
        h = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()
        e1 = Rotation.from_matrix((h @ exp_chart_mean(Rs)).T @ exp_chart_mean([h @ R for R in Rs])).magnitude()
        e2 = Rotation.from_matrix((h @ karcher(Rs)).T @ karcher([h @ R for R in Rs])).magnitude()
        worst_exp, worst_k = max(worst_exp, e1), max(worst_k, e2)
        errs.append(e1)
    report("H4: exponential-chart averaging is not left-equivariant (convention-dependent)",
           np.median(errs) > 1e-2, f"median equivariance error {np.median(errs):.3f} rad, max {worst_exp:.3f}")
    report("H4: Karcher (bi-invariant) mean is left-equivariant", worst_k < 1e-8,
           f"max equivariance error {worst_k:.1e} rad")


# ---------------------------------------------------------------------------
def vlp(Wm, p, q, r, s):
    """Van Loan-Pitsianis rearrangement: A (x) B becomes vec(A) vec(B)^T."""
    return Wm.reshape(p, r, q, s).transpose(0, 2, 1, 3).reshape(p * q, r * s)


def check_H5():
    rng = np.random.default_rng(5)
    p, q, r, s, N = 3, 4, 2, 5, 6
    As = [np.exp(rng.normal(size=(p, q))) for _ in range(N)]
    Bs = [np.exp(rng.normal(size=(r, s))) for _ in range(N)]
    w = rng.dirichlet(np.ones(N))
    raw = sum(wi * np.kron(A, B) for wi, A, B in zip(w, As, Bs))
    logavg = np.exp(sum(wi * np.log(np.kron(A, B)) for wi, A, B in zip(w, As, Bs)))
    sv_raw = np.linalg.svd(vlp(raw, p, q, r, s), compute_uv=False)
    sv_log = np.linalg.svd(vlp(logavg, p, q, r, s), compute_uv=False)
    rank = lambda sv: int((sv > 1e-9 * sv[0]).sum())
    report("H5: raw averaging of positive Kronecker products raises Kronecker rank", rank(sv_raw) > 1,
           f"Kronecker rank {rank(sv_raw)}")
    report("H5: log-domain averaging keeps Kronecker rank 1 (outer sums are linear)", rank(sv_log) == 1,
           f"Kronecker rank {rank(sv_log)}")


# ---------------------------------------------------------------------------
def check_H6():
    rng = np.random.default_rng(6)
    G = nx.gnm_random_graph(12, 30, seed=3)
    assert nx.is_connected(G)
    E = list(G.edges())
    V, b1 = G.number_of_nodes(), len(E) - G.number_of_nodes() + 1
    B = np.zeros((len(E), V))
    for k, (u, v) in enumerate(E):
        B[k, u], B[k, v] = -1.0, 1.0                      # y_e = phi_v - phi_u
    Bg = B[:, 1:]                                         # root potential fixed (gauge)
    sigma, trials = 0.1, 4000
    stat = []
    for _ in range(trials):
        phi = np.concatenate([[0.0], rng.normal(size=V - 1)])
        y = B @ phi + sigma * rng.normal(size=len(E))
        sol, *_ = np.linalg.lstsq(Bg, y, rcond=None)
        stat.append(np.sum((y - Bg @ sol) ** 2) / sigma ** 2)
    stat = np.array(stat)
    ks = stats.kstest(stat, "chi2", args=(b1,)).pvalue
    report("H6: coherent genome needs (|V|-1) potentials instead of |E| edge genes",
           (V - 1) + b1 == len(E), f"|V|-1 = {V-1}, |E| = {len(E)}, saving beta_1 = {b1}")
    report("H6: abelian residual / sigma^2 is chi-square with beta_1 dof (KS test)",
           abs(stat.mean() - b1) < 0.5 and ks > 0.01, f"mean {stat.mean():.2f} vs {b1}, KS p = {ks:.2f}")
    # planted holonomy on one non-tree edge is detected
    phi = np.concatenate([[0.0], rng.normal(size=V - 1)])
    y = B @ phi + sigma * rng.normal(size=len(E))
    T_edges = {frozenset(e) for e in nx.minimum_spanning_tree(G).edges()}
    k = next(i for i, e in enumerate(E) if frozenset(e) not in T_edges)
    y[k] += 1.0
    sol, *_ = np.linalg.lstsq(Bg, y, rcond=None)
    rss = np.sum((y - Bg @ sol) ** 2) / sigma ** 2
    report("H6: a planted holonomy of 1.0 is far outside the chi-square acceptance region",
           rss > stats.chi2.ppf(0.999, b1), f"statistic {rss:.0f} vs 99.9% point {stats.chi2.ppf(0.999, b1):.0f}")
    # SO(3): small-noise synchronization residual ~ 3 beta_1 sigma^2
    def solve_so3(Redges, g0):
        def res(d):
            gs = [g0[0]] + [g0[i] @ Rotation.from_rotvec(d[3 * (i - 1):3 * i]).as_matrix() for i in range(1, V)]
            return np.concatenate([Rotation.from_matrix(gs[v].T @ Re @ gs[u]).as_rotvec()
                                   for (u, v), Re in zip(E, Redges)])
        sol = least_squares(res, np.zeros(3 * (V - 1)), method="lm", xtol=1e-14, ftol=1e-14)
        return np.sum(sol.fun ** 2)
    vals = []
    for _ in range(60):
        g = [np.eye(3)] + [Rotation.random(random_state=int(rng.integers(1e9))).as_matrix() for _ in range(V - 1)]
        Redges = [g[v] @ Rotation.from_rotvec(sigma * rng.normal(size=3)).as_matrix() @ g[u].T for (u, v) in E]
        vals.append(solve_so3(Redges, g) / sigma ** 2)
    m = float(np.mean(vals))
    report("H6: SO(3) synchronization residual / sigma^2 is close to 3 beta_1", abs(m - 3 * b1) < 0.1 * 3 * b1,
           f"mean {m:.1f} vs {3 * b1}")


# ---------------------------------------------------------------------------
def fit_models(X, Y):
    n = X.shape[0]
    out = {}
    M, *_ = np.linalg.lstsq(X, Y, rcond=None)              # GL(2): 4 parameters
    out["GL2"] = (np.sum((Y - X @ M) ** 2), 4)
    best = np.inf
    for sig in (0, 1):
        for eps in (0, 1):
            f = lambda s: np.sum((Y - X @ T((sig, s, eps)).T) ** 2)
            r = minimize_scalar(f, bounds=(-5, 5), method="bounded")
            best = min(best, r.fun)
    out["O11"] = (best, 1)
    bestl = np.inf
    for sig in (0, 1):
        for eps in (0, 1):
            def f2(z):
                s, le = z
                Mz = np.diag([math.exp(le), 1.0]) @ T((sig, s, eps))
                return Y - X @ Mz.T
            r = least_squares(lambda z: f2(z).ravel(), [0.0, 0.0])
            bestl = min(bestl, np.sum(r.fun ** 2))
    out["lossy"] = (bestl, 2)
    bits = {"GL2": 0, "O11": 2, "lossy": 2}
    code = {k: n * 2 * 0.5 * math.log(rss / (2 * n)) + 0.5 * kk * math.log(2 * n) + bits[k] * math.log(2)
            for k, (rss, kk) in out.items()}
    return min(code, key=code.get)


def check_H7():
    rng = np.random.default_rng(7)
    picks = {"lossless": [], "lossy": []}
    for _ in range(100):
        g = (int(rng.integers(2)), float(rng.normal()), int(rng.integers(2)))
        X = rng.normal(size=(40, 2))
        for kind in picks:
            Mtrue = T(g) if kind == "lossless" else np.diag([0.8, 1.0]) @ T(g)
            Y = X @ Mtrue.T + 0.05 * rng.normal(size=X.shape)
            picks[kind].append(fit_models(X, Y))
    f_ll = picks["lossless"].count("O11") / 100
    f_ly = picks["lossy"].count("O11") / 100
    report("H7: power-preserving genome selected when the data conserve power", f_ll >= 0.9,
           f"selected in {f_ll:.0%} of runs")
    report("H7: power-preserving genome rejected when the transport loses 20% of effort",
           f_ly <= 0.05, f"selected in {f_ly:.0%}; lossy genome selected in "
                         f"{picks['lossy'].count('lossy') / 100:.0%}")


# ---------------------------------------------------------------------------
def h0h1(dims, edges, maps):
    off = np.concatenate([[0], np.cumsum(dims)])
    rows = sum(dims[v] for (_, v) in edges)
    D = np.zeros((rows, off[-1]))
    r = 0
    for (u, v) in edges:
        D[r:r + dims[v], off[u]:off[u] + dims[u]] += maps[(u, v)]
        D[r:r + dims[v], off[v]:off[v] + dims[v]] -= np.eye(dims[v])
        r += dims[v]
    rank = np.linalg.matrix_rank(D, tol=1e-9)
    return int(off[-1] - rank), int(rows - rank)


def sym2(R):
    """Action Q -> R Q R^T on symmetric 2x2 matrices in the basis E11, E22, (E12+E21)."""
    basis = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
    coords = lambda Q: np.array([Q[0, 0], Q[1, 1], Q[0, 1]])
    return np.stack([coords(R @ Bq @ R.T) for Bq in basis], axis=1)


def check_H8():
    th = 1.0
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    m1 = {(0, 1): np.eye(2), (1, 2): np.eye(2), (2, 3): np.eye(2), (3, 0): R}
    h0_v, _ = h0h1([2] * 4, edges, m1)
    m2 = {k: sym2(M) for k, M in m1.items()}
    h0_s, _ = h0h1([3] * 4, edges, m2)
    # the surviving section is the identity matrix, i.e. the squared norm
    fixed = np.linalg.svd(sym2(R) - np.eye(3))[2][-1]
    is_norm = np.allclose(fixed / fixed[0], [1, 1, 0])
    report("H8: rotation holonomy -- no compatible state on V (H0 = 0)", h0_v == 0, f"H0(V) = {h0_v}")
    report("H8: lifted to Sym^2 V the squared norm is a global section (H0 = 1)",
           h0_s == 1 and is_norm, f"H0(Sym2 V) = {h0_s}")


# ---------------------------------------------------------------------------
def vp(x, p):
    if x == 0:
        return math.inf
    x = Fraction(x)
    v, a, b = 0, x.numerator, x.denominator
    while a % p == 0:
        a //= p; v += 1
    while b % p == 0:
        b //= p; v -= 1
    return v


def check_H9():
    p, k, N = 3, 4, 200
    rng = np.random.default_rng(9)
    centers = [int(c) for c in rng.integers(0, 10**6, size=N)]
    rad = p ** (-k)
    # p-adic disks D(c, p^-k) are cosets c + p^k Z_p; any element of the sum lies within p^-k of the sum of centers
    worst = 0.0
    for _ in range(200):
        pts = [c + (p ** k) * int(rng.integers(-10**6, 10**6)) for c in centers]
        diff = sum(pts) - sum(centers)
        worst = max(worst, p ** (-vp(diff, p)) if diff else 0.0)
    report("H9: sum of 200 p-adic disks of radius p^-k has radius p^-k (max, not sum)", worst <= rad + 1e-15,
           f"observed radius {worst:.4g} = p^-{k}; real intervals would give {N} x radius")
    u = 7
    a_pt, b_pt = u, u + p ** k                      # both lie in the disk D(u, p^-k)
    r_obs = p ** (-vp(a_pt - b_pt, p))
    report("H9: dependency problem persists -- D(u,r) - D(u,r) has radius r, not 0",
           abs(r_obs - rad) < 1e-15, f"|a - b|_p = {r_obs:.4g} for a, b in the same disk")


# ---------------------------------------------------------------------------
def de_run(dim, fitness, N=64, T_steps=30, seed=0, eta=1.0):
    rng = np.random.default_rng(seed)
    alpha = lambda t: 0.5 * math.cos(math.pi * t / T_steps) + 0.5
    X = rng.normal(size=(N, dim))
    evals = 0
    for t in range(T_steps, 0, -1):
        f = np.array([fitness(x) for x in X]); evals += N
        a_t, a_p = alpha(t), alpha(t - 1)
        W = de_weights(X, f, a_t)
        x0 = W @ X
        eps_hat = (X - math.sqrt(a_t) * x0) / math.sqrt(1 - a_t)
        sig = eta * math.sqrt((1 - a_p) / (1 - a_t)) * math.sqrt(max(1 - a_t / a_p, 0.0)) if a_p > 0 else 0.0
        X = math.sqrt(a_p) * x0 + math.sqrt(max(1 - a_p - sig ** 2, 0.0)) * eps_hat + sig * rng.normal(size=X.shape)
    f = np.array([fitness(x) for x in X]); evals += N
    return X[int(np.argmax(f))], evals


def check_H10():
    noise = 0.05
    floor = noise ** 2
    inv = lambda g: (g[0], -((-1) ** g[2]) * g[1], g[2])
    E = [(0, 1), (1, 2), (2, 0)]
    gv = {0: (0, 0.0, 0), 1: (0, 0.7, 1), 2: (1, -0.4, 0)}          # vertex potentials (root = identity)

    def run(holonomy, seed):
        rng = np.random.default_rng(seed)
        Ftrue = {e: T(compose(gv[e[1]], inv(gv[e[0]]))) for e in E}
        if holonomy:
            Ftrue[(2, 0)] = T((0, 0.5, 0)) @ Ftrue[(2, 0)]           # loop transport becomes diag(e^.5, e^-.5)

        def data(n):
            D = {}
            for e in E:
                X = rng.normal(size=(n, 2))
                D[e] = (X, X @ Ftrue[e].T + noise * rng.normal(size=X.shape))
            return D
        train, test = data(20), data(400)
        mse = lambda F, D: float(np.mean([np.mean((D[e][1] - D[e][0] @ F[e].T) ** 2) for e in E]))
        # 1. discrete proposals: sign and swap bits per edge by local one-dimensional fits
        def local(e, b):
            X, Y = train[e]
            return minimize_scalar(lambda s: np.sum((Y - X @ T((b[0], s, b[1])).T) ** 2),
                                   bounds=(-5, 5), method="bounded")
        bits_e = {e: min(((a, c) for a in (0, 1) for c in (0, 1)), key=lambda b: local(e, b).fun) for e in E}
        # vertex bits by spanning-tree gauge fixing: compose edge bits along 0 -> 1 -> 2
        b1 = (bits_e[(0, 1)][0], bits_e[(0, 1)][1])
        b2 = (b1[0] ^ bits_e[(1, 2)][0], b1[1] ^ bits_e[(1, 2)][1])

        def dec_vertex(z):
            g = {0: (0, 0.0, 0), 1: (b1[0], z[0], b1[1]), 2: (b2[0], z[1], b2[1])}
            return {e: T(compose(g[e[1]], inv(g[e[0]]))) for e in E}
        genomes = {
            "dense edges": (12, lambda z: {e: z[4 * i:4 * i + 4].reshape(2, 2) for i, e in enumerate(E)}),
            "LNS edges": (3, lambda z: {e: T((bits_e[e][0], z[i], bits_e[e][1])) for i, e in enumerate(E)}),
            "LNS vertices": (2, dec_vertex),
        }
        out = {}
        for name, (dim, decode) in genomes.items():
            fit = lambda z: -mse(decode(z), train)
            z, ev = de_run(dim, fit, seed=seed + 1)
            de_only = mse(decode(z), test)
            # 2. correction: bounded local least squares from the DE result
            resid = lambda zz: np.concatenate([(train[e][1] - train[e][0] @ decode(zz)[e].T).ravel() for e in E])
            sol = least_squares(resid, z, max_nfev=200)
            F = decode(sol.x)
            out[name] = dict(de=de_only, test=mse(F, test), train=mse(F, train),
                             law=max(law_defect(M) for M in F.values()),
                             loop=float(np.linalg.norm(F[(2, 0)] @ F[(1, 2)] @ F[(0, 1)] - np.eye(2))),
                             evals=ev + sol.nfev, genes=dim)
        return out

    coh, hol = run(False, 11), run(True, 12)
    print("\n    H10 three-domain power-preserving diagram, 20 pairs per edge, noise 0.05 (MSE floor 0.0025)")
    print(f"    {'case':10s} {'genome':14s} {'reals':>5s} {'DE-only':>9s} {'+correction':>11s} {'law defect':>11s}"
          f" {'loop defect':>11s} {'evals':>6s}")
    for case, res in (("coherent", coh), ("holonomy", hol)):
        for name, r in res.items():
            print(f"    {case:10s} {name:14s} {r['genes']:5d} {r['de']:9.4f} {r['test']:11.5f} {r['law']:11.1e}"
                  f" {r['loop']:11.1e} {r['evals']:6d}")
    print()
    d, le, lv = coh["dense edges"], coh["LNS edges"], coh["LNS vertices"]
    report("H10: Diffusion Evolution alone stops above the noise floor (correction needed)",
           min(d["de"], le["de"], lv["de"]) > 2 * floor, f"best DE-only test MSE {min(d['de'], le['de'], lv['de']):.4f}")
    report("H10: dense genome after correction still violates the law and the loop",
           d["law"] > 1e-3 and d["loop"] > 1e-3, f"law {d['law']:.1e}, loop {d['loop']:.1e}")
    report("H10: LNS edge genome keeps the law exactly; its loop stays open", le["law"] < 1e-12 and le["loop"] > 1e-4)
    report("H10: LNS vertex genome keeps law and loop exactly and reaches the noise floor",
           lv["law"] < 1e-12 and lv["loop"] < 1e-12 and lv["test"] < 1.2 * floor, f"test MSE {lv['test']:.5f}")
    hv, he = hol["LNS vertices"], hol["LNS edges"]
    report("H10: with a true holonomy the coherent genome cannot fit; its training residual flags it",
           hv["train"] > 10 * floor and he["train"] < 1.5 * floor,
           f"train MSE: vertex genome {hv['train']:.4f}, edge genome {he['train']:.5f}")


if __name__ == "__main__":
    for fn in (check_H1, check_H2, check_H3, check_H4, check_H5, check_H6, check_H7, check_H8, check_H9,
               check_H10):
        fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
