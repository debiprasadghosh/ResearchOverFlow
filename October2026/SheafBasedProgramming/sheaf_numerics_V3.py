"""Seeded numerical verification harness for 'Sheaf Based Programming Paradigm'.
Every experiment reports residuals or agreement counts against the paper's statements.
Run:  python3 sheaf_numerics.py   (numpy, scipy, matplotlib, networkx)
"""
import itertools, json, math
from fractions import Fraction
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261021
rng = np.random.default_rng(SEED)
EPS = np.finfo(float).eps
R = {}

# ---------- helpers -------------------------------------------------------
def incidence(n, edges):
    D = np.zeros((len(edges), n))
    for e, (u, v) in enumerate(edges):
        D[e, u] -= 1.0; D[e, v] += 1.0          # (Ds)_uv = s_v - s_u
    return D

def triangles(G):
    return [t for t in nx.enumerate_all_cliques(G) if len(t) == 3]

def d1_matrix(edges, tris):
    idx = {e: i for i, e in enumerate(edges)}
    E = np.zeros((len(tris), len(edges)))
    for t, (a, b, c) in enumerate(tris):       # (dg)_abc = g_bc - g_ac + g_ab
        E[t, idx[(b, c)]] += 1; E[t, idx[(a, c)]] -= 1; E[t, idx[(a, b)]] += 1
    return E

def rank_R(A):
    if A.size == 0: return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > max(A.shape) * EPS * s[0]).sum()) if s.size and s[0] > 0 else 0

def rank_gap(A, tol):
    """Rank with relative tolerance tol, plus certificate (smallest kept, largest dropped)/s_max."""
    if A.size == 0: return 0, 1.0, 0.0
    s = np.linalg.svd(A, compute_uv=False)
    if s[0] == 0: return 0, 1.0, 0.0
    r = int((s > tol * s[0]).sum())
    kept = s[r - 1] / s[0] if r else 1.0
    dropped = s[r] / s[0] if r < s.size else 0.0
    return r, float(kept), float(dropped)

def rank_F2(A):
    M = (np.abs(np.rint(A)).astype(np.int64) % 2).astype(np.uint8)
    r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r

def er_graph(n, p):
    G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 31)))
    edges = sorted(tuple(sorted(e)) for e in G.edges())
    return G, edges

def null_basis(A, tol=None):
    if A.shape[0] == 0: return np.eye(A.shape[1])
    u, s, vt = np.linalg.svd(A)
    tol = tol or max(A.shape) * EPS * (s[0] if s.size else 1)
    r = int((s > tol).sum())
    return vt[r:].T

# ---------- E1: dimensions, d∘d = 0, Euler characteristic (Lemma 3.1, Thm 4.1)
def e1():
    rows = []
    for n, p in [(10, 0.3), (50, 0.08), (200, 0.02)]:
        trials, bad_R, bad_F2, max_dd, bad_euler = 100, 0, 0, 0, 0
        msum = csum = h1sum = 0
        for _ in range(trials):
            G, edges = er_graph(n, p)
            m, c = len(edges), nx.number_connected_components(G)
            D = incidence(n, edges)
            rR, rF = rank_R(D), rank_F2(D)
            bad_R += (n - rR != c) or (m - rR != m - n + c)
            bad_F2 += (n - rF != c) or (m - rF != m - n + c)
            T = triangles(G); E = d1_matrix(edges, T)
            if len(T):
                max_dd = max(max_dd, int(np.abs(E @ D).max()))
                rE = rank_R(E)
                h = [n - rR, m - rR - rE, len(T) - rE]
                bad_euler += (n - m + len(T)) != (h[0] - h[1] + h[2])
            msum += m; csum += c; h1sum += m - n + c
        rows.append(dict(n=n, p=p, trials=trials, mean_m=msum / trials, mean_c=csum / trials,
                         mean_H1=h1sum / trials, mismatches_R=bad_R, mismatches_F2=bad_F2,
                         max_abs_dd=max_dd, euler_mismatches=bad_euler))
    R["E1"] = rows

# ---------- E2: exact rational triangle example (Section 4) ---------------
def e2():
    g = [Fraction(2), Fraction(3), Fraction(4)]
    z = [1, 1, -1]
    kappa = sum(zi * gi for zi, gi in zip(z, g))
    c = [-kappa * zi / 3 for zi in z]
    gp = [a + b for a, b in zip(g, c)]
    timeline = [Fraction(0), gp[0], gp[0] + gp[1]]
    R["E2"] = dict(kappa=str(kappa), correction=[str(x) for x in c], repaired=[str(x) for x in gp],
                   timeline=[str(x) for x in timeline], cost=str(sum(x * x for x in c)),
                   closure_after=str(gp[0] + gp[1] - gp[2]),
                   timeline_consistent=(timeline[2] - timeline[0] == gp[2]))
    # Proposition 4.2: add a face
    D = incidence(3, [(0, 1), (1, 2), (0, 2)]); E = d1_matrix([(0, 1), (1, 2), (0, 2)], [(0, 1, 2)])
    R["E2"]["face_dd"] = float(np.abs(E @ D).max())
    R["E2"]["face_H1"] = 3 - rank_R(D) - rank_R(E)
    R["E2"]["face_Eg"] = float((E @ np.array([2., 3, 4]))[0])

# ---------- E3: weighted repair (Thm 7.1, Cor 7.2, Prop 7.3) --------------
def e3():
    rows = []
    for k in [3, 5, 10, 50, 200]:
        worst = dict(idem_pinv_default=0, idem_pinv_1e10=0, idem_gauge=0, selfadj=0,
                     closed_form=0, pyth=0, cost=0, exact_after=0, null_eig_max=0, above_default=0)
        edges = [tuple(sorted((i, (i + 1) % k))) for i in range(k)]
        D = incidence(k, edges)
        z = np.ones(k); z[-1] = -1.0                 # edge (0,k-1) is traversed backwards
        assert np.allclose(z @ D, 0)
        Dg = D[:, 1:]                                # gauge: s_0 = 0
        for _ in range(200):
            w = rng.uniform(0.2, 5.0, k); W = np.diag(w); g = rng.normal(0, 1, k)
            A = D.T @ W @ D
            sv = np.linalg.svd(A, full_matrices=False)[1]   # same LAPACK path as np.linalg.pinv (with vectors)
            worst["null_eig_max"] = max(worst["null_eig_max"], sv[-1] / sv[0])
            worst["above_default"] += int(sv[-1] / sv[0] > 1e-15)
            P_def = D @ np.linalg.pinv(A) @ D.T @ W
            P_tol = D @ np.linalg.pinv(A, rtol=1e-10, hermitian=True) @ D.T @ W
            P = Dg @ np.linalg.solve(Dg.T @ W @ Dg, Dg.T @ W)
            cstar = P @ g - g
            kap = z @ g
            cf = -kap * (z / w) / (z @ (z / w))
            r = g - P @ g; u = rng.normal(0, 1, k); Du = D @ u
            nW = lambda v: v @ W @ v
            worst["idem_pinv_default"] = max(worst["idem_pinv_default"], np.abs(P_def @ P_def - P_def).max())
            worst["idem_pinv_1e10"] = max(worst["idem_pinv_1e10"], np.abs(P_tol @ P_tol - P_tol).max())
            worst["idem_gauge"] = max(worst["idem_gauge"], np.abs(P @ P - P).max())
            worst["selfadj"] = max(worst["selfadj"], np.abs(P.T @ W - W @ P).max())
            worst["closed_form"] = max(worst["closed_form"], np.abs(cstar - cf).max())
            worst["pyth"] = max(worst["pyth"], abs(nW(r + Du) - nW(r) - nW(Du)) / max(1, nW(r + Du)))
            worst["cost"] = max(worst["cost"], abs(nW(cstar) - kap ** 2 / (z @ (z / w))) / max(1, nW(cstar)))
            worst["exact_after"] = max(worst["exact_after"], abs(z @ (g + cstar)))
        rows.append(dict(k=k, trials=200, **{k_: float(v) for k_, v in worst.items()}))
    R["E3"] = rows

# ---------- E4: Hodge decomposition on clique complexes (Thm 8.1) ---------
def e4():
    rows = []
    for n, p in [(12, 0.35), (30, 0.15), (60, 0.08)]:
        worst = dict(recon=0, orth_DE=0, orth_DH=0, orth_EH=0); dim_bad = 0; h1s = []
        for _ in range(50):
            G, edges = er_graph(n, p)
            if not edges: continue
            D = incidence(n, edges); T = triangles(G)
            E = d1_matrix(edges, T) if T else np.zeros((0, len(edges)))
            L1 = D @ D.T + E.T @ E
            Hb = null_basis(L1, tol=1e-9)
            dim_h = Hb.shape[1]; dim_H1 = len(edges) - rank_R(D) - rank_R(E)
            dim_bad += dim_h != dim_H1; h1s.append(dim_H1)
            v = rng.normal(0, 1, len(edges))
            ex = D @ np.linalg.lstsq(D, v, rcond=None)[0]
            co = E.T @ np.linalg.lstsq(E.T, v, rcond=None)[0] if E.size else np.zeros_like(v)
            ha = Hb @ (Hb.T @ v)
            nv = np.linalg.norm(v)
            worst["recon"] = max(worst["recon"], np.linalg.norm(v - ex - co - ha) / nv)
            worst["orth_DE"] = max(worst["orth_DE"], abs(ex @ co) / nv ** 2)
            worst["orth_DH"] = max(worst["orth_DH"], abs(ex @ ha) / nv ** 2)
            worst["orth_EH"] = max(worst["orth_EH"], abs(co @ ha) / nv ** 2)
        rows.append(dict(n=n, p=p, trials=50, mean_H1=float(np.mean(h1s)), dim_mismatch=dim_bad,
                         **{k: float(v) for k, v in worst.items()}))
    R["E4"] = rows

# ---------- E5: diffusion rates (Thm 8.2) ---------------------------------
def e5():
    n = 40
    edges = [(i, i + 1) for i in range(n - 1)]               # path graph
    D = incidence(n, edges); L0 = D.T @ D
    lam = np.sort(np.linalg.eigvalsh(L0)); lam_plus = lam[1]; lam_max = lam[-1]
    x0 = rng.normal(0, 1, n); pi = np.full(n, x0.mean())
    w, V = np.linalg.eigh(L0)
    ts = np.linspace(0, 400, 81)
    err = [np.linalg.norm(V @ (np.exp(-w * t) * (V.T @ x0)) - pi) for t in ts]
    bound = [math.exp(-lam_plus * t) * np.linalg.norm(x0 - pi) for t in ts]
    slope = -np.polyfit(ts[40:], np.log(err[40:]), 1)[0]
    viol = int(sum(e > b * (1 + 1e-10) for e, b in zip(err, bound)))
    # discrete scheme
    disc = []
    for frac in [0.25, 0.5, 0.9, 0.99, 1.01]:
        eta = frac * 2 / lam_max
        rho_pred = max(abs(1 - eta * l) for l in lam[1:])
        x = x0.copy(); e0 = np.linalg.norm(x - pi); K = 2000
        for _ in range(K): x = x - eta * (L0 @ x)
        eK = np.linalg.norm(x - pi)
        rho_meas = (eK / e0) ** (1 / K) if eK > 0 and np.isfinite(eK) else float("nan")
        disc.append(dict(eta_over_2_lmax=frac, rho_pred=rho_pred, rho_meas=float(rho_meas),
                         converged=bool(np.isfinite(eK) and eK < e0)))
    R["E5"] = dict(n=n, lam_plus=lam_plus, lam_plus_exact=2 - 2 * math.cos(math.pi / n),
                   lam_max=lam_max, fitted_rate=slope, bound_violations=viol, discrete=disc)
    plt.figure(figsize=(6.2, 3.4))
    plt.semilogy(ts, err, lw=2, label="‖x(t) − Π₀x(0)‖  (measured)")
    plt.semilogy(ts, bound, "--", lw=1.5, label="exp(−λ₊t)‖x(0) − Π₀x(0)‖  (Thm 8.2)")
    plt.xlabel("t"); plt.ylabel("distance to coherent state"); plt.legend(frameon=False, fontsize=8)
    plt.title(f"Path graph n={n}: fitted rate {slope:.5f} vs λ₊ = {lam_plus:.5f}", fontsize=9)
    plt.tight_layout(); plt.savefig("fig_diffusion.png", dpi=200); plt.close()

# ---------- E6: Schur complement boundary reduction (Prop 9.2) -----------
def e6():
    rows = []
    for n, nb in [(50, 5), (200, 10), (800, 20)]:
        G = nx.random_regular_graph(4, n, seed=int(rng.integers(1 << 31)))
        edges = sorted(tuple(sorted(e)) for e in G.edges())
        D = incidence(n, edges); L = D.T @ D + 0.01 * np.eye(n)
        b = rng.normal(0, 1, n)
        Bi = np.arange(nb); Ii = np.arange(nb, n)
        LII, LIB, LBI, LBB = L[np.ix_(Ii, Ii)], L[np.ix_(Ii, Bi)], L[np.ix_(Bi, Ii)], L[np.ix_(Bi, Bi)]
        S = LBB - LBI @ np.linalg.solve(LII, LIB)
        bB = b[Bi] - LBI @ np.linalg.solve(LII, b[Ii])
        xB = np.linalg.solve(S, bB)
        xI = np.linalg.solve(LII, b[Ii] - LIB @ xB)
        x = np.linalg.solve(L, b)
        err = max(np.abs(x[Bi] - xB).max(), np.abs(x[Ii] - xI).max()) / np.abs(x).max()
        rows.append(dict(n=n, boundary=nb, rel_err=float(err), cond=float(np.linalg.cond(L))))
    R["E6"] = rows

# ---------- E7: relative extension criterion (Thm 9.1) -------------------
def e7():
    agree = total = blocked = 0
    for _ in range(500):
        n = int(rng.integers(5, 12))
        G, edges = er_graph(n, 0.45)
        if not nx.is_connected(G) or len(edges) < n: continue
        D = incidence(n, edges)
        A = sorted(rng.choice(n, size=int(rng.integers(2, n - 1)), replace=False).tolist())
        a = rng.normal(0, 1, len(A))
        # coherent on A requires a constant on each component of A's induced subgraph (scalar constant sheaf)
        GA = G.subgraph(A)
        for comp in nx.connected_components(GA):
            val = rng.normal()
            for v in comp: a[A.index(v)] = val
        free = [v for v in range(n) if v not in A]
        # criterion: relative class zero  <=>  exists u with D_free u = -D_A a  (edges touching free)
        rhs = -D[:, A] @ a
        u, *_ = np.linalg.lstsq(D[:, free], rhs, rcond=None)
        ext_exists = np.linalg.norm(D[:, free] @ u - rhs) < 1e-9
        # independent check: global sections are constant per component (connected graph -> constant)
        independent = np.allclose(a, a[0])
        agree += ext_exists == independent; total += 1; blocked += not ext_exists
    R["E7"] = dict(trials=total, agreements=agree, blocked_extensions=blocked)

# ---------- E8: Mayer–Vietoris interface identity (Prop 6.2) -------------
def e8():
    agree = agree_default = total = 0; nonzero = 0; min_kept = 1.0; max_dropped = 0.0
    for _ in range(300):
        n = int(rng.integers(6, 16))
        G, edges = er_graph(n, 0.35)
        if len(edges) < 3: continue
        # random split of edges into A and B with overlap; vertices of each = endpoints
        lab = rng.integers(0, 3, len(edges))          # 0:A only 1:B only 2:both
        EA = [e for e, l in zip(edges, lab) if l in (0, 2)]
        EB = [e for e, l in zip(edges, lab) if l in (1, 2)]
        VA = sorted({v for e in EA for v in e} | set(rng.choice(n, 2).tolist()))
        VB = sorted({v for e in EB for v in e} | (set(range(n)) - set(VA)))
        VC = sorted(set(VA) & set(VB)); EC = [e for e in edges if e in EA and e in EB]
        if not VC: continue
        def sub(V, Es):
            ix = {v: i for i, v in enumerate(V)}
            return incidence(len(V), [(ix[u], ix[v]) for u, v in Es]) if Es else np.zeros((0, len(V)))
        DA, DB, DC, DX = sub(VA, EA), sub(VB, EB), sub(VC, EC), incidence(n, edges)
        KA, KB, KC = null_basis(DA), null_basis(DB), null_basis(DC)
        RA = np.zeros((len(VC), len(VA))); RB = np.zeros((len(VC), len(VB)))
        for i, v in enumerate(VC): RA[i, VA.index(v)] = 1; RB[i, VB.index(v)] = 1
        # beta in coordinates of KC (KC orthonormal): coords = KC^T (...)
        beta = np.hstack([-KC.T @ RA @ KA, KC.T @ RB @ KB])
        coker_default = KC.shape[1] - rank_R(beta)
        rb, kept, dropped = rank_gap(beta, 1e-9)
        coker = KC.shape[1] - rb
        min_kept = min(min_kept, kept); max_dropped = max(max_dropped, dropped)
        # kernel of H1(X) -> H1(A) ⊕ H1(B)
        eA = [edges.index(e) for e in EA]; eB = [edges.index(e) for e in EB]
        SA = np.zeros((len(EA), len(edges))); SA[np.arange(len(EA)), eA] = 1
        SB = np.zeros((len(EB), len(edges))); SB[np.arange(len(EB)), eB] = 1
        M = np.block([[SA, -DA, np.zeros((len(EA), len(VB)))],
                      [SB, np.zeros((len(EB), len(VA))), -DB]])
        dimS = M.shape[1] - rank_R(M) - (KA.shape[1] + KB.shape[1])
        kern = dimS - rank_R(DX)
        H = nx.Graph(); H.add_nodes_from(VC); H.add_edges_from(EC); hC = nx.number_connected_components(H)
        HA = nx.Graph(); HA.add_nodes_from(VA); HA.add_edges_from(EA)
        HB = nx.Graph(); HB.add_nodes_from(VB); HB.add_edges_from(EB)
        exact = hC - (nx.number_connected_components(HA) + nx.number_connected_components(HB) - nx.number_connected_components(G))
        agree += (coker == kern == exact); agree_default += (coker_default == kern); total += 1; nonzero += coker > 0
    R["E8"] = dict(trials=total, agreements=agree, agreements_default_tol=agree_default,
                   nonzero_interface_classes=nonzero, min_kept_sv=min_kept, max_dropped_sv=max_dropped)

# ---------- E9: min-edit F2 repair = max-cut (Prop 10.1); sparse vs quadratic
def e9():
    agree = total = 0
    for _ in range(200):
        n = int(rng.integers(4, 10))
        G, edges = er_graph(n, 0.5)
        if not edges: continue
        best_edit = min(sum(1 for (u, v) in edges if not (s[u] ^ s[v])) for s in itertools.product([0, 1], repeat=n))
        maxcut = max(sum(1 for (u, v) in edges if s[u] != s[v]) for s in itertools.product([0, 1], repeat=n))
        agree += best_edit == len(edges) - maxcut; total += 1
    # sparse vs quadratic on a k-cycle with one corrupted edge
    k = 8; edges = [tuple(sorted((i, (i + 1) % k))) for i in range(k)]
    D = incidence(k, edges); s = rng.normal(0, 1, k); g = D @ s; g[3] += 2.0
    P = D @ np.linalg.pinv(D) ; cq = P @ g - g
    z = np.ones(k); z[-1] = -1; kap = z @ g
    cs = np.zeros(k); cs[3] = -kap * z[3]
    R["E9"] = dict(maxcut_trials=total, maxcut_agreements=agree, cycle_k=k, kappa=float(kap),
                   quadratic_support=int((np.abs(cq) > 1e-12).sum()), quadratic_l2=float(np.linalg.norm(cq)),
                   quadratic_max=float(np.abs(cq).max()), sparse_support=1, sparse_l2=float(np.linalg.norm(cs)),
                   sparse_exact=float(abs(z @ (g + cs))))

# ---------- E10: locality of responses (Conjecture 12.4, evidence only) ---
def e10():
    out = []; plt.figure(figsize=(6.2, 3.4))
    side = 31; G = nx.grid_2d_graph(side, side); nodes = list(G.nodes()); ix = {v: i for i, v in enumerate(nodes)}
    edges = [tuple(sorted((ix[u], ix[v]))) for u, v in G.edges()]
    D = incidence(len(nodes), edges); L0 = D.T @ D
    src = ix[(side // 2, side // 2)]
    dist = np.array([abs(v[0] - side // 2) + abs(v[1] - side // 2) for v in nodes])
    for mu in [1.0, 0.1, 0.01, 0.001]:
        b = np.zeros(len(nodes)); b[src] = 1.0
        x = np.linalg.solve(L0 + mu * np.eye(len(nodes)), b)
        prof = np.array([np.abs(x[dist == r]).max() for r in range(0, side)])
        rr = np.arange(2, 14); slope = -np.polyfit(rr, np.log(prof[rr]), 1)[0]
        out.append(dict(mu=mu, fitted_decay=slope, decay_length=1 / slope))
        plt.semilogy(np.arange(side), prof, label=f"μ = {mu}")
    plt.xlabel("graph distance from perturbation"); plt.ylabel("max |response|")
    plt.title("Grid 31×31: response to a unit local load, (L₀+μI)⁻¹", fontsize=9)
    plt.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("fig_locality.png", dpi=200); plt.close()
    R["E10"] = out

# ---------- E11: power simulation for Hypothesis 12.2 ---------------------
def e11():
    """Quadratic vs sparse repair policy recovery. The analyst does not know the
    subject's true edge weights: it uses estimates with log-normal error (sd tau)."""
    z = np.array([1., 1, -1]); out = []
    for tau in [0.0, 0.5]:
        for sigma in [0.25, 0.5, 1.0]:
            for N in [5, 10, 20, 40]:
                correct = 0; reps = 1000
                for _ in range(reps):
                    truth = rng.integers(0, 2)              # 0 quadratic, 1 sparse
                    w = rng.uniform(0.5, 3, 3)
                    w_hat = w * np.exp(rng.normal(0, tau, 3))
                    ll = np.zeros(2)
                    for _ in range(N):
                        kap = rng.choice([-1, -0.5, 0.5, 1])
                        def policies(wt):
                            cq = -kap * (z / wt) / (z @ (z / wt))
                            cs = np.zeros(3); j = int(np.argmin(wt)); cs[j] = -kap * z[j]
                            return cq, cs
                        tq, ts = policies(w); hq, hs = policies(w_hat)
                        obs = (tq if truth == 0 else ts) + rng.normal(0, sigma, 3)
                        ll[0] -= np.sum((obs - hq) ** 2); ll[1] -= np.sum((obs - hs) ** 2)
                    correct += int(np.argmax(ll) == truth)
                p = correct / reps
                out.append(dict(tau=tau, sigma=sigma, trials_per_subject=N, recovery=p,
                                se=math.sqrt(p * (1 - p) / reps)))
    R["E11"] = out

if __name__ == "__main__":
    for f in (e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11):
        f(); print(f.__name__, "done", flush=True)
    R["seed"] = SEED; R["numpy"] = np.__version__
    json.dump(R, open("results.json", "w"), indent=1, default=float)
