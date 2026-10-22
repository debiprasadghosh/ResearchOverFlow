"""Golden reference values for the C# reference runtime (Section 32). Seeded; writes golden.json."""
import json, math
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261021 + 7)
G = {}

def rank_f2(M):
    M = (M.copy() % 2).astype(np.uint8); r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r

def d0_f2(n, edges):
    D = np.zeros((len(edges), n), np.uint8)
    for e, (u, v) in enumerate(edges):
        D[e, u] ^= 1; D[e, v] ^= 1
    return D

# --- F2 core (Appendix A) -----------------------------------------------------------------------
f2 = []
for _ in range(60):
    n = int(rng.integers(3, 14)); m = int(rng.integers(n - 1, 2 * n + 2))
    edges = []
    for _ in range(m):
        u = int(rng.integers(0, n)); v = int(rng.integers(0, n)) if rng.random() > 0.08 else u   # occasional loop
        edges.append([u, v])
    D = d0_f2(n, edges); r = rank_f2(D)
    g = rng.integers(0, 2, m).astype(np.uint8)
    exact = rank_f2(np.concatenate([D, g[:, None]], axis=1)) == r
    f2.append(dict(n=n, edges=edges, H0=n - r, H1=m - r, g=g.tolist(), exact=bool(exact)))
G["f2"] = f2

# --- chi-square upper tail ------------------------------------------------------------------------
chi = []
for df in [1, 2, 3, 5, 10, 17, 40, 66, 100, 250]:
    for x in [1e-3, 0.1, 0.5, 1.0, 2.0, 0.5 * df, df, 1.5 * df, 2 * df + 3, 3 * df + 10, 5 * df + 40, 10 * df + 100]:
        chi.append(dict(df=df, x=float(x), q=float(stats.chi2.sf(x, df))))
G["chi2"] = chi

# --- calibrated cycle verifier (Proposition 29.4 statistic) ---------------------------------------
def incidence(n, edges):
    D = np.zeros((len(edges), n))
    for e, (u, v) in enumerate(edges): D[e, u] -= 1; D[e, v] += 1
    return D

ver = []
for _ in range(40):
    while True:
        n = int(rng.integers(6, 30)); p = min(1.0, 3.5 / n)
        A = np.triu(rng.random((n, n)) < p, 1)
        edges = [[int(i), int(j)] for i, j in zip(*np.nonzero(A))]
        import networkx as nx
        Gx = nx.Graph(); Gx.add_nodes_from(range(n)); Gx.add_edges_from(edges)
        if nx.is_connected(Gx) and len(edges) > n: break
    m = len(edges); w = rng.uniform(0.2, 5.0, m); sigma = float(rng.uniform(0.5, 2.0))
    D = incidence(n, edges); s = rng.normal(0, 1, n)
    g = D @ s + sigma * rng.normal(0, 1, m) / np.sqrt(w)
    if rng.random() < 0.5:                                   # half the instances carry an extra cyclic offset
        g[int(rng.integers(0, m))] += float(rng.uniform(2, 6)) * sigma
    Wh = np.sqrt(w); Q, sv, _ = np.linalg.svd(Wh[:, None] * D, full_matrices=False)
    Q = Q[:, : int((sv > 1e-10 * sv[0]).sum())]
    r = g - (Q @ (Q.T @ (Wh * g))) / Wh
    T = float(r @ (w * r)) / sigma ** 2; h = m - n + 1
    ver.append(dict(n=n, edges=edges, w=w.tolist(), g=g.tolist(), sigma=sigma, h=h, T=T, p=float(stats.chi2.sf(T, h))))
G["verifier"] = ver

# --- SE(2) holonomy of fundamental walks (deterministic BFS: roots ascending, neighbours ascending) ---
def se2_mul(a, b):
    th, x, y = a; th2, x2, y2 = b
    c, s = math.cos(th), math.sin(th)
    return ((th + th2 + math.pi) % (2 * math.pi) - math.pi, x + c * x2 - s * y2, y + s * x2 + c * y2)
def se2_inv(a):
    th, x, y = a; c, s = math.cos(th), math.sin(th)
    return ((-th + math.pi) % (2 * math.pi) - math.pi, -(c * x + s * y), -(-s * x + c * y))
def bfs_tree(n, edges):
    adj = {u: [] for u in range(n)}
    for e, (u, v) in enumerate(edges):
        if u != v: adj[u].append((v, e)); adj[v].append((u, e))
    for u in adj: adj[u].sort()
    parent = [-1] * n; pedge = [-1] * n; seen = [False] * n; order = []
    for r in range(n):
        if seen[r]: continue
        seen[r] = True; queue = [r]
        while queue:
            u = queue.pop(0); order.append(u)
            for v, e in adj[u]:
                if not seen[v]: seen[v] = True; parent[v] = u; pedge[v] = e; queue.append(v)
    return parent, pedge, order

hol = []
for case in range(30):
    n = int(rng.integers(4, 14))
    while True:
        A = np.triu(rng.random((n, n)) < 0.45, 1)
        edges = [[int(i), int(j)] for i, j in zip(*np.nonzero(A))]
        import networkx as nx
        Gx = nx.Graph(); Gx.add_nodes_from(range(n)); Gx.add_edges_from(edges)
        if nx.is_connected(Gx) and len(edges) > n: break
    poses = [(float(rng.uniform(-math.pi, math.pi)), float(rng.normal(0, 3)), float(rng.normal(0, 3))) for _ in range(n)]
    meas = [se2_mul(se2_inv(poses[u]), poses[v]) for u, v in edges]
    perturbed = -1
    if case % 2 == 1:                                         # perturb one edge measurement
        perturbed = int(rng.integers(0, len(edges)))
        th, x, y = meas[perturbed]; meas[perturbed] = (th + 0.3, x + 0.5, y - 0.2)
    parent, pedge, order = bfs_tree(n, edges)
    s = [None] * n
    for u in order:
        if parent[u] < 0: s[u] = (0.0, 0.0, 0.0)
        else:
            p_, e = parent[u], pedge[u]
            m_ = meas[e] if edges[e][0] == p_ else se2_inv(meas[e])
            s[u] = se2_mul(s[p_], m_)
    tree_edges = {pedge[u] for u in range(n) if parent[u] >= 0}
    walks = []
    for e, (u, v) in enumerate(edges):
        if e in tree_edges: continue
        Hh = se2_mul(se2_mul(s[u], meas[e]), se2_inv(s[v]))
        walks.append(dict(edge=e, theta=Hh[0], x=Hh[1], y=Hh[2]))
    hol.append(dict(n=n, edges=edges, meas=[list(mm) for mm in meas], perturbed=perturbed, walks=walks))
G["holonomy"] = hol

# --- angular synchronisation with certificate -----------------------------------------------------
def sync_matrix(n, edges, ang, w):
    M = np.zeros((n, n), complex)
    for (u, v), a, we in zip(edges, ang, w):
        r = np.exp(1j * a); M[u, u] += we; M[v, v] += we; M[u, v] += -we * np.conj(r); M[v, u] += -we * r
    return M
def gpm(M, z0, iters=5000, tol=1e-14):
    c = np.linalg.eigvalsh(M)[-1] + 1.0; z = z0 / np.abs(z0)
    for _ in range(iters):
        y = c * z - M @ z; zn = y / np.abs(y)
        if np.abs(zn - z).max() < tol: return zn
        z = zn
    return z
syn = []
for case in range(24):
    n = int(rng.integers(8, 26)); sigma = [0.2, 0.5, 0.9, 1.6][case % 4]
    while True:
        A = np.triu(rng.random((n, n)) < 0.35, 1)
        edges = [[int(i), int(j)] for i, j in zip(*np.nonzero(A))]
        import networkx as nx
        Gx = nx.Graph(); Gx.add_nodes_from(range(n)); Gx.add_edges_from(edges)
        if nx.is_connected(Gx): break
    th = rng.uniform(-math.pi, math.pi, n)
    ang = [float(th[v] - th[u] + rng.normal(0, sigma)) for u, v in edges]
    w = rng.uniform(0.5, 2.0, len(edges)).tolist()
    M = sync_matrix(n, edges, ang, w)
    z = gpm(M, np.linalg.eigh(M)[1][:, 0])
    lam = np.real(np.conj(z) * (M @ z)); S = M - np.diag(lam); lmin = float(np.linalg.eigvalsh(S)[0])
    cost = float(np.real(np.conj(z) @ M @ z))
    syn.append(dict(n=n, edges=edges, angles=ang, w=w, sigma=sigma, cost=cost, lambda_min=lmin,
                    certified=bool(lmin >= -1e-9 * np.abs(M).sum(axis=1).max())))
G["sync"] = syn
json.dump(G, open("golden.json", "w"))
print({k: len(v) for k, v in G.items()})
