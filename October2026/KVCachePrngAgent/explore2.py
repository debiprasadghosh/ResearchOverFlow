"""(a) separation vs pair distance; (b) min-count bound; (c) marginal non-identifiability;
(d) read error with key-dependent values across attention sharpness."""
import numpy as np, json
rng = np.random.default_rng(11)
dk, dv, B, K = 16, 64, 16, 3
out = {}

def assign(X, C):
    return (((X[:, None, :] - C[None]) ** 2).sum(-1)).argmin(1)

def softmax_rows(Z):
    Z = Z - Z.max(-1, keepdims=True); E = np.exp(Z); return E / E.sum(-1, keepdims=True)

# (a) alias probability vs separation delta, isotropic keys, many codebooks
deltas = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
P = 4000; ncb = 300
x = rng.normal(size=(P, dk)); u = rng.normal(size=(P, dk)); u /= np.linalg.norm(u, axis=1, keepdims=True)
curve = {"delta": deltas.tolist(), "p16": [], "colony3": [], "p48": [], "p128": [], "colony3_naive": []}
for dlt in deltas:
    y = x + dlt * u
    s16 = []; s48 = []; s128 = []
    for _ in range(ncb // 3):
        C = rng.normal(size=(B, dk)); s16.append(assign(x, C) == assign(y, C))
        C = rng.normal(size=(3 * B, dk)); s48.append(assign(x, C) == assign(y, C))
    for _ in range(20):
        C = rng.normal(size=(8 * B, dk)); s128.append(assign(x, C) == assign(y, C))
    s16 = np.array(s16)                       # (codebooks, pairs)
    p_pair = s16.mean(0)                      # per-pair alias prob estimate
    curve["p16"].append(float(s16.mean()))
    # colony: exact conditional formula E[p(x,y)^3] estimated with unbiased product of 3 disjoint codebooks
    trip = s16[: (len(s16) // 3) * 3].reshape(-1, 3, P).all(1)
    curve["colony3"].append(float(trip.mean()))
    curve["colony3_naive"].append(float(s16.mean() ** 3))
    curve["p48"].append(float(np.mean(s48)))
    curve["p128"].append(float(np.mean(s128)))
out["curve"] = curve
small = 0
r = (1 - np.array(curve["colony3"][:3])) / (1 - np.array(curve["p48"][:3]))
out["small_delta_sep_ratio_colony_vs_48"] = r.tolist()
out["predicted_ratio_K^(1-1/d)"] = 3 ** (1 - 1 / dk)
r2 = (1 - np.array(curve["p48"][:3])) / (1 - np.array(curve["p16"][:3]))
out["small_delta_sep_ratio_48_vs_16"] = r2.tolist()
out["predicted_3^(1/d)"] = 3 ** (1 / dk)
r3 = (1 - np.array(curve["p128"][:3])) / (1 - np.array(curve["p16"][:3]))
out["small_delta_sep_ratio_128_vs_16"] = r3.tolist()
out["predicted_8^(1/d)"] = 8 ** (1 / dk)

# (b) min-count bound: count in common-refinement cell <= min_s n_{s,b_s(x)}
T = 2000
X = rng.normal(size=(T, dk)); Cs = [rng.normal(size=(B, dk)) for _ in range(K)]
As = np.stack([assign(X, C) for C in Cs])          # K x T
ns = [np.bincount(a, minlength=B) for a in As]
viol = 0; ratio = []
for i in range(0, T, 7):
    cell = (As == As[:, [i]]).all(0).sum()
    mn = min(ns[s][As[s, i]] for s in range(K))
    viol += cell > mn; ratio.append(cell / mn)
out["mincount_violations"] = int(viol)
out["mincount_mean_cell_over_min"] = float(np.mean(ratio))
out["mincount_mean_single_over_min"] = float(np.mean([ns[0][As[0, i]] / min(ns[s][As[s, i]] for s in range(K)) for i in range(0, T, 7)]))

# (c) marginal non-identifiability: two multisets, identical K=2 count states
C1, C2 = rng.normal(size=(B, dk)), rng.normal(size=(B, dk))
Z = rng.normal(size=(200000, dk)); a1, a2 = assign(Z, C1), assign(Z, C2)
pairs = {}
for z, i, j in zip(Z, a1, a2):
    pairs.setdefault((i, j), z)
found = None
for (i, j) in pairs:
    for (k, l) in pairs:
        if i != k and j != l and (i, l) in pairs and (k, j) in pairs:
            found = ((i, j), (k, l), (i, l), (k, j)); break
    if found: break
S1 = np.stack([pairs[found[0]], pairs[found[1]]]); S2 = np.stack([pairs[found[2]], pairs[found[3]]])
same_counts = all(np.array_equal(np.bincount(assign(S1, C), minlength=B), np.bincount(assign(S2, C), minlength=B)) for C in (C1, C2))
out["nonident_cells"] = [list(map(int, c)) for c in found]
out["nonident_same_counts_both_visitors"] = bool(same_counts)
out["nonident_multisets_differ"] = bool(not np.allclose(S1, S2))

# (d) read error with key-dependent values, sweep sharpness beta
W = rng.normal(size=(dk, dv)) / np.sqrt(dk)
def kmeans(Xd, k, it=30):
    C = Xd[rng.choice(len(Xd), k, replace=False)].copy()
    for _ in range(it):
        a = assign(Xd, C)
        for j in range(k):
            if (a == j).any(): C[j] = Xd[a == j].mean(0)
    return C
def counted_read(Q, C, a, V):
    n = np.bincount(a, minlength=len(C)).astype(float)
    S = np.zeros((len(C), V.shape[1])); np.add.at(S, a, V); m = n > 0
    return softmax_rows(Q @ C[m].T + np.log(n[m])) @ (S[m] / n[m][:, None])
def make(kind, T):
    if kind == "isotropic": return rng.normal(size=(T, dk))
    Qb, _ = np.linalg.qr(rng.normal(size=(dk, 4)))
    return (rng.normal(size=(T, 4)) * 2.0) @ Qb.T + 0.1 * rng.normal(size=(T, dk))
rd = {}
for kind in ["isotropic", "subspace"]:
    for beta in [0.25, 1.0, 2.0]:
        e = {k: [] for k in ["1x16", "mean3x16", "1x48", "learned16"]}
        for _ in range(25):
            X = make(kind, 512); sc = X.std(); Xn = X / sc
            V = np.tanh(Xn @ W * 2) + 0.1 * rng.normal(size=(512, dv))
            Qm = rng.normal(size=(64, dk)) * beta / np.sqrt(dk)
            ref = softmax_rows(Qm @ Xn.T) @ V
            vc = V.mean(0); den = np.linalg.norm(ref - vc, axis=1)  # error relative to deviation from mean value
            rel = lambda Y: float(np.mean(np.linalg.norm(Y - ref, axis=1) / den))
            Cs = [rng.normal(size=(B, dk)) for _ in range(3)]
            reads = [counted_read(Qm, C, assign(Xn, C), V) for C in Cs]
            e["1x16"].append(rel(reads[0])); e["mean3x16"].append(rel(np.mean(reads, 0)))
            Cb = rng.normal(size=(48, dk)); e["1x48"].append(rel(counted_read(Qm, Cb, assign(Xn, Cb), V)))
            Cl = kmeans(Xn, 16); e["learned16"].append(rel(counted_read(Qm, Cl, assign(Xn, Cl), V)))
        rd[f"{kind}_beta{beta}"] = {k: (float(np.mean(v)), float(np.std(v) / 5)) for k, v in e.items()}
out["read_error"] = rd
print(json.dumps(out, indent=1))
json.dump(out, open("explore2.json", "w"), indent=1)
