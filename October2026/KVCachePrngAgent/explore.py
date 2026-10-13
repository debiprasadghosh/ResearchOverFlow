"""Diversity vs detail at matched state: K seeded B-bin codebooks vs one KB-bin codebook."""
import numpy as np, json
rng = np.random.default_rng(7)
dk, dv, B, K = 16, 64, 16, 3

def softmax_rows(Z):
    Z = Z - Z.max(-1, keepdims=True); E = np.exp(Z); return E / E.sum(-1, keepdims=True)

def assign(X, C):
    return (((X[:, None, :] - C[None]) ** 2).sum(-1)).argmin(1)

def kmeans(X, k, it=40):
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(it):
        a = assign(X, C)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return C

def make_keys(kind, T):
    if kind == "isotropic":
        return rng.normal(size=(T, dk))
    # low intrinsic dimension: 4-D subspace + small noise, same overall scale
    Q, _ = np.linalg.qr(rng.normal(size=(dk, 4)))
    return (rng.normal(size=(T, 4)) * 2.0) @ Q.T + 0.1 * rng.normal(size=(T, dk))

def codebook(kind_scale, n_bins):
    return rng.normal(size=(n_bins, dk)) * kind_scale

def eff_bins(a, n_bins):
    p = np.bincount(a, minlength=n_bins) / len(a); p = p[p > 0]
    return np.exp(-(p * np.log(p)).sum()), (np.bincount(a, minlength=n_bins) > 0).sum()

def counted_read(Q, C, a, V):
    n = np.bincount(a, minlength=len(C)).astype(float)
    S = np.zeros((len(C), V.shape[1])); np.add.at(S, a, V)
    m = n > 0
    W = softmax_rows(Q @ C[m].T + np.log(n[m]))
    return W @ (S[m] / n[m][:, None])

res = {}
for kind in ["isotropic", "subspace"]:
    trials = 40; T = 512; nq = 64
    acc = {k: [] for k in ["alias_1x16", "alias_colony3x16", "alias_1x48", "alias_learned16",
                            "naive_product", "eff_1x16", "occ_1x16", "eff_learned16",
                            "err_1x16", "err_mean3x16", "err_1x48", "err_learned16",
                            "near_alias_1x16", "near_alias_colony", "near_alias_1x48"]}
    for _ in range(trials):
        X = make_keys(kind, T); V = rng.normal(size=(T, dv))
        scale = X.std()
        Cs = [codebook(scale, B) for _ in range(K)]
        Cb = codebook(scale, K * B)
        Cl = kmeans(X, B)
        As = [assign(X, C) for C in Cs]; Ab = assign(X, Cb); Al = assign(X, Cl)
        # pairwise aliasing over all pairs
        iu = np.triu_indices(T, 1)
        same = [(A[:, None] == A[None])[iu] for A in As]
        acc["alias_1x16"].append(same[0].mean())
        acc["alias_colony3x16"].append(np.logical_and.reduce(same).mean())
        acc["naive_product"].append(np.mean([s.mean() for s in same]) ** K)
        acc["alias_1x48"].append((Ab[:, None] == Ab[None])[iu].mean())
        acc["alias_learned16"].append((Al[:, None] == Al[None])[iu].mean())
        # near pairs: closest 1% by distance
        Dm = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))[iu]
        near = Dm <= np.quantile(Dm, 0.01)
        acc["near_alias_1x16"].append(same[0][near].mean())
        acc["near_alias_colony"].append(np.logical_and.reduce(same)[near].mean())
        acc["near_alias_1x48"].append((Ab[:, None] == Ab[None])[iu][near].mean())
        e, o = eff_bins(As[0], B); acc["eff_1x16"].append(e); acc["occ_1x16"].append(o)
        acc["eff_learned16"].append(eff_bins(Al, B)[0])
        # read error vs original-key token reference (relative L2)
        Qm = rng.normal(size=(nq, dk)) * (0.5 / scale)
        ref = softmax_rows(Qm @ X.T) @ V
        nr = np.linalg.norm(ref, axis=1)
        rel = lambda Y: float(np.mean(np.linalg.norm(Y - ref, axis=1) / nr))
        reads = [counted_read(Qm, C, A, V) for C, A in zip(Cs, As)]
        acc["err_1x16"].append(rel(reads[0]))
        acc["err_mean3x16"].append(rel(np.mean(reads, 0)))
        acc["err_1x48"].append(rel(counted_read(Qm, Cb, Ab, V)))
        acc["err_learned16"].append(rel(counted_read(Qm, Cl, Al, V)))
    res[kind] = {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) for k, v in acc.items()}

for kind, d in res.items():
    print("==", kind)
    for k, (m, se) in d.items():
        print(f"  {k:22s} {m:.4f} ± {se:.4f}")
json.dump(res, open("explore.json", "w"), indent=1)
