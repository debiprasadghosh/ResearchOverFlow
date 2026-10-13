"""Executable mechanism checks for Same Flower (Section 11 list + arithmetic)."""
import numpy as np
from math import isclose, tanh

rng = np.random.default_rng(20261012)
out = {}

# ---------- arithmetic in the paper ----------
B, dk, dv = 16, 16, 64
pay = lambda pn, pv: 4 + B * (pn + dv * pv)
out["table2"] = dict(seed=4, counts=4 + B * 4, fp16=pay(4, 2), fp32=pay(4, 4), fp32_u64=pay(8, 4))
out["workspace_fp16"] = B * (dk + dv) * 2
out["three_active"] = 4 * 10**6 + 3 * (pay(4, 4) - 4)
out["macs_per_token_1M"] = 10**6 * B * (dk + dv)
M = 10**6
out["coll32"] = M * (M - 1) / 2**33
out["coll64"] = M * (M - 1) / 2**65
out["crossover_T_fp32"] = pay(4, 4) / 160      # token cache 160T bytes
out["crossover_T_fp16"] = pay(4, 2) / 160
n, S = 10, 20.0
out["reopen"] = dict(resid=(n - 1, S - 8), combined_mean=(S - 8 + 8) / n, wrong=(S + 8) / (n + 1))

# ---------- helpers ----------
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def token_read(q, K, V):
    return softmax(K @ q) @ V

def counted_read(q, C, n, Ssum):
    m = n > 0
    w = softmax(C[m] @ q + np.log(n[m]))
    return w @ (Ssum[m] / n[m][:, None])

def assign(K, C):
    d = ((K[:, None, :] - C[None]) ** 2).sum(-1)
    return d.argmin(1)  # argmin breaks ties by lowest index = stable identity

def accumulate(a, V, B):
    n = np.bincount(a, minlength=B).astype(float)
    S = np.zeros((B, V.shape[1])); np.add.at(S, a, V)
    return n, S

# ---------- Prop 1: equal-score aggregation (quantized reference) ----------
errs = []
for _ in range(200):
    T = 300
    C = rng.normal(size=(B, dk)); K = rng.normal(size=(T, dk)); V = rng.normal(size=(T, dv))
    q = rng.normal(size=dk) / 4
    a = assign(K, C); n, S = accumulate(a, V, B)
    errs.append(np.abs(counted_read(q, C, n, S) - token_read(q, C[a], V)).max())
out["prop1_max_abs_err"] = max(errs)

# ---------- Corollary: quantization error vs original-key reference ----------
ratios = []
for _ in range(500):
    T = 200
    C = rng.normal(size=(B, dk)); K = rng.normal(size=(T, dk)); V = rng.normal(size=(T, dv))
    q = rng.normal(size=dk) * rng.uniform(0.05, 0.6)
    a = assign(K, C); n, S = accumulate(a, V, B)
    rho = np.linalg.norm(K - C[a], axis=1).max()
    D = max(np.linalg.norm(V[i] - V[j]) for i, j in rng.integers(0, T, (2000, 2)))  # lower est. of diameter
    D_true = D  # conservative check uses an underestimate of D -> bound smaller -> stricter test
    err = np.linalg.norm(counted_read(q, C, n, S) - token_read(q, K, V))
    bound = D_true * tanh(np.linalg.norm(q) * rho / 2)
    ratios.append(err / bound)
out["cor_quant_max_err_over_bound"] = max(ratios)

# ---------- Paired variant: S = n u, read in convex hull ----------
U = rng.normal(size=(B, dv)); C = rng.normal(size=(B, dk)); K = rng.normal(size=(500, dk))
a = assign(K, C); n = np.bincount(a, minlength=B).astype(float)
y1 = counted_read(rng.normal(size=dk), C, n, n[:, None] * U)
out["paired_affine_dim"] = np.linalg.matrix_rank(U[1:] - U[0])

# ---------- Prop 4: paired permutation leaves the read unchanged ----------
perm = rng.permutation(B)
K = rng.normal(size=(400, dk)); V = rng.normal(size=(400, dv)); q = rng.normal(size=dk) / 3
a = assign(K, C); n, S = accumulate(a, V, B)
inv = np.argsort(perm)
out["prop4_perm_err"] = np.abs(counted_read(q, C, n, S) - counted_read(q, C[perm], n[perm], S[perm])).max()
a2 = perm[assign(K, C[perm])]  # reassign under permuted codebook, map back to original ids
out["prop4_assignment_equivariant"] = bool((a2 == a).all())
# reversing only values (keys fixed) changes the paired read:
Ur = U[::-1]
nn = np.bincount(a, minlength=B).astype(float)
out["values_only_reversal_changes"] = float(np.abs(counted_read(q, C, nn, nn[:, None] * U) - counted_read(q, C, nn, nn[:, None] * Ur)).max())

# ---------- Prop 2: AB vs BA collide ----------
emb = rng.normal(size=(2, dk)); val = rng.normal(size=(2, dv))
def state(seq):
    ids = np.array(seq); return accumulate(assign(emb[ids], C), val[ids], B)
sAB, sBA = state([0, 1]), state([1, 0])
out["prop2_AB_eq_BA"] = bool(np.array_equal(sAB[0], sBA[0]) and np.allclose(sAB[1], sBA[1]))

# ---------- reopening conserves counts and sums ----------
vals = np.array([1, 2, 3, 1, 0, 8, 1, 2, 1, 1.0]); assert vals.sum() == 20
resid_n, resid_S = 10 - 1, 20 - 8
out["reopen_conserve"] = (resid_n + 1, resid_S + 8, (resid_S + 8) / (resid_n + 1))

# ---------- checkpoint + replay == uninterrupted; stale fails ----------
T = 1000; K = rng.normal(size=(T, dk)); V = rng.normal(size=(T, dv)); a = assign(K, C)
full = accumulate(a, V, B)
t0 = 600
ck = accumulate(a[:t0], V[:t0], B)
rep = accumulate(a[t0:], V[t0:], B)
out["replay_exact"] = bool(np.array_equal(full[0], ck[0] + rep[0]) and np.allclose(full[1], ck[1] + rep[1]))
out["stale_count_deficit"] = float(full[0].sum() - ck[0].sum())

# ---------- fp16 accumulation failure (Section 2.2) ----------
s16 = np.float16(0); s32 = np.float32(0)
for _ in range(5000):
    s16 = np.float16(s16 + np.float16(1.0)); s32 = np.float32(s32 + np.float32(1.0))
out["fp16_count_after_5000_ones"] = float(s16)
out["fp32_count_after_5000_ones"] = float(s32)
x16 = np.float16(0)
for _ in range(100000):
    x16 = np.float16(x16 + np.float16(0.01))
out["fp16_sum_of_1e5_x_0.01"] = float(x16)

for k, v in out.items():
    print(f"{k}: {v}")
