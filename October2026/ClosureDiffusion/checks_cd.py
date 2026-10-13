"""Checks for the Closure Diffusion conceptual section (VE parameterisation x = x0 + sigma*eps)."""
import numpy as np
rng = np.random.default_rng(0)
ok = True
def report(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info)

# K1 Tweedie for a weighted point prior: E[x0|x] = sum r_t x_t, r_t ~ w_t exp(-|x-x_t|^2/(2 sigma^2))
X = np.array([[0.2, 0.2], [0.8, 0.8], [0.2, 0.9]]); w = np.array([0.5, 0.3, 0.2]); sig = 0.3
idx = rng.choice(3, 2_000_000, p=w); x0 = X[idx]; x = x0 + sig * rng.standard_normal(x0.shape)
q = np.array([0.5, 0.55]); near = np.linalg.norm(x - q, axis=1) < 0.02
mc = x0[near].mean(0)
r = w * np.exp(-((q - X) ** 2).sum(1) / (2 * sig ** 2)); r /= r.sum(); cf = r @ X
report("K1 Tweedie posterior mean of a point-mass prior", np.abs(mc - cf).max() < 0.01, f"MC {mc.round(3)} vs closed {cf.round(3)}")

# K2 product of two unimodal priors collapses to the midpoint; mixture keeps both modes
a, b, tau = 0.2, 0.8, 0.05
grid = np.linspace(-0.5, 1.5, 200001); N = lambda m, s: np.exp(-(grid - m) ** 2 / (2 * s * s))
prod = N(a, tau) * N(b, tau); prod /= prod.sum()
mean, sd = (grid * prod).sum(), np.sqrt(((grid - (grid * prod).sum()) ** 2 * prod).sum())
mix = 0.5 * N(a, tau) + 0.5 * N(b, tau)
report("K2a product = N((a+b)/2, tau^2/2)", abs(mean - 0.5) < 1e-6 and abs(sd - tau / np.sqrt(2)) < 1e-6, f"mean {mean:.4f} sd {sd:.4f}")
report("K2b product density at either optimum relative to its peak = exp(-(b-a)^2/(4 tau^2))",
       np.isclose(N(0.5, tau / np.sqrt(2))[np.argmin(abs(grid - a))], np.exp(-(b - a) ** 2 / (4 * tau ** 2))), f"{np.exp(-(b-a)**2/(4*tau**2)):.2e}")
report("K2c mixture has modes at a and b", abs(grid[np.argmax(mix * (grid < 0.5))] - a) < 1e-3 and abs(grid[np.argmax(mix * (grid > 0.5))] - b) < 1e-3)

# K3 gate formula for a mixture prior: E[x0|x] = g E_I + (1-g) E_O with g = a p_I / (a p_I + (1-a) p_O)
XI, wI = np.array([[0.2, 0.2], [0.25, 0.15]]), np.array([0.6, 0.4]); XO, wO = np.array([[0.8, 0.8]]), np.array([1.0]); A = 0.7
def marg(xq, P, W, s): return (W * np.exp(-((xq - P) ** 2).sum(1) / (2 * s * s))).sum()
def post(xq, P, W, s):
    rr = W * np.exp(-((xq - P) ** 2).sum(1) / (2 * s * s)); return rr @ P / rr.sum()
xq = np.array([0.4, 0.5]); s = 0.25
full = post(xq, np.vstack([XI, XO]), np.concatenate([A * wI, (1 - A) * wO]), s)
g = A * marg(xq, XI, wI, s) / (A * marg(xq, XI, wI, s) + (1 - A) * marg(xq, XO, wO, s))
report("K3 mixture posterior = x-dependent gate of component posteriors", np.allclose(full, g * post(xq, XI, wI, s) + (1 - g) * post(xq, XO, wO, s)), f"gate {g:.3f}")

# K4 gate limits: -> prior weight A as sigma -> inf; -> 0/1 as sigma -> 0
gs = [A * marg(xq, XI, wI, s) / (A * marg(xq, XI, wI, s) + (1 - A) * marg(xq, XO, wO, s)) for s in (100.0, 0.02)]
report("K4 gate -> prior weight at high noise, -> hard at low noise", abs(gs[0] - A) < 1e-4 and min(gs[1], 1 - gs[1]) < 1e-6, f"{gs[0]:.5f}, {gs[1]:.2e}")

# K5 recall decision scale: two equal-weight points at distance D; for a draw x = x_true + sigma*eps,
# probability the gate picks the true point falls from ~1 to ~1/2 as sigma passes through D; midpoint near D/2 .. 2D
D = 1.0; P2 = np.array([[0.0], [D]])
def p_correct(s, n=200000):
    xx = 0.0 + s * rng.standard_normal(n)
    return np.mean(np.abs(xx - 0.0) < np.abs(xx - D))   # equal weights: gate picks nearer point
pc = {s: p_correct(s) for s in (0.05 * D, 0.5 * D, D, 2 * D, 20 * D)}
report("K5 recall is decided at sigma of order D", pc[0.05 * D] > 0.99 and pc[20 * D] < 0.52 and 0.6 < pc[D] < 0.75,
       " ".join(f"s={k:g}:{v:.3f}" for k, v in pc.items()))
print("ALL PASS" if ok else "SOME FAIL")
