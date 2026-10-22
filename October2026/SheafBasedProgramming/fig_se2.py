"""Figure 30.1: dead-reckoned (open) square loop versus certified SE(2) repair, for two heading discrepancies."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sync_matrix(n, edges, rot, w):
    M = np.zeros((n, n), complex)
    for (u, v), r, we in zip(edges, rot, w):
        M[u, u] += we; M[v, v] += we; M[u, v] += -we * np.conj(r); M[v, u] += -we * r
    return M

def solve_loop(delta, L=4.0, kw=(4.0, 1.0, 4.0, 1.0), tw=(4.0, 1.0, 4.0, 1.0), seed=1):
    rng = np.random.default_rng(seed)
    turns = [math.pi / 2] * 4; turns[-1] += delta
    E2 = [(0, 1), (1, 2), (2, 3), (3, 0)]
    R2 = [np.exp(1j * a) for a in turns]; T2 = [L + 0j] * 4
    kw, tw = np.array(kw), np.array(tw); m, n = 4, 4
    Mr = sync_matrix(n, E2, R2, kw)
    Dt = np.zeros((m, n), complex); B = np.zeros((m, n), complex)
    for e, (u, v) in enumerate(E2): Dt[e, u] = -1; Dt[e, v] = 1; B[e, u] = T2[e]
    Om = np.diag(tw.astype(complex)); Dg = Dt[:, 1:]; A = Dg.conj().T @ Om @ Dg
    P = Dg @ np.linalg.solve(A, Dg.conj().T @ Om)
    M = Mr + B.conj().T @ Om @ (np.eye(m) - P) @ B; M = (M + M.conj().T) / 2
    best = None
    c = np.linalg.eigvalsh(M)[-1] + 1
    for start in [np.linalg.eigh(M)[1][:, 0]] + [np.exp(1j * rng.uniform(-math.pi, math.pi, n)) for _ in range(30)]:
        z = start / np.abs(start)
        for _ in range(4000):
            y = c * z - M @ z; zn = y / np.abs(y)
            if np.abs(zn - z).max() < 1e-14: z = zn; break
            z = zn
        f = float(np.real(np.conj(z) @ M @ z))
        if best is None or f < best[1]: best = (z, f)
    z = best[0] * np.conj(best[0][0])
    t = np.concatenate([[0], np.linalg.solve(A, Dg.conj().T @ Om @ (B @ z))])
    lam = np.real(np.conj(z) * (M @ z)); lmin = float(np.linalg.eigvalsh(M - np.diag(lam))[0])
    # dead reckoning from the measurements (open chain 0 -> 1 -> 2 -> 3 -> 0')
    pose_z, pose_t = [1 + 0j], [0j]
    for a in turns:
        pose_t.append(pose_t[-1] + pose_z[-1] * L); pose_z.append(pose_z[-1] * np.exp(1j * a))
    return z, t, np.array(pose_t), np.array(pose_z), lmin

def two_stage(delta, L=4.0, kw=np.array([4.0, 1.0, 4.0, 1.0]), tw=np.array([4.0, 1.0, 4.0, 1.0])):
    turns = [math.pi / 2] * 4; turns[-1] += delta
    E2 = [(0, 1), (1, 2), (2, 3), (3, 0)]
    R2 = [np.exp(1j * a) for a in turns]; T2 = [L + 0j] * 4
    zc = np.ones(4); ang = np.array([np.angle(r) for r in R2])
    k = float(zc @ ang); k = (k + math.pi) % (2 * math.pi) - math.pi
    fitted = ang - k * (zc / kw) / (zc @ (zc / kw))
    ph = [0.0]
    for q in range(3): ph.append(ph[-1] + fitted[q])
    z = np.exp(1j * np.array(ph))
    Dt = np.zeros((4, 4), complex); B = np.zeros((4, 4), complex)
    for e, (u, v) in enumerate(E2): Dt[e, u] = -1; Dt[e, v] = 1; B[e, u] = T2[e]
    Om = np.diag(tw.astype(complex)); Dg = Dt[:, 1:]; A = Dg.conj().T @ Om @ Dg
    t = np.concatenate([[0], np.linalg.solve(A, Dg.conj().T @ Om @ (B @ z))])
    return z, t

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6))
for ax, d in zip(axes, [15, 60]):
    z, t, pt, pz, lmin = solve_loop(math.radians(d))
    z2, t2 = two_stage(math.radians(d))
    ax.plot(pt.real, pt.imag, "--", color="#b0b7c3", lw=1.3, label="integrated measurements")
    ax.annotate("", xy=(0.9 * pz[-1].real, 0.9 * pz[-1].imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#8a929e", lw=1.0, ls="--"))
    l2 = np.concatenate([t2, t2[:1]])
    ax.plot(l2.real, l2.imag, ":", color="#c0612b", lw=1.6, label="two-stage heuristic")
    loop = np.concatenate([t, t[:1]])
    ax.plot(loop.real, loop.imag, "-", color="#2b5d8a", lw=1.8, label="certified joint repair")
    for k in range(4):
        ax.annotate("", xy=(t[k].real + 0.8 * z[k].real, t[k].imag + 0.8 * z[k].imag), xytext=(t[k].real, t[k].imag),
                    arrowprops=dict(arrowstyle="-|>", color="#2b5d8a", lw=1.0))
    for k, (dx, dy) in enumerate([(-0.45, -0.45), (0.15, -0.45), (0.15, 0.15), (-0.45, 0.15)]):
        ax.text(pt[k].real + dx, pt[k].imag + dy, str(k), fontsize=7.5, color="#33404d")
    ax.set_xlim(-1.2, 7.2); ax.set_ylim(-1.2, 5.6); ax.set_aspect("equal")
    ax.set_title(f"δ = {d}°;  λmin(S) = {lmin:.0e}", fontsize=8.5)
    ax.tick_params(labelsize=7); ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig("fig_se2.png", dpi=220); plt.close()
print("ok")
