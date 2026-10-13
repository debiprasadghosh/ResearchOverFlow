import numpy as np
from scipy.stats import norm
rng=np.random.default_rng(3); ok=True
def rep(n,c,i=""):
    global ok; ok&=bool(c); print(("PASS " if c else "FAIL ")+n,i)
# L1 KV-Tweedie: E[v | k+sigma eps] = sum r_t(z) v_t with r from keys
K=np.array([[0.,0.],[1.,0.3],[0.2,1.1]]); V=np.array([[5.,1.,0.],[-2.,3.,1.],[0.,0.,7.]]); q=np.array([.5,.3,.2]); s=.4
idx=rng.choice(3,3_000_000,p=q); z=K[idx]+s*rng.standard_normal((len(idx),2))
z0=np.array([.5,.4]); m=np.linalg.norm(z-z0,axis=1)<.02
r=q*np.exp(-((z0-K)**2).sum(1)/(2*s*s)); r/=r.sum()
se=V[idx[m]].std(0)/np.sqrt(m.sum())
rep("L1 KV-Tweedie exact (within 4 s.e.)", (np.abs(V[idx[m]].mean(0)-r@V)<=4*se+0.01).all(), f"{V[idx[m]].mean(0).round(2)} vs {(r@V).round(2)}")
# L2 Jensen gap bound for decoder G(z)=(sin z1, z2^2, z1 z2): Hessian norm bound M
def G(Z): return np.stack([np.sin(Z[...,0]),Z[...,1]**2,Z[...,0]*Z[...,1]],-1)
# operator norm of the vector Hessian bounded by sqrt(sum of squared Frobenius) <= sqrt(1+4+2)=sqrt7
M=np.sqrt(7.)
worst=0
for _ in range(200):
    P=rng.normal(size=(4,2)); w=rng.dirichlet(np.ones(4)); mu=w@P
    gap=np.linalg.norm(w@G(P)-G(mu)); bound=0.5*M*(w*((P-mu)**2).sum(1)).sum()
    worst=max(worst,gap/bound)
rep("L2 Jensen gap <= M/2 tr Cov", worst<=1+1e-9, f"max ratio {worst:.3f}")
# T3 weighted RMS identity and simplex optimality
for _ in range(200):
    n=5; Z=rng.normal(size=(n,4)); w=rng.dirichlet(np.ones(n)); c=w@Z
    lhs=(w*((Z-c)**2).sum(1)).sum(); D=np.linalg.norm(Z[:,None]-Z[None],axis=2)
    rhs=0.5*(w[:,None]*w[None]*D**2).sum(); Dmin=D[~np.eye(n,dtype=bool)].min()
    assert np.isclose(lhs,rhs) and lhs>=0.5*Dmin**2*(1-(w**2).sum())-1e-12
E=np.eye(5)/np.sqrt(2); w=rng.dirichlet(np.ones(5)); c=w@E   # regular simplex, edge 1
rep("T3 RMS identity + bound, simplex attains", np.isclose((w*((E-c)**2).sum(1)).sum(),0.5*(1-(w**2).sum())))
rep("T3b simplex circumradius", np.isclose(np.linalg.norm(E[0]-E.mean(0)),np.sqrt(4/10)))
# P product-latent decoupling: posterior over t independent of z_geo when chart shared
ec=np.eye(3)*2.0; zc=np.array([1.2,0.9,0.1]); zg=rng.normal(size=2); s=.7
def post(zc,zg,shared=True):
    lw=np.log(q)-((zc-ec)**2).sum(1)/(2*s*s)
    if not shared:
        mus=np.array([[0,0],[1,1],[-1,2]]); lw=lw-((zg-mus)**2).sum(1)/(2*(1+s*s))
    w=np.exp(lw-lw.max()); return w/w.sum()
rep("P5 decoupling with shared chart", np.allclose(post(zc,zg),post(zc,zg+5)))
rep("P5 coupling without shared chart", not np.allclose(post(zc,zg,False),post(zc,zg+5,False)))
# two-point misassignment = Phi_bar(d/2s) equal weights
d=0.5;s=0.3;x=s*rng.standard_normal(1_000_000)
rep("P misassignment two points", abs(np.mean(x>d/2)-norm.sf(d/(2*s)))<2e-3)
print("ALL PASS" if ok else "SOME FAIL")
