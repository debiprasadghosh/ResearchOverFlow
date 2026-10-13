import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
rng=np.random.default_rng(1); ok=True
def rep(n,c,i=""):
    global ok; ok&=bool(c); print(("PASS " if c else "FAIL ")+n,i)
# Square-root law
for _ in range(5):
    p=rng.dirichlet(np.ones(6)*0.5)
    obj=lambda z:(p/np.exp(z)*np.exp(z).sum()).sum()
    r=minimize(obj,np.zeros(6),method='Nelder-Mead',options={'maxiter':20000,'xatol':1e-10,'fatol':1e-12})
    q=np.exp(r.x);q/=q.sum()
    rep("sqrt law optimum", abs(r.fun-np.sqrt(p).sum()**2)<1e-5 and np.allclose(q,np.sqrt(p)/np.sqrt(p).sum(),atol=2e-3), f"{r.fun:.5f} vs {np.sqrt(p).sum()**2:.5f}; freq-weight cost {(p/p).sum():.0f}")
# Monte Carlo of geometric recall with sqrt weights
p=np.array([.6,.25,.1,.05]); q=np.sqrt(p)/np.sqrt(p).sum()
e=rng.choice(4,200000,p=p); draws=rng.geometric(q[e])
rep("MC recall cost", abs(draws.mean()-np.sqrt(p).sum()**2)<0.03, f"{draws.mean():.3f} vs {np.sqrt(p).sum()**2:.3f}")
# guessing bound
ps=np.sort(p)[::-1]; EG=(ps*np.arange(1,5)).sum()
rep("sorted guessing <= (sum sqrt p)^2", EG<=np.sqrt(p).sum()**2, f"{EG:.3f}")
# Markov conditioning
for _ in range(5):
    P=rng.dirichlet(np.ones(5)*0.4,size=5)
    w,v=np.linalg.eig(P.T); pi=np.real(v[:,np.argmin(abs(w-1))]); pi/=pi.sum()
    cond=(pi*np.sqrt(P).sum(1)**2).sum(); unc=np.sqrt(pi).sum()**2
    rep("conditional <= unconditional", cond<=unc+1e-12, f"{cond:.3f} <= {unc:.3f}")
# ratio bound
p=rng.dirichlet(np.ones(5)); ph=p*np.exp(rng.normal(0,.2,5)); ph/=ph.sum()
qh=np.sqrt(ph)/np.sqrt(ph).sum(); cost=(p/qh).sum(); a,b=np.sqrt(p),np.sqrt(ph)
rep("ratio bound", cost<=(b/a).max()*(a/b).max()*a.sum()**2+1e-12, f"{cost:.4f} <= {(b/a).max()*(a/b).max()*a.sum()**2:.4f}")
# Prior realisation via probability-flow ODE in 1D, VE
X=np.array([-1.0,0.3,1.5]); q=np.array([.5,.3,.2]); smax,smin=20.0,0.02
def D(x,s):
    l=np.log(q)-(x[:,None]-X)**2/(2*s*s); l-=l.max(1,keepdims=True); r=np.exp(l); r/=r.sum(1,keepdims=True); return r@X
x=smax*rng.standard_normal(20000)
ss=np.exp(np.linspace(np.log(smax),np.log(smin),4000))
for s0,s1 in zip(ss[:-1],ss[1:]):  # Heun on dx/ds=(x-D)/s
    d0=(x-D(x,s0))/s0; xe=x+(s1-s0)*d0; d1=(xe-D(xe,s1))/s1; x=x+(s1-s0)*(d0+d1)/2
lab=np.argmin(abs(x[:,None]-X),1); freq=np.bincount(lab,minlength=3)/len(x)
R=abs(X-(q@X)).max()  # bound uses centred witnesses? use |x_t| with N(0) init
bound=(q*abs(X)).sum()/(2*smax)+2*norm.sf(min(np.diff(np.sort(X)))/(2*smin))
rep("prior realisation", abs(freq-q).max()<=bound+3*np.sqrt(.25/len(x)), f"freq {freq.round(3)} q {q} bound {bound:.4f}")
# interval-noise boundary with sqrt weights halves L
rep("sqrt weights halve L", np.isclose(np.log(np.sqrt(.8)/np.sqrt(.2)),0.5*np.log(.8/.2)))
print("ALL PASS" if ok else "SOME FAIL")
