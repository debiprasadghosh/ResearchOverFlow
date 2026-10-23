import numpy as np, json
rng=np.random.default_rng(20261022)
sm=lambda x: np.exp(x-x.max())/np.exp(x-x.max()).sum()
def proj_simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1; i=np.arange(1,len(v)+1)
    rho=np.nonzero(u-c/i>0)[0][-1]; th=c[rho]/(rho+1); return np.maximum(v-th,0)
def code(k,D,s,iters=300):
    d2=((D-k)**2).sum(1); S=np.argsort(d2)[:s]; A=D[S]
    if s==1: return S,np.ones(1)
    a=np.ones(s)/s; L=np.linalg.norm(A@A.T,2)+1e-12
    for _ in range(iters): a=proj_simplex(a-(A@(A.T@a)-A@k)/L)
    return S,a
R={}
# 1 identity + lazy ALiBi for coefficient memory
err=0
for tr in range(200):
    K,r,dv,T=13,6,5,80; D=rng.normal(size=(K,r)); V=rng.normal(size=(T,dv))
    Al=np.zeros((T,K))
    for i in range(T):
        S=rng.choice(K,3,replace=False); a=rng.dirichlet(np.ones(3)); Al[i,S]=a
    m=rng.choice([0,0.05,0.5]); q=rng.normal(size=r)
    # direct kernel attention with age decay at time T-1
    ages=(T-1)-np.arange(T); w=(Al@np.exp(D@q))*np.exp(-m*ages); o1=w@V/w.sum()
    # lazy per-bin state
    g=np.full(K,-np.inf); mu=np.zeros((K,dv)); tau=np.zeros(K)
    for i in range(T):
        for j in np.nonzero(Al[i])[0]:
            a0=g[j]-m*(i-tau[j]); gn=np.logaddexp(a0,np.log(Al[i,j]))
            mu[j]+=np.exp(np.log(Al[i,j])-gn)*(V[i]-mu[j]); g[j]=gn; tau[j]=i
    occ=np.isfinite(g); sc=D[occ]@q+g[occ]-m*((T-1)-tau[occ]); p=sm(sc); o2=p@mu[occ]
    err=max(err,np.abs(o1-o2).max())
R['lazy_identity_maxerr']=err
# 2 Jensen/Hoeffding interval for logit perturbation
viol=0; n=0
for tr in range(3000):
    r=16;K=64;D=rng.normal(size=(K,r));k=rng.normal(size=r);q=rng.normal(size=r)*rng.uniform(0.05,1)
    S,a=code(k,D,rng.integers(1,10)); z=D[S]@q
    delta=np.log(a@np.exp(z))-q@k; lin=q@(a@D[S]-k); spread=z.max()-z.min()
    lo,hi=lin,lin+spread**2/8; n+=1
    if delta<lo-1e-10 or delta>hi+1e-10: viol+=1
R['hoeffding_trials']=n; R['hoeffding_violations']=viol
# 3 signed coefficients can make the denominator nonpositive
D=np.array([[1.0],[-1.0]]); q=np.array([3.0]); alpha=np.array([-1.0,1.5])  # codes k=-2.5? still
R['signed_Z']=float(alpha@np.exp(D@q))
# 4 approximation study: keys on 4-dim subspace-curved manifold in r=16
def run(K,s,intr=4,T=2000,Q=200,scale=1.0,atoms='gauss'):
    r=16; B=rng.normal(size=(intr,r))/np.sqrt(intr)
    Z=rng.normal(size=(T,intr)); keys=np.tanh(Z)@B*2
    if atoms=='gauss': D=rng.normal(size=(K,r))*keys.std(0)+keys.mean(0)
    else: D=keys[rng.choice(T,K,replace=False)]+0.01*rng.normal(size=(K,r))
    V=rng.normal(size=(T,8)); coded=[code(k,D,s) for k in keys]
    Al=np.zeros((T,K))
    for i,(S,a) in enumerate(coded): Al[i,S]=a
    N=Al.sum(0); Sv=Al.T@V
    es=[];res=np.mean([np.linalg.norm(keys[i]-a@D[S]) for i,(S,a) in enumerate(coded)])
    for _ in range(Q):
        q=keys[rng.integers(T)]*scale
        o=sm(keys@q)@V; e=np.exp(D@q-(D@q).max())*0; g=D@q; occ=N>0
        p=np.exp(g[occ]-g[occ].max())*N[occ]; oh=(p@ (Sv[occ]/N[occ,None]))/p.sum()
        es.append(np.linalg.norm(o-oh)/np.linalg.norm(V.max(0)-V.min(0)))
    return dict(K=K,s=s,atoms=atoms,scale=scale,mean_rel_err=float(np.mean(es)),p95=float(np.percentile(es,95)),mean_residual=float(res),occupied=int((N>0).sum()))
study=[]
for atoms in ['gauss','sampled']:
  for K in [256,1024]:
    for s in [1,2,4,8,17]:
      for sc in [0.25,1.0]:
        study.append(run(K,s,scale=sc,atoms=atoms))
R['study']=study
# storage (bytes) d=512,H=8,r=16,dv=64,K=4096,W=256
d,H,r,dv,K,W,T=512,8,16,64,4096,256,131072
MiB=2**20
R['storage']={
 'counter_window (paper)': (K*(4*H*dv+8+4)+2*W*(r+H*dv))/MiB,
 'coef_per_head_means_fp32mass_fp64': (K*(4*H*dv+8)+2*W*(r+H*dv))/MiB,
 'coef_input_space_d': (K*(4*d+8)+2*W*(r+H*dv))/MiB,
 'coef_value_dict_L128': (K*(4*128*H+8)+2*W*(r+H*dv))/MiB,
 'coef_input_dict_L128': (K*(4*128+8)+2*W*(r+H*dv))/MiB,
 'exact_shared_slice': 2*T*(r+H*dv)/MiB}
json.dump(R,open('results.json','w'),indent=1); print(json.dumps(R,indent=1))
