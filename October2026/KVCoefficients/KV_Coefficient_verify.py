import numpy as np, json
from scipy.spatial import Delaunay
rng=np.random.default_rng(20261022); R={}
def read(logits,V): l=logits-logits.max(); w=np.exp(l); return w@V/w.sum()
# 1 worked example
tau=1.0; q=np.log(2); keys=np.array([.25,.75]); vals=np.array([[2.],[-1.]]); A=np.array([0.,1.])
al=np.array([[.75,.25],[.25,.75]]); kh=al@np.exp(q*A/tau); k=np.exp(q*keys/tau)
y=read(q*keys/tau,vals)[0]; yh=(kh@vals[:,0])/kh.sum(); D=np.log(kh/k)
DV=3.; E=np.abs(kh-k).sum(); Z=k.sum(); eps=E/Z
R['example']=dict(y=y,yhat=yh,err=abs(y-yh),Delta=D.tolist(),tanh_bound=DV*np.tanh((D.max()-D.min())/4),
  thm6_bound=2*2*eps/(1-eps),thm6_cheb=2*1.5*eps/(1-eps),thm6_onesided=2*1.5*eps)
# 2 identity + bounds, exact barycentres
viol=dict(identity=0.0,interval=0,tanh=0,thm6=0); ratios=[]
for t in range(5000):
    d=rng.integers(1,9); M=rng.integers(d+1,40); s=rng.integers(1,min(M,d+1)+1); n=rng.integers(1,60); p=rng.integers(1,6)
    At=rng.normal(size=(M,d)); qq=rng.normal(size=d)*rng.uniform(.05,3); tau=np.sqrt(d)
    Al=np.zeros((n,M)); 
    for i in range(n):
        S=rng.choice(M,s,replace=False); Al[i,S]=rng.dirichlet(np.ones(s))
    K=Al@At; Vv=rng.normal(size=(n,p)); w=rng.uniform(.1,2,size=n)
    ka=np.exp(At@qq/tau); c=(w[:,None]*Al).sum(0); Sm=(w[:,None]*Al).T@Vv
    yh=(ka@Sm)/(ka@c); khat=Al@ka; Dl=np.log(khat)-K@qq/tau
    y=read(K@qq/tau+np.log(w),Vv); yh2=read(np.log(khat)+np.log(w),Vv)
    viol['identity']=max(viol['identity'],np.abs(yh-yh2).max())
    dl=(At@qq/tau)[None,:]-(K@qq/tau)[:,None]
    h=np.where(Al>0,np.abs(dl),0).max(1); sig=(Al*dl**2).sum(1)
    rng_=np.array([ (dl[i][Al[i]>0].max()-dl[i][Al[i]>0].min()) for i in range(n)])
    ub=np.minimum(np.log1p(np.exp(h)*sig/2),rng_**2/8)
    if (Dl<-1e-10).any() or (Dl>ub+1e-10).any(): viol['interval']+=1
    DVv=max(np.linalg.norm(Vv[i]-Vv[j]) for i in range(n) for j in range(n)) if n>1 else 0
    err=np.linalg.norm(y-yh); tb=DVv*np.tanh((Dl.max()-Dl.min())/4)
    if err>tb+1e-10: viol['tanh']+=1
    Vn=np.linalg.norm(Vv,axis=1).max(); kk=np.exp(K@qq/tau); E=(w*np.abs(khat-kk)).sum(); Z=(w*kk).sum(); e=E/Z
    t6=2*Vn*e/(1-e) if e<1 else 2*Vn
    if err>t6+1e-10: viol['thm6']+=1
    if e<1 and err>1e-9: ratios.append(t6/max(tb,1e-300))
R['trials']=5000; R['violations']=viol; R['thm6_over_tanh_median']=float(np.median(ratios)); R['thm6_over_tanh_p10']=float(np.percentile(ratios,10))
# 3 inexact barycentre interval
v3=0
for t in range(3000):
    d=16;M=64;At=rng.normal(size=(M,d));k=rng.normal(size=d);qq=rng.normal(size=d)*rng.uniform(.05,1)
    S=np.argsort(((At-k)**2).sum(1))[:rng.integers(1,10)]; a=rng.dirichlet(np.ones(len(S)))
    z=At[S]@qq/4; Dl=np.log(a@np.exp(z))-qq@k/4; lin=qq@(a@At[S]-k)/4; rr=z.max()-z.min()
    if Dl<lin-1e-10 or Dl>lin+rr**2/8+1e-10: v3+=1
R['inexact_interval_violations']=v3
# 4 rates: 1-D and 2-D, nearest vs barycentric, fixed query ball
def rate(dim,Ms,Q=2.0,n=4000,nq=200):
    out=[]
    X=rng.uniform(0,1,size=(n,dim)); Vv=rng.normal(size=(n,4))
    Qs=rng.normal(size=(nq,dim)); Qs=Q*Qs/np.linalg.norm(Qs,axis=1,keepdims=True)
    for m in Ms:
        g=np.linspace(0,1,m); At=np.array(np.meshgrid(*[g]*dim)).reshape(dim,-1).T
        near=np.argmin(((X[:,None,:]-At[None])**2).sum(2),1)
        if dim==1:
            j=np.clip(np.searchsorted(g,X[:,0])-1,0,m-2); tt=(X[:,0]-g[j])/(g[j+1]-g[j])
            idx=np.stack([j,j+1],1); bc=np.stack([1-tt,tt],1)
        else:
            tri=Delaunay(At); sx=tri.find_simplex(X); T=tri.transform[sx]
            b=np.einsum('nij,nj->ni',T[:,:dim],X-T[:,dim]); bc=np.c_[b,1-b.sum(1)]; idx=tri.simplices[sx]
        en=[];eb=[];ln=[];lb=[]
        for qq in Qs:
            l=X@qq; y=read(l,Vv); ka=np.exp(At@qq)
            ln_=At[near]@qq; yn=read(ln_,Vv)
            lb_=np.log((bc*ka[idx]).sum(1)); yb=read(lb_,Vv)
            en.append(np.linalg.norm(y-yn)); eb.append(np.linalg.norm(y-yb))
            ln.append(np.abs(ln_-l).max()); lb.append(np.abs(lb_-l).max())
        out.append(dict(atoms=int(len(At)),spacing=float(1/(m-1)),nearest_out=float(np.mean(en)),bary_out=float(np.mean(eb)),nearest_logit=float(np.mean(ln)),bary_logit=float(np.mean(lb))))
    base=list(out)
    for key in['nearest_out','bary_out','nearest_logit','bary_logit']:
        hs=np.log([o['spacing'] for o in base]); es=np.log([o[key] for o in base])
        out.append({key+'_slope':float(np.polyfit(hs,es,1)[0])})
    return out
R['rate_1d']=rate(1,[5,9,17,33,65])
R['rate_2d']=rate(2,[4,6,9,13,17],n=3000,nq=100)
# 5 crossover in Q for 1-D, spacing 1/8
cross=[]
for Q in [1,2,4,8,16,32,64]:
    r=rate(1,[9],Q=Q,nq=100)[0]; cross.append(dict(Q=Q,h=Q/8,nearest=r['nearest_out'],bary=r['bary_out']))
R['crossover_1d_spacing_0.125']=cross
json.dump(R,open('results2.json','w'),indent=1,default=float); print(json.dumps(R,indent=1,default=float))
