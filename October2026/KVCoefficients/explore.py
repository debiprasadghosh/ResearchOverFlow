import numpy as np, json, time
from scipy.spatial import Delaunay
SEEDS=[20261022,20261023,20261024]; d=16; M=4096; n=128; NK=96; NQ=96; p=8; tau=4.0; MU=0.1; BETAS=[0.25,1.0,4.0]
def lse(x,ax=-1):
    m=x.max(ax,keepdims=True); return (m+np.log(np.exp(x-m).sum(ax,keepdims=True))).squeeze(ax)
def nearest_code(A,An,k,Gm=None):
    d2=An-2*A@k+k@k; j=np.argmin(d2); return np.array([j]),np.ones(1)
def local16(A,An,k,steps=16,mu=MU):
    d2=An-2*A@k+k@k; P=np.argpartition(d2,n)[:n]; P=P[np.argsort(d2[P])]; Ap=A[P]; h=d2[P]
    al=np.zeros(n); al[0]=1; r=Ap[0]-k
    for _ in range(steps):
        g=2*Ap@r+mu*h; u=np.argmin(g); act=np.nonzero(al>0)[0]; v=act[np.argmax(g[act])]
        G=g[v]-g[u]
        if G<=1e-14: break
        pv=Ap[u]-Ap[v]; pp=pv@pv
        t=al[v] if pp==0 else min(al[v],G/(2*pp))
        al[u]+=t; al[v]-=t; r=r+t*pv
    m=al>1e-12; return P[m],al[m]/al[m].sum()
def code_all(A,keys,coder,T=None):
    if T is not None: At=A@T.T; Kt=keys@T.T
    else: At,Kt=A,keys
    An=(At**2).sum(1); return [coder(At,An,k) for k in Kt]
def surrogate_logits(A,codes,q,beta):
    return np.array([lse(np.log(a)+beta*(A[C]@q)/tau) for C,a in codes])
def probs(l): w=np.exp(l-l.max()); return w/w.sum()
def metrics(lt,ls,V,rngV):
    P=probs(lt); Ph=probs(ls); o=P@V; oh=Ph@V
    return 100*np.linalg.norm(o-oh)/rngV, 0.5*np.abs(P-Ph).sum(), float(np.argmax(P)==np.argmax(Ph))
def evaluate(A,keys,codes,Q,V,decomp=False):
    rngV=np.linalg.norm(V.max(0)-V.min(0)); out={}
    khat=np.array([a@A[C] for C,a in codes])
    for b in BETAS:
        acc={'full':[], 'resid_only':[], 'jensen_only':[]}
        for q in Q:
            lt=b*keys@q/tau; ls=surrogate_logits(A,codes,q,b)
            acc['full'].append(metrics(lt,ls,V,rngV))
            if decomp:
                lr=b*khat@q/tau; J=ls-lr
                acc['resid_only'].append(metrics(lt,lr,V,rngV)); acc['jensen_only'].append(metrics(lt,lt+J,V,rngV))
        out[b]={kk:np.mean(vv,0).tolist() for kk,vv in acc.items() if vv}
    return out
def stats(A,keys,codes):
    res=np.mean([np.linalg.norm(k-a@A[C]) for k,(C,a) in zip(keys,codes)])
    sup=np.mean([len(C) for C,a in codes]); return res,sup
R={}
# ---------- A, D, E: decomposition, expansion, TV/top1 ----------
rowsA={}
for scale in [0.5,1.0]:
  for expand in [1.0,1.3]:
    for cname,coder in [('nearest',nearest_code),('local16',local16)]:
      accs=[];res=[]
      for sd in SEEDS:
        g=np.random.default_rng(sd); A=g.normal(size=(M,d)); keys=g.normal(size=(NK,d))*scale; Q=g.normal(size=(NQ,d)); V=g.normal(size=(NK,p))
        if expand!=1.0: A=A.mean(0)+expand*(A-A.mean(0))
        codes=code_all(A,keys,coder); accs.append(evaluate(A,keys,codes,Q,V,decomp=(cname=='local16' and expand==1.0))); res.append(stats(A,keys,codes))
      key=f"scale{scale}_x{expand}_{cname}"
      rowsA[key]={'residual':float(np.mean([r[0] for r in res])),'support':float(np.mean([r[1] for r in res]))}
      for b in BETAS:
        for kk in accs[0][b]:
          arr=np.array([a[b][kk] for a in accs]); rowsA[key][f"b{b}_{kk}"]={'err%':float(arr[:,0].mean()),'err_sd':float(arr[:,0].std(ddof=1)),'TV':float(arr[:,1].mean()),'top1':float(arr[:,2].mean())}
R['A_decomp_expand']=rowsA; print('A done')
# ---------- B: query-adapted metric (Conjecture 2) ----------
rowsB={}
for cname in ['nearest_E','local16_E','nearest_H','local16_H']:
  accs=[];res=[]
  for sd in SEEDS:
    g=np.random.default_rng(sd+100); A=g.normal(size=(M,d)); keys=g.normal(size=(NK,d))*0.5; V=g.normal(size=(NK,p))
    U,_=np.linalg.qr(g.normal(size=(d,d))); lam=np.logspace(0,-2,d)  # 100:1 variance anisotropy
    L=U*np.sqrt(lam); Qc=g.normal(size=(2048,d))@L.T; Q=g.normal(size=(NQ,d))@L.T; Q=Q/np.sqrt(np.mean((Q**2).sum(1)))*np.sqrt(d)
    Qc=Qc/np.sqrt(np.mean((Qc**2).sum(1)))*np.sqrt(d)
    H=Qc.T@Qc/len(Qc)+1e-6*np.eye(d); w,E=np.linalg.eigh(H); T=(E*np.sqrt(w))@E.T
    coder=nearest_code if cname.startswith('nearest') else local16
    codes=code_all(A,keys,coder,T if cname.endswith('H') else None)
    accs.append(evaluate(A,keys,codes,Q,V)); res.append(stats(A,keys,codes))
    resH=np.mean([np.sqrt((k-a@A[C])@H@(k-a@A[C])) for k,(C,a) in zip(keys,codes)])
    res[-1]=res[-1]+(resH,)
  rowsB[cname]={'residual':float(np.mean([r[0] for r in res])),'H_residual':float(np.mean([r[2] for r in res])),'support':float(np.mean([r[1] for r in res]))}
  for b in BETAS:
    arr=np.array([a[b]['full'] for a in accs]); rowsB[cname][f"b{b}"]={'err%':float(arr[:,0].mean()),'err_sd':float(arr[:,0].std(ddof=1)),'TV':float(arr[:,1].mean())}
R['B_query_metric']=rowsB; print('B done')
# ---------- C: certificate-driven selective controller (Conjecture 3) ----------
rowsC={}
for scale in [0.5,1.0]:
  out={c:{b:[] for b in BETAS} for c in ['nearest','local16','selective']}; frac={b:[] for b in BETAS}
  for sd in SEEDS:
    g=np.random.default_rng(sd); A=g.normal(size=(M,d)); keys=g.normal(size=(NK,d))*scale; Q=g.normal(size=(NQ,d)); V=g.normal(size=(NK,p))
    Qcal=np.random.default_rng(sd+7).normal(size=(1024,d)); qbound=np.percentile(np.linalg.norm(Qcal,axis=1),95)
    cn=code_all(A,keys,nearest_code); cs=code_all(A,keys,local16)
    for b in BETAS:
      Qb=b*qbound; sel=[];
      for k,(Cn,an),(Cs,as_) in zip(keys,cn,cs):
        rh=np.linalg.norm(k-A[Cn[0]]); rs=np.linalg.norm(k-as_@A[Cs])
        Ad=A[Cs]; diam=np.sqrt(((Ad[:,None]-Ad[None])**2).sum(-1).max()) if len(Cs)>1 else 0
        ws=2*Qb*rs/tau+(Qb*diam/tau)**2/8; wh=2*Qb*rh/tau
        sel.append((Cs,as_) if ws<wh else (Cn,an))
      frac[b].append(np.mean([len(c[0])>1 for c in sel]))
      for name,codes in [('nearest',cn),('local16',cs),('selective',sel)]:
        rngV=np.linalg.norm(V.max(0)-V.min(0))
        out[name][b].append(np.mean([metrics(b*keys@q/tau,surrogate_logits(A,codes,q,b),V,rngV) for q in Q],0))
  rowsC[f"scale{scale}"]={name:{f"b{b}":{'err%':float(np.mean([x[0] for x in out[name][b]])),'TV':float(np.mean([x[1] for x in out[name][b]]))} for b in BETAS} for name in out}
  rowsC[f"scale{scale}"]['soft_fraction']={f"b{b}":float(np.mean(frac[b])) for b in BETAS}
R['C_selective']=rowsC; print('C done')
# ---------- F: matched-budget rates (Conjecture 1) ----------
def rate(dim,ms,Q=2.0,nk=3000,nq=100,seed=5):
    g=np.random.default_rng(seed); X=g.uniform(0,1,size=(nk,dim)); V=g.normal(size=(nk,4))
    Qs=g.normal(size=(nq,dim)); Qs=Q*Qs/np.linalg.norm(Qs,axis=1,keepdims=True); rows=[]
    for m in ms:
        gr=np.linspace(0,1,m); At=np.array(np.meshgrid(*[gr]*dim)).reshape(dim,-1).T
        near=np.argmin(((X[:,None,:]-At[None])**2).sum(2),1)
        tri=Delaunay(At) if dim>1 else None
        if dim==1:
            j=np.clip(np.searchsorted(gr,X[:,0])-1,0,m-2); tt=(X[:,0]-gr[j])/(gr[j+1]-gr[j]); idx=np.stack([j,j+1],1); bc=np.stack([1-tt,tt],1)
        else:
            sx=tri.find_simplex(X); T=tri.transform[sx]; b=np.einsum('nij,nj->ni',T[:,:dim],X-T[:,dim]); bc=np.c_[b,1-b.sum(1)]; idx=tri.simplices[sx]
        en=[];eb=[];tn=[];tb=[]
        for q in Qs:
            l=X@q; P=probs(l); ka=At@q
            Pn=probs(ka[near]); Pb=probs(np.log((bc*np.exp(ka[idx])).sum(1)))
            en.append(np.linalg.norm(P@V-Pn@V)); eb.append(np.linalg.norm(P@V-Pb@V)); tn.append(.5*np.abs(P-Pn).sum()); tb.append(.5*np.abs(P-Pb).sum())
        rows.append(dict(M=int(len(At)),h=1/(m-1),near=float(np.mean(en)),bary=float(np.mean(eb)),TVnear=float(np.mean(tn)),TVbary=float(np.mean(tb))))
    lM=np.log([r['M'] for r in rows])
    sl={k:float(np.polyfit(lM,np.log([r[k] for r in rows]),1)[0]) for k in['near','bary','TVnear','TVbary']}
    return dict(rows=rows,slope_vs_M=sl,pred_near=-1/dim,pred_bary=-2/dim)
R['F_rates']={'1d':rate(1,[5,9,17,33,65,129]),'2d':rate(2,[4,6,9,13,17,25],nk=2500),'3d':rate(3,[3,4,5,7,9,11],nk=2000,nq=60)}
print('F done')
# ---------- G: FP32 / FP64 / Kahan mass accumulation ----------
g=np.random.default_rng(9); N=30_000_000
inc=g.uniform(0,1,size=N)  # fractional coefficient written to one atom each step
ex=np.cumsum(inc.astype(np.float64))  # float64 reference (accurate to ~1e-9 rel)
f32=np.cumsum(inc.astype(np.float32),dtype=np.float32)
checkpoints=[10**5,10**6,4*10**6,10**7,2*10**7,N]
G={}
for c in checkpoints:
    G[str(c)]={'fp32_rel_err':float(abs(f32[c-1]-ex[c-1])/ex[c-1])}
# Kahan in float32 on a 3e6 slice via chunked python is slow; do vectorized pairwise reference instead
G['note']='fp32 cumsum is sequential; increments U(0,1)'
# expected tail: once mass > 2^24*0.5, increments below half ulp vanish
R['G_fp32']=G; print('G done')
json.dump(R,open('explore_results.json','w'),indent=1)
