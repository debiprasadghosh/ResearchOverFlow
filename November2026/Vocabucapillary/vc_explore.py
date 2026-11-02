import numpy as np, json, time, sys
from scipy.optimize import nnls
VEC=sys.argv[1]
E0=np.load(VEC).astype(np.float64)
_,first=np.unique(E0,axis=0,return_index=True); E=E0[np.sort(first)]   # unique rows, first-occurrence order
N,dE=E.shape
R={}
def softmax(l): l=l-l.max(); w=np.exp(l); return w/w.sum()
# ---------------- coders ----------------
def simplex_code(D,x,s,pool=None,rho=None):
    """greedy positive simplex coder: Frank-Wolfe pricing over the whole dictionary, then exact
    simplex least squares on the support by augmented NNLS"""
    rho=rho or 10*np.sqrt((x**2).sum()+1)
    d2=((D-x)**2).sum(1); sup=[int(np.argmin(d2))]; a=np.ones(1); banned=set()
    for _ in range(60):
        if len(sup)>=s: break
        m=a@D[sup]; g=D@(m-x)
        g[sup]=np.inf
        for b in banned: g[b]=np.inf
        j=int(np.argmin(g))
        if g[j]>=m@(m-x)-1e-12: break
        trial=sup+[j]; A=np.vstack([D[trial].T, rho*np.ones(len(trial))]); b=np.r_[x,rho]
        at,_=nnls(A,b); at=np.maximum(at,0); at/=at.sum()
        keep=at>1e-12
        if not keep[-1]: banned.add(j); continue
        sup=[t for t,k in zip(trial,keep) if k]; a=at[keep]
    return np.array(sup),a
def cone_code(D,x,s,pool=None):
    sup=[]; a=np.zeros(0); r=x.copy(); banned=set()
    for _ in range(60):
        if len(sup)>=s: break
        g=D@r; g[sup]=-np.inf
        for b in banned: g[b]=-np.inf
        j=int(np.argmax(g))
        if g[j]<=1e-12: break
        trial=sup+[j]; at,_=nnls(D[trial].T,x); keep=at>1e-12
        if not keep[-1]: banned.add(j); continue
        sup=[t for t,k in zip(trial,keep) if k]; a=at[keep]; r=x-a@D[sup]
    return np.array(sup),a
# =============== E1: representation geometry with real vocabulary embeddings ===============
rng=np.random.default_rng(20261102)
Kd=4096; D=E[:Kd]                       # dictionary: 4096 most frequent rows
targets={}
outside=rng.choice(np.arange(Kd,N),60,replace=False)
targets['word outside dictionary']=E[outside]
for k in [2,4,8]:
    T=[];
    for _ in range(60):
        idx=rng.choice(Kd,k,replace=False); w=rng.dirichlet(np.ones(k)); T.append(w@D[idx])
    targets[f'mixture of {k} dictionary words']=np.array(T)
mu=D.mean(0); Cv=np.cov(D.T); L=np.linalg.cholesky(Cv+1e-6*np.eye(dE))
targets['Gaussian with dictionary covariance']=mu+rng.normal(size=(60,dE))@L.T
S_LIST=[1,2,4,8,16,32,64]
E1={}
for name,X in targets.items():
    E1[name]={}
    for kind,coder in [('simplex',simplex_code),('cone',cone_code)]:
        row=[]
        for s in S_LIST:
            res=[]
            for x in X:
                sup,a=coder(D,x,s); res.append(np.linalg.norm(x-a@D[sup])/np.linalg.norm(x))
            row.append(float(np.mean(res)))
        E1[name][kind]=row
    print('E1',name,E1[name],flush=True)
R['E1_representation']={'s':S_LIST,'rows':E1,'K':Kd,'d':dE}
# =============== E2: identities ===============
viol={}
g=np.random.default_rng(1)
mx={'thm6':0,'thm7':0,'cor61':0,'prop12':0,'lemma10_tv':0}; l10_viol=0; prop8=None
for t in range(500):
    Nn=g.integers(3,12); dk=g.integers(2,6); p=g.integers(1,4); T=g.integers(1,30); s=g.integers(1,Nn+1)
    Dk=g.normal(size=(Nn,dk)); B=g.normal(size=(p,Nn)); q=g.normal(size=dk); tau=np.sqrt(dk)
    Cc=np.zeros((T,Nn))
    for i in range(T):
        sup=g.choice(Nn,s,replace=False); Cc[i,sup]=g.dirichlet(np.ones(s))
    w=g.uniform(.1,2,T); V=g.normal(size=(T,p)); phi=np.exp(Dk@q/tau)
    kap=Cc@phi; o_tok=(w*kap)@V/(w*kap).sum()
    m=w@Cc; S=(w[:,None]*Cc).T@V; o_agg=S.T@phi/(m@phi); mx['thm6']=max(mx['thm6'],np.abs(o_tok-o_agg).max())
    Vs=Cc@B.T; Cm=(w[:,None]*Cc).T@Cc; o7=B@Cm@phi/(m@phi); o7t=(w*kap)@Vs/(w*kap).sum(); mx['thm7']=max(mx['thm7'],np.abs(o7-o7t).max(),np.abs(Cm.sum(1)-m).max())
    # Cor 6.1: one-hot tokens, ordinary attention == count read
    xs=g.integers(0,Nn,T); Wv=g.normal(size=(p,Nn)); lo=Dk[xs]@q/tau; o_ord=softmax(lo)@Wv[:,xs].T
    cnt=np.bincount(xs,minlength=Nn).astype(float); o_cnt=(cnt*phi)@Wv.T/(cnt@phi); mx['cor61']=max(mx['cor61'],np.abs(o_ord-o_cnt).max())
    # Prop 12 decay recurrence
    lam=g.uniform(.5,1); mt=np.zeros(Nn); St=np.zeros((Nn,p))
    for i in range(T): mt=lam*mt+Cc[i]; St=lam*St+np.outer(Cc[i],V[i])
    ww=lam**(T-1-np.arange(T)); o_d=((ww*kap)@V)/(ww*kap).sum(); mx['prop12']=max(mx['prop12'],np.abs(St.T@phi/(mt@phi)-o_d).max())
    # Lemma 10
    c=Cc[-1]; order=np.argsort(-c); keep=order[:max(1,g.integers(1,s+1))]; ct=np.zeros(Nn); ct[keep]=c[keep]; eta=1-ct.sum(); ct/=ct.sum()
    mx['lemma10_tv']=max(mx['lemma10_tv'],abs(0.5*np.abs(c-ct).sum()-eta))
    Dst=max(np.linalg.norm(Dk[i]-Dk[j]) for i in range(Nn) for j in range(Nn))
    if np.linalg.norm(c@Dk-ct@Dk)>eta*Dst+1e-12: l10_viol+=1
R['E2_identities']={'trials':500,'max_abs_discrepancy':{k:float(v) for k,v in mx.items()},'lemma10_bound_violations':l10_viol}
print('E2',R['E2_identities'],flush=True)
# =============== E3: associative recall through the aggregate state, real embeddings ===============
dk=64; tau=np.sqrt(dk); g=np.random.default_rng(7)
WK=g.normal(size=(dk,dE))/np.sqrt(dE); WV=g.normal(size=(dk,dE))/np.sqrt(dE)
Kall=E@WK.T; Kall=Kall/np.linalg.norm(Kall,axis=1,keepdims=True)*np.sqrt(dk)  # unit-RMS keys
Vall=E@WV.T; Vn=Vall/np.linalg.norm(Vall,axis=1,keepdims=True)
Katoms=Kall[:Kd]
def decode(o): return int(np.argmax(Vn@o))
E3={}
for T in [16,64,256,1024]:
  for beta in [0.5,1.0,2.0]:
    res={}
    keys_in=g.choice(Kd,T,replace=False); keys_out=g.choice(np.arange(Kd,N),T,replace=False); vals=g.choice(N,T,replace=False)
    # (a) coordinate baseline: exact ordinary attention over retained records (outside keys)
    acc=[]
    for cond,keys in [('in',keys_in),('out',keys_out)]:
        Kk=Kall[keys]; Vv=Vall[vals]
        acc=np.mean([decode(softmax(beta*Kk@Kall[keys[i]]/tau)@Vv)==vals[i] for i in range(T)])
        res[f'coordinate_{cond}']=float(acc)
    # (b) aggregate, keys inside dictionary: one-hot codes, exact kernel
    def agg(codes,keys):
        m=np.zeros(Kd); S=np.zeros((Kd,dk))
        for (sup,a),v in zip(codes,vals): m[sup]+=a; S[sup]+=np.outer(a,Vall[v])
        occ=m>0; hits=0
        for i in range(T):
            l=beta*Katoms[occ]@Kall[keys[i]]/tau; w=np.exp(l-l.max())*m[occ]; o=w@S[occ]/w.sum()
            hits+=decode(o)==vals[i]
        return hits/T
    res['aggregate_in_onehot']=agg([(np.array([j]),np.ones(1)) for j in keys_in],keys_in)
    for s in [1,4,16]:
        codes=[simplex_code(Katoms,Kall[j],s,pool=128) for j in keys_out]
        res[f'aggregate_out_s{s}']=agg(codes,keys_out)
    E3[f'T{T}_b{beta}']=res; print('E3',T,beta,res,flush=True)
R['E3_recall']=E3
# =============== E4: support growth across depth and pruning profiles ===============
g=np.random.default_rng(11); Tlen=64; L=6
z=1/np.arange(1,Kd+1); z/=z.sum()
Wq=[g.normal(size=(dk,dE))/np.sqrt(dE) for _ in range(L)]; Wk=[g.normal(size=(dk,dE))/np.sqrt(dE) for _ in range(L)]
profiles={'uncapped':[None]*L,'32,16,64,16,4,1 (positional)':[32,16,64,16,4,1],'1,4,16,64,16,4':[4,16,64,16,4,1],'flat 16':[16]*L,'flat 4':[4]*L}
E4={}
for pname,caps in profiles.items():
  stats=[]
  for rep in range(4):
    xs=g.choice(Kd,Tlen,p=z); Cm=np.zeros((Tlen,Kd)); Cm[np.arange(Tlen),xs]=1; Cu=Cm.copy()
    per=[]
    for l in range(L):
        H=Cm@D; q=H@Wq[l].T; k=H@Wk[l].T; Sc=q@k.T/np.sqrt(dk); Sc[np.triu_indices(Tlen,1)]=-np.inf
        P=np.exp(Sc-Sc.max(1,keepdims=True)); P/=P.sum(1,keepdims=True)
        new=0.5*Cm+0.5*P@Cm
        Hu=Cu@D; qu=Hu@Wq[l].T; ku=Hu@Wk[l].T; Su=qu@ku.T/np.sqrt(dk); Su[np.triu_indices(Tlen,1)]=-np.inf
        Pu=np.exp(Su-Su.max(1,keepdims=True)); Pu/=Pu.sum(1,keepdims=True); Cu=0.5*Cu+0.5*Pu@Cu
        cap=caps[l]; eta=np.zeros(Tlen)
        if cap is not None:
            for t in range(Tlen):
                nz=np.nonzero(new[t])[0]
                if len(nz)>cap:
                    keep=nz[np.argsort(-new[t,nz])[:cap]]; kept=new[t,keep].sum(); eta[t]=1-kept
                    row=np.zeros(Kd); row[keep]=new[t,keep]/kept; new[t]=row
        Cm=new
        sup=(Cm>0).sum(1); H_=-(np.where(Cm>0,Cm*np.log(np.where(Cm>0,Cm,1)),0)).sum(1)
        err=np.linalg.norm(Cm@D-Cu@D,axis=1)/np.linalg.norm(Cu@D,axis=1)
        per.append([sup.mean(),np.exp(H_).mean(),eta.mean(),err.mean()])
    stats.append(per)
  st=np.mean(stats,0); E4[pname]={'active':st[:,0].tolist(),'effective':st[:,1].tolist(),'discarded':st[:,2].tolist(),'rel_hidden_err_vs_uncapped':st[:,3].tolist()}
  print('E4',pname,np.round(st,3).tolist(),flush=True)
R['E4_depth']=E4
# =============== E5: storage and second-moment density ===============
E5={}
for s in [1,4,16]:
    nnz=[]; Cset=set(); mset=set(); xs=g.choice(Kd,4096,p=z)
    for t,x in enumerate(xs):
        sup,a=simplex_code(D,D[x]*0.5+D[g.choice(Kd,p=z)]*0.5,s,pool=128) if s>1 else (np.array([x]),np.ones(1))
        mset.update(sup.tolist()); Cset.update((int(i),int(j)) for i in sup for j in sup)
        if t+1 in (64,256,1024,4096): nnz.append((t+1,len(mset),len(Cset)))
    E5[f's{s}']=nnz; print('E5',s,nnz,flush=True)
R['E5_density']=E5
json.dump(R,open('vc_results.json','w'),indent=1)
