import numpy as np, json, sys
src=open('vc_explore.py').read(); ns={}
exec(src[:src.index('# =============== E1')].replace("VEC=sys.argv[1]","VEC='/tmp/kvc/vct/wv/en_core_web_md/en_core_web_md-3.8.0/vocab/vectors'"),ns)
E=ns['E']; simplex_code=ns['simplex_code']; softmax=ns['softmax']; N,dE=E.shape; Kd=4096; dk=64; tau=8.0
out={}
for seed in [101,102,103]:
    g=np.random.default_rng(seed)
    WK=g.normal(size=(dk,dE))/np.sqrt(dE); WV=g.normal(size=(dk,dE))/np.sqrt(dE)
    Kall=E@WK.T; Kall=Kall/np.linalg.norm(Kall,axis=1,keepdims=True)*np.sqrt(dk); Vall=E@WV.T; Vn=Vall/np.linalg.norm(Vall,axis=1,keepdims=True)
    Ka=Kall[:Kd]; dec=lambda o:int(np.argmax(Vn@o))
    for T in [256,1024]:
        beta=2.0; kin=g.choice(Kd,T,replace=False); kout=g.choice(np.arange(Kd,N),T,replace=False); vals=g.choice(N,T,replace=False)
        r={}
        r['coordinate_out']=np.mean([dec(softmax(beta*Kall[kout]@Kall[kout[i]]/tau)@Vall[vals])==vals[i] for i in range(T)])
        def agg(codes,keys):
            m=np.zeros(Kd); S=np.zeros((Kd,dk))
            for (sup,a),v in zip(codes,vals): m[sup]+=a; S[sup]+=np.outer(a,Vall[v])
            occ=m>0; h=0
            for i in range(T):
                l=beta*Ka[occ]@Kall[keys[i]]/tau; w=np.exp(l-l.max())*m[occ]; h+=dec(w@S[occ]/w.sum())==vals[i]
            return h/T
        r['aggregate_in_onehot']=agg([(np.array([j]),np.ones(1)) for j in kin],kin)
        for s in [1,4,16]:
            codes=[simplex_code(Ka,Kall[j],s) for j in kout]
            r[f'aggregate_out_s{s}']=agg(codes,kout)
            r[f'resid_s{s}']=float(np.mean([np.linalg.norm(Kall[j]-a@Ka[sp])/np.linalg.norm(Kall[j]) for j,(sp,a) in zip(kout,codes)]))
            r[f'support_s{s}']=float(np.mean([len(sp) for sp,a in codes]))
            r[f'distinct_atoms_s{s}']=int(len(set(int(x) for sp,a in codes for x in sp)))
            if s==1: r['hard_collisions']=int(T-len(set(int(sp[0]) for sp,a in codes)))
        out.setdefault(str(T),[]).append({k:float(v) for k,v in r.items()}); print(seed,T,r,flush=True)
json.dump(out,open('e3_seeds.json','w'),indent=1)
