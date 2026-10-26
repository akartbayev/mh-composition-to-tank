import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, json, warnings, sys
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from tank2 import design_run
from SALib.sample import saltelli
from SALib.analyze import sobol
OUT=ROOT+'out/'
names=['lnP0','dH','beta','zeta','eta','lam','hbw','hout','lkm']
def topar(z): p=dict(zip(names,z)); p['km']=float(np.exp(p.pop('lkm'))); return p
best=np.array(json.load(open(OUT+'fit_ts_10.json'))['x'])
ens=[]
for t in ['5','10','15']:
    f=json.load(open(OUT+f'fit_ts_{t}.json'))
    for x,c in f['all']:
        if c<=1.1*f['cost']: ens.append(x)
ens=np.array(ens)
K=['m_abs','m_des','Tmax','Tmin','t60','Tcmax','Tcmin']
def kpi(p):
    try: o=design_run(p); return [o[k] for k in K]
    except Exception: return [np.nan]*len(K)
stage=sys.argv[1]
if stage=='ens':
    print('ensemble size',len(ens)); print(pd.DataFrame(ens,columns=names).describe().round(3))
    with Pool(2) as pool: r=pool.map(kpi,[topar(z) for z in ens])
    d=pd.DataFrame(r,columns=K); print(d.describe().round(2)); 
    print('best',dict(zip(K,np.round(kpi(topar(best)),2))))
    pd.DataFrame(ens,columns=names).to_csv(OUT+'ts_ensemble.csv',index=False)
if stage=='scen':
    o=pd.read_csv(OUT+'oof_gbr.csv'); o=o[o.cls2=='AB2']; eH=(o.pH-o.dH).values; eL=(o.pL-o.lnP).values
    pr=pd.read_csv(OUT+'pred_targets.csv').iloc[0]; rng=np.random.default_rng(0)
    jobs=[];tags=[]
    for z in ens:
        for rep in range(9):
            base=topar(z)
            if rep==0:
                jobs.append(dict(base)); tags.append('A')
                for tg,(H,L) in {'B1':(24.6,3.170),'B2':(30.0,1.801)}.items():
                    p=dict(base); p['dH']=H; p['lnP0']=L; jobs.append(p); tags.append(tg)
                p=dict(base); p['dH']=pr.dH; p['lnP0']=pr.lnP; jobs.append(p); tags.append('C0')
            j=rng.integers(len(eH)); p=dict(base); p['dH']=float(np.clip(pr.dH-eH[j],10,60)); p['lnP0']=float(pr.lnP-eL[j]); jobs.append(p); tags.append('C')
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=4)
    df=pd.DataFrame(res,columns=K); df['scen']=tags; df.to_csv(OUT+'scen3.csv',index=False)
    A=df[df.scen=='A'].m_des.median()
    print(df.groupby('scen')[['m_abs','m_des','Tmax','Tmin','t60']].quantile([.05,.5,.95]).round(1))
    for c in ['B1','B2','C0','C']: print(c,'frac within 10% of A delivery',np.mean(np.abs(df[df.scen==c].m_des-A)<=0.1*A).round(3),'n',(df.scen==c).sum())
if stage=='window':
    base=topar(best); jobs=[];meta=[]
    for d in np.linspace(-3,3,25):
        p=dict(base); p['lnP0']+=d; jobs.append(p); meta.append(('lnP0',d,np.nan))
    for d in np.linspace(-10,10,21):
        p=dict(base); p['dH']+=d; jobs.append(p); meta.append(('dH',d,np.nan))
    for d1 in np.linspace(-3,3,17):
        for d2 in np.linspace(-10,10,11):
            p=dict(base); p['lnP0']+=d1; p['dH']+=d2; jobs.append(p); meta.append(('grid',d1,d2))
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=4)
    df=pd.DataFrame(res,columns=K); df[['var','d1','d2']]=pd.DataFrame(meta)
    df.to_csv(OUT+'window3.csv',index=False); print(df[df['var']!='grid'][['var','d1','m_abs','m_des','Tmin','t60']].round(2).to_string())
if stage=='sobol':
    N=int(sys.argv[2]); b=dict(zip(names,best))
    pb={'num_vars':8,'names':['lnP0','dH','beta','eta','lam','hout','lkm','hbw'],
        'bounds':[[b['lnP0']-1.75,b['lnP0']+1.75],[b['dH']-7,b['dH']+7],[0.0,1.0],[0.0,0.6],[0.5,5.0],[3,20],[np.log(0.05),np.log(20)],[100,3000]]}
    X=saltelli.sample(pb,N,calc_second_order=False)
    jobs=[]
    for x in X:
        z=dict(b); z.update(dict(zip(pb['names'],x))); jobs.append(topar([z[n] for n in names]))
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=16)
    Y=pd.DataFrame(res,columns=K); Y.to_csv(OUT+'sobol3_runs.csv',index=False); out={}
    for k in ['m_abs','m_des','Tmin','t60']:
        y=Y[k].values; y=np.where(np.isnan(y),np.nanmax(y) if k=='t60' else np.nanmean(y),y)
        S=sobol.analyze(pb,y,calc_second_order=False,seed=0)
        out[k]=dict(S1=S['S1'].tolist(),ST=S['ST'].tolist(),S1c=S['S1_conf'].tolist(),STc=S['ST_conf'].tolist())
        print(k,'ST',np.round(S['ST'],2),'conf',np.round(S['ST_conf'],2))
    json.dump(dict(problem=pb,res=out,nfail=int(Y.m_abs.isna().sum()),nt60nan=int(Y.t60.isna().sum())),open(OUT+'sobol3.json','w'),indent=1)
if stage=='screen':
    base=topar(best); pr=pd.read_csv(OUT+'pred_targets.csv'); jobs=[];meta=[]
    for _,r in pr.iterrows():
        isab5=r.comp.startswith('La')
        if not isab5 and not r.comp.startswith('Ti0.815'): continue
        for src in ['ML','DB']:
            if src=='ML': H,L=r.dH,r.lnP
            else:
                if not isinstance(r.in_data,str) or r.in_data in ('None','nan'): continue
                t=eval(r.in_data); H,L=t[0],t[2]
            p=dict(base); p['dH']=H; p['lnP0']=L
            if isab5: p.update(ms=10.8,rho_s=8.3e3,wmax=0.014,beta=0.25,eta=0.25,zeta=0.08,km=1.0)
            jobs.append(p); meta.append((r.comp,src,H,L))
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=2)
    df=pd.DataFrame(res,columns=K); df[['comp','src','dH','lnP0']]=pd.DataFrame(meta)
    df.to_csv(OUT+'screen3.csv',index=False); print(df[['comp','src','dH','lnP0','m_abs','m_des','Tmax','Tmin','t60']].round(2).to_string())
