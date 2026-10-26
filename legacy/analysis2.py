import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, json, warnings, sys
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from tank import cycle
from SALib.sample import saltelli
from SALib.analyze import sobol
OUT=ROOT+'out/'
post=json.load(open(OUT+'posterior.json')); a=pd.read_csv(OUT+'abc_accepted.csv')
names=['lnP0','dH','beta','zeta','eta','lam','hbw','hout','lkm','Psup','Pcut','Tamb']
def topar(row):
    p={k:float(row[k]) for k in names}; p['km']=float(np.exp(p.pop('lkm'))); return p
def kpi(args):
    p,F=args
    try:
        o=cycle(p,F); return [o[k] for k in ['m_abs','m_des','Tmax','Tmin','Tcmax','Tcmin','t_abs','t_des']]
    except Exception: return [np.nan]*8
K=['m_abs','m_des','Tmax','Tmin','Tcmax','Tcmin','t_abs','t_des']
stage=sys.argv[1]
med={k:post['summ'][k]['post'][1] for k in names}
if stage=='ppc':
    # posterior predictive at 5 and 15 SLPM (and 10 for reference), up to 150 samples
    s=a.copy()
    jobs=[(topar(r),F) for F in [5,10,15] for _,r in s.iterrows()]
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=5)
    df=pd.DataFrame(res,columns=K); df['F']=[F for F in [5,10,15] for _ in range(len(s))]
    df.to_csv(OUT+'ppc.csv',index=False); print(df.groupby('F').quantile([.05,.5,.95]).round(1))
if stage=='scen':
    # transport/operating params from posterior samples; thermodynamics per scenario
    s=a.sample(100,replace=True,weights='w_abc',random_state=1)
    o=pd.read_csv(OUT+'oof_gbr.csv'); o=o[o.cls2=='AB2']; eH=(o.pH-o.dH).values; eL=(o.pL-o.lnP).values
    pr=pd.read_csv(OUT+'pred_targets.csv').iloc[0]
    rng=np.random.default_rng(0)
    scen={}
    jobs=[];tags=[]
    for i,(_,r) in enumerate(s.iterrows()):
        base=topar(r)
        # A: calibrated (sample itself)
        jobs.append((dict(base),10)); tags.append('A')
        # B1/B2: nearest database analogues
        for tg,(H,L) in {'B1':(24.6,3.170),'B2':(30.0,1.801)}.items():
            p=dict(base); p['dH']=H; p['lnP0']=L; jobs.append((p,10)); tags.append(tg)
        # C0: ML point prediction; C: ML + sampled CV residual pair (prediction - truth => truth = pred - e)
        p=dict(base); p['dH']=pr.dH; p['lnP0']=pr.lnP; jobs.append((p,10)); tags.append('C0')
        j=rng.integers(len(eH)); p=dict(base); p['dH']=float(np.clip(pr.dH-eH[j],12,60)); p['lnP0']=float(pr.lnP-eL[j]); jobs.append((p,10)); tags.append('C')
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=5)
    df=pd.DataFrame(res,columns=K); df['scen']=tags; df.to_csv(OUT+'scen.csv',index=False)
    print(df.groupby('scen').quantile([.05,.5,.95]).round(1))
if stage=='window':
    # one-at-a-time sweeps around posterior median: lnP0 offsets and dH offsets
    base=dict(med); base['km']=float(np.exp(base.pop('lkm')))
    jobs=[];meta=[]
    for d in np.linspace(-3,3,25):
        p=dict(base); p['lnP0']=base['lnP0']+d; jobs.append((p,10)); meta.append(('lnP0',d))
    for d in np.linspace(-10,10,21):
        p=dict(base); p['dH']=base['dH']+d; jobs.append((p,10)); meta.append(('dH',d))
    for d1 in np.linspace(-3,3,17):
        for d2 in np.linspace(-10,10,11):
            p=dict(base); p['lnP0']=base['lnP0']+d1; p['dH']=base['dH']+d2; jobs.append((p,10)); meta.append(('grid',(d1,d2)))
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=3)
    df=pd.DataFrame(res,columns=K); df['var']=[m[0] for m in meta]; df['delta']=[m[1] if m[0]!='grid' else np.nan for m in meta]; df['g1']=[m[1][0] if m[0]=='grid' else np.nan for m in meta]; df['g2']=[m[1][1] if m[0]=='grid' else np.nan for m in meta]
    df.to_csv(OUT+'window.csv',index=False); print(df[df['var']!='grid'][['var','delta','m_abs','m_des','Tmin','t_abs']].round(2).to_string())
if stage=='sobol':
    # thermodynamics varied within +-ML MAE (AB2) around posterior median; transport within engineering ranges
    N=int(sys.argv[2])
    pb={'num_vars':8,'names':['lnP0','dH','beta','eta','lam','hout','lkm','hbw'],
        'bounds':[[med['lnP0']-1.75,med['lnP0']+1.75],[med['dH']-7,med['dH']+7],[0.2,1.2],[0.1,0.7],[0.5,3.0],[5,40],[np.log(0.2),np.log(5)],[150,1500]]}
    X=saltelli.sample(pb,N,calc_second_order=False)
    jobs=[]
    for x in X:
        p=dict(med); p.update(dict(zip(pb['names'],x))); p['km']=float(np.exp(p.pop('lkm'))); jobs.append((p,10))
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=20)
    Y=pd.DataFrame(res,columns=K); Y.to_csv(OUT+'sobol_runs.csv',index=False)
    out={}
    for k in ['m_abs','m_des','Tmin','t_abs']:
        y=Y[k].values; y=np.where(np.isnan(y),np.nanmean(y),y)
        S=sobol.analyze(pb,y,calc_second_order=False,seed=0)
        out[k]=dict(S1=S['S1'].tolist(),ST=S['ST'].tolist(),S1c=S['S1_conf'].tolist(),STc=S['ST_conf'].tolist())
        print(k,'S1',np.round(S['S1'],2),'ST',np.round(S['ST'],2))
    json.dump(dict(problem=pb,res=out,nfail=int(Y.m_abs.isna().sum())),open(OUT+'sobol.json','w'),indent=1)
if stage=='screen':
    base=dict(med); base['km']=float(np.exp(base.pop('lkm')))
    pr=pd.read_csv(OUT+'pred_targets.csv')
    jobs=[];meta=[]
    for _,r in pr.iterrows():
        isab5=r.comp.startswith('La')
        if not isab5 and not r.comp.startswith('Ti0.815'): continue
        for src in ['ML','DB']:
            if src=='ML': H,L=r.dH,r.lnP
            else:
                if not isinstance(r.in_data,str) or r.in_data=='None' or r.in_data=='nan': continue
                t=eval(r.in_data); H,L=t[0],t[2]
            p=dict(base); p['dH']=H; p['lnP0']=L
            if isab5:
                p.update(ms=10.8,rho_s=8.3e3,wmax=0.014,beta=0.25,eta=0.25,zeta=0.08,km=1.0)
            jobs.append((p,10)); meta.append((r.comp,src,H,L))
    with Pool(2) as pool: res=pool.map(kpi,jobs,chunksize=2)
    df=pd.DataFrame(res,columns=K); df[['comp','src','dH','lnP0']]=pd.DataFrame(meta)
    df.to_csv(OUT+'screen.csv',index=False); print(df[['comp','src','dH','lnP0','m_abs','m_des','Tmax','Tmin','t_abs','t_des']].round(2).to_string())
