import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, json, sys, warnings, time
warnings.filterwarnings('ignore')
from scipy.optimize import least_squares
from multiprocessing import Pool
from tank2 import simulate_run, META
DD=ROOT+'tank_data/'; OUT=ROOT+'out/'
names=['lnP0','dH','beta','zeta','eta','lam','hbw','hout','lkm']
lo=np.array([-1.5,12,0.0,0.03,0.0,0.3,100,2,np.log(0.05)]); hi=np.array([4.0,45,2.0,0.4,1.2,5.0,3000,40,np.log(20)])
DATA={}
for F in [5,10,15]:
    a=pd.read_csv(DD+f'ds_abs_{F}.csv',index_col=0).dropna(subset=['Tm']); d=pd.read_csv(DD+f'ds_des_{F}.csv',index_col=0).dropna(subset=['Tm'])
    DATA[F]=(a,d)
def topar(z): p=dict(zip(names,z)); p['km']=float(np.exp(p.pop('lkm'))); return p
def blocks(z,F):
    a,d=DATA[F]; p=topar(z)
    ta=a.index.values.astype(float); td=d.index.values.astype(float)
    try: A,D=simulate_run(p,F,tabs=ta,tdes=td)
    except Exception: return None
    if len(A['t'])!=len(ta) or len(D['t'])!=len(td): return None
    gA=A['g']; band=np.clip(gA,a.g_lo.values,a.g_hi.values)
    sel=(ta%1800)<1  # every 30 min
    r={'abs_T':(A['Tw']-a.Tm.values)/2.0,
       'abs_P':((A['P']-(a.P.values+1.013))/0.7),
       'abs_g':((gA-band)/3.0)[sel],
       'des_T':((D['Tw']-d.Tm.values)/2.0)[td<6*3600],
       'des_P':((D['P']-(d.P.values+1.013))/0.5)[td<3*3600],
       'des_g':((D['g']-d.g.values)/2.0)[td<6*3600]}
    return r
def resid(z,runs=(10,)):
    out=[]
    for F in runs:
        r=blocks(z,F)
        if r is None: return np.full(sum(1 for _ in range(1)),1e3)*np.ones(2000)
        for k,v in r.items(): out.append(v*np.sqrt(30/len(v)))
    return np.concatenate(out)
def fitone(args):
    z0,runs=args
    try:
        s=least_squares(resid,z0,bounds=(lo,hi),args=(runs,),x_scale=(hi-lo)/10,diff_step=1e-3,max_nfev=400)
        return s.x.tolist(),float(s.cost),s.jac.tolist(),s.fun.tolist()
    except Exception as e:
        return None
if __name__=='__main__':
    runs=tuple(int(x) for x in sys.argv[1].split(','))
    rng=np.random.default_rng(3)
    starts=[np.array([2.0,24,0.8,0.12,0.35,1.5,700,6,0.0]),np.array([1.0,28,0.5,0.1,0.3,1.0,400,10,0.5]),
            np.array([2.5,20,1.0,0.15,0.4,2.5,1200,5,-0.5]),np.array([0.6,26.7,0.6,0.1,0.3,1.5,600,8,0.0])]
    t=time.time()
    with Pool(2) as pool: res=pool.map(fitone,[(z,runs) for z in starts])
    res=[r for r in res if r]; best=min(res,key=lambda r:r[1])
    for r in res: print('cost',round(r[1],2),np.round(r[0],3))
    tag='_'.join(map(str,runs))
    json.dump(dict(names=names,x=best[0],cost=best[1],jac=best[2],fun=best[3],all=[(r[0],r[1]) for r in res]),open(OUT+f'fit_ts_{tag}.json','w'))
    print('time',round(time.time()-t))
