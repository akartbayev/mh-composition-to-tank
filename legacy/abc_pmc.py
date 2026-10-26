import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, sys, time, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from abc_run import names, lo, hi, job
OUT=ROOT+'out/'
obs=dict(m_abs=92.69,m_des=61.06,Tmax=75.0,Tmin=-18.0); sc=dict(m_abs=0.10*92.69,m_des=0.10*61.06,Tmax=7.0,Tmin=7.0)
K=['m_abs','m_des','Tmax','Tmin','Tcmax','Tcmin','t_abs','t_des']
def dist(df): return np.sqrt(sum(((df[k]-obs[k])/sc[k])**2 for k in obs))
def boxok(df): return np.all([(df[k]-obs[k]).abs()<=sc[k] for k in obs],axis=0)
rng=np.random.default_rng(7)
d0=pd.read_csv(OUT+'abc_f10.csv'); d0['D']=dist(d0)
rounds=[(2.5,int(sys.argv[1])),(None,int(sys.argv[2]))]   # eps for next population; None -> final box criterion
pop=d0.nsmallest(120,'D'); popw=np.ones(len(pop))/len(pop)
allr=[]
for r,(eps,N) in enumerate(rounds):
    Z=(pop[names].values-lo)/(hi-lo)
    cov=2*np.cov(Z.T,aweights=popw)+1e-6*np.eye(len(names))
    L=np.linalg.cholesky(cov)
    cand=[]
    while len(cand)<N:
        i=rng.choice(len(Z),p=popw); z=Z[i]+L@rng.standard_normal(len(names))
        if np.all((z>=0)&(z<=1)): cand.append(z)
    cand=np.array(cand); P=lo+cand*(hi-lo)
    t=time.time()
    with Pool(2) as pool: res=pool.map(job,list(P),chunksize=10)
    df=pd.DataFrame(P,columns=names); df[K]=np.array(res); df['D']=dist(df); df['round']=r+1
    # importance weight (uniform prior / mixture proposal density, up to const)
    Ci=np.linalg.inv(cov)
    dz=cand[:,None,:]-Z[None,:,:]; m=np.einsum('nkd,de,nke->nk',dz,Ci,dz)
    q=(np.exp(-0.5*m)*popw[None,:]).sum(1); df['w']=1/q
    df.to_csv(OUT+f'abc_pmc_r{r+1}.csv',index=False); allr.append(df)
    print('round',r+1,'N',N,'time',round(time.time()-t),'box-accepted',int(boxok(df).sum()),'D<2.5',int((df.D<2.5).sum()),flush=True)
    if eps is not None:
        sel=df[df.D<eps]
        if len(sel)<60: sel=df.nsmallest(60,'D')
        pop=sel; popw=(sel.w/sel.w.sum()).values
final=allr[-1]; acc=final[boxok(final)].copy(); acc['w']=acc.w/acc.w.sum()
acc.to_csv(OUT+'abc_accepted_raw.csv',index=False); print('final accepted',len(acc),'ESS',1/np.sum(acc.w**2))
