import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, sys, warnings, time
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from scipy.stats import qmc
from tank import cycle
names=['lnP0','dH','beta','zeta','eta','lam','hbw','hout','lkm','Psup','Pcut','Tamb']
lo=np.array([-1.5,18,0.2,0.05,0.1,0.5,150,5,np.log(0.2),10,1.1,295])
hi=np.array([4.0,40,1.2,0.20,0.7,3.0,1500,40,np.log(5),50,2.5,303])
F=10.0
def job(row):
    p=dict(zip(names,row)); p['km']=np.exp(p.pop('lkm'))
    try:
        o=cycle(p,F); return [o[k] for k in ['m_abs','m_des','Tmax','Tmin','Tcmax','Tcmin','t_abs','t_des']]
    except Exception as e:
        return [np.nan]*8
if False:
    t=time.time()
    with Pool(2) as pool: res=pool.map(job,list(P),chunksize=20)
    df=pd.DataFrame(P,columns=names); df[['m_abs','m_des','Tmax','Tmin','Tcmax','Tcmin','t_abs','t_des']]=np.array(res)
    df.to_csv(fROOT+'out/abc_{tag}.csv',index=False); print('done',time.time()-t)
