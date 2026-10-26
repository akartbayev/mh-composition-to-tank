import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, warnings, json, sys
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from tank import cycle
names=['lnP0','dH','beta','zeta','eta','lam','hbw','hout','lkm','Psup','Pcut','Tamb']
d=pd.read_csv(ROOT+'out/abc_f10.csv')
print('runs',len(d),'failed',d.m_abs.isna().sum())
raw=pd.read_csv(ROOT+'out/abc_accepted_raw.csv')
obs=dict(m_abs=92.69,m_des=61.06,Tmax=75.0,Tmin=-18.0)
tol=dict(m_abs=0.10*92.69,m_des=0.10*61.06,Tmax=7.0,Tmin=7.0)
a=raw.copy(); a['w_abc']=a.w/a.w.sum(); acc=np.ones(len(a),bool)
print('accepted',len(a),'ESS_abc',1/np.sum(a.w_abc**2))
# ML prior from AB2 out-of-fold residuals
o=pd.read_csv(ROOT+'out/oof_gbr.csv'); o=o[o.cls2=='AB2']
eH=(o.pH-o.dH).values; eL=(o.pL-o.lnP).values
C=np.cov(np.vstack([eL,eH])); print('AB2 residual cov',C, 'corr',C[0,1]/np.sqrt(C[0,0]*C[1,1]))
pr=pd.read_csv(ROOT+'out/pred_targets.csv').iloc[0]
mu=np.array([pr.lnP,pr.dH]); Ci=np.linalg.inv(C)
z=a[['lnP0','dH']].values-mu
w=np.exp(-0.5*np.einsum('ij,jk,ik->i',z,Ci,z))*a.w_abc.values; w/=w.sum(); a['w_ml']=w
print('ESS',1/np.sum(w**2))
def q(x,ww=None,p=(5,50,95)):
    if ww is None: return np.percentile(x,p)
    i=np.argsort(x); c=np.cumsum(ww[i]); return np.interp(np.array(p)/100,c,x[i])
summ={}
for k in names:
    summ[k]=dict(prior=[float(d[k].min()),float(d[k].max())],post=list(map(float,q(a[k].values,a.w_abc.values))),post_ml=list(map(float,q(a[k].values,a.w_ml.values))))
    print(k,np.round(summ[k]['post'],3),np.round(summ[k]['post_ml'],3))
json.dump(dict(n_acc=int(len(a)),ess_abc=float(1/np.sum(a.w_abc**2)),summ=summ,ml_mu=mu.tolist(),ml_cov=C.tolist(),ess=float(1/np.sum(w**2))),open(ROOT+'out/posterior.json','w'),indent=1)
a.to_csv(ROOT+'out/abc_accepted.csv',index=False)
