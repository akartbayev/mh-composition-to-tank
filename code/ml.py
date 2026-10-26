import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
R=8.314462618e-3  # kJ/mol/K
T0=298.15
df=pd.read_csv(ROOT+'data/mlhydpark_extracted.csv')
newmap={'V75Ti17.5Zr7.5':'SS','LaNi4.7Sn0.3':'AB5','V75Ti10Zr7.5Cr7.5':'SS','LaNi4.8Sn0.2':'AB5','LaNi4.8Al0.2':'AB5',
'V0.85Ti0.1Fe0.05':'SS','TiFe0.9Mn0.1':'AB','La0.85Ce0.15Ni5':'AB5','V92.5Zr7.5':'SS','La0.2Y0.8Ni4.6Mn0.4':'AB5',
'Zr0.7Ti0.3Mn2':'AB2','Ti0.9Zr0.1Mn1.4Cr0.35V0.2Fe0.05':'AB2','La0.4Ce0.4Ca0.2Ni5':'AB5','Ti0.8Zr0.2CrMn':'AB2',
'Zr0.8Ti0.2FeNi0.8V0.2':'AB2','Ti0.77Zr0.3Cr0.85Fe0.7Mn0.25Ni0.2Cu0.03':'AB2','TiCr1.9Mo0.01':'AB2','TiCr1.9':'AB2',
'ZrFe1.8Cr0.2':'AB2','(Ti0.97Zr0.03)1.1Cr1.6Mn0.4':'AB2','TiCr1.5Mn0.25Fe0.25':'AB2','TiCr1.5Mn0.2Fe0.3':'AB2','TiCrMn':'AB2'}
df['cls2']=[newmap.get(c,k) if k=='NEW' else k for c,k in zip(df.comp,df.cls)]
df['comp_obj']=[Composition(c) for c in df.comp]
# duplicates after normalisation
red=[c.fractional_composition.formula for c in df.comp_obj]
df['norm']=red
print('normalised duplicates:',df.norm.duplicated().sum())
fz=[ElementProperty.from_preset('magpie'),Stoichiometry(),ValenceOrbital()]
X=[]
for c in df.comp_obj:
    row=[]
    for f in fz: row+=list(f.featurize(c))
    X.append(row)
names=sum([f.feature_labels() for f in fz],[])
X=pd.DataFrame(X,columns=names).fillna(0).values
np.save(ROOT+'out/X.npy',X); json.dump(names,open(ROOT+'out/featnames.json','w'))
dH=df.dH.values; dS=df.dS.values/1000; lnP=df.lnP.values
def gbr(): return GradientBoostingRegressor(n_estimators=1500,learning_rate=0.005,max_depth=4,subsample=0.75,random_state=0)
def rf(): return RandomForestRegressor(n_estimators=500,min_samples_leaf=2,random_state=0,n_jobs=-1)
def lnp_at(H,S,T): return -H/(R*T)+S/R
res={}
oof={}
for mname,mk in [('GBR',gbr),('RF',rf)]:
  for rep in range(3):
    kf=KFold(5,shuffle=True,random_state=rep)
    pH=np.zeros(len(df)); pS=np.zeros(len(df)); pL=np.zeros(len(df))
    for tr,te in kf.split(X):
        for y,p in [(dH,pH),(dS,pS),(lnP,pL)]:
            m=mk().fit(X[tr],y[tr]); p[te]=m.predict(X[te])
    oof[(mname,rep)]=(pH.copy(),pS.copy(),pL.copy())
rows=[]
mask=(np.abs(lnP)<6)  # tank-relevant window ~0.0025-400 atm
for (mname,rep),(pH,pS,pL) in oof.items():
    for scheme in ['indep','vH']:
        H=pH
        if scheme=='indep': S=pS
        else: S=R*pL+pH/T0   # dS (positive magnitude convention) from lnP298 = -H/RT + S/R
        for T in [298.15,348.15]:
            e=lnp_at(H,S,T)-lnp_at(dH,dS,T)
            rows.append(dict(model=mname,rep=rep,scheme=scheme,T=T,
              mae_lnP=np.mean(np.abs(e)),mae_lnP_win=np.mean(np.abs(e[mask])),
              mae_H=np.mean(np.abs(H-dH)),mae_H_win=np.mean(np.abs(H-dH)[mask]),
              mae_S=np.mean(np.abs(S-dS))*1000,mae_S_win=np.mean(np.abs(S-dS)[mask])*1000,
              r2_H=1-np.sum((H-dH)**2)/np.sum((dH-dH.mean())**2)))
r=pd.DataFrame(rows).groupby(['model','scheme','T']).agg(['mean','std']).drop(columns='rep')
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(r.round(3))
r.to_csv(ROOT+'out/cv_metrics.csv')
print('window n=',mask.sum())
# save oof for GBR rep0 for parity + residuals
pH,pS,pL=oof[('GBR',0)]
o=df[['cls2','comp','dH','dS','lnP']].copy(); o['pH']=pH; o['pS']=pS*1000; o['pL']=pL
o['pS_vH']=(R*pL+pH/T0)*1000
o.to_csv(ROOT+'out/oof_gbr.csv',index=False)
df.drop(columns=['comp_obj']).to_csv(ROOT+'out/data_clean.csv',index=False)
