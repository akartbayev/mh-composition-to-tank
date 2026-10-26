import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from sklearn.ensemble import GradientBoostingRegressor
R=8.314462618e-3; T0=298.15
df=pd.read_csv(ROOT+'out/data_clean.csv'); X=np.load(ROOT+'out/X.npy')
dH=df.dH.values; lnP=df.lnP.values; dS=df.dS.values/1000
def gbr(seed=0): return GradientBoostingRegressor(n_estimators=1500,learning_rate=0.005,max_depth=4,subsample=0.75,random_state=seed)
o=pd.read_csv(ROOT+'out/oof_gbr.csv')
# per class OOF metrics (vH scheme)
o['eH']=o.pH-o.dH; o['eL']=o.pL-o.lnP
print(o.groupby('cls2').agg(n=('eH','size'),maeH=('eH',lambda e:np.abs(e).mean()),maeL=('eL',lambda e:np.abs(e).mean())).round(2))
win=np.abs(o.lnP)<6
print('corr(eH,eL) window',np.corrcoef(o.eH[win],o.eL[win])[0,1], 'std eH',o.eH[win].std(),'std eL',o.eL[win].std())
# class-out
rows=[]
for cl in ['AB2','AB5']:
    tr=df.cls2.values!=cl; te=~tr
    mH=gbr().fit(X[tr],dH[tr]); mL=gbr().fit(X[tr],lnP[tr])
    eH=mH.predict(X[te])-dH[te]; eL=mL.predict(X[te])-lnP[te]
    w=np.abs(lnP[te])<6
    rows.append(dict(held=cl,n=te.sum(),maeH=np.abs(eH).mean(),maeL=np.abs(eL).mean(),maeL_win=np.abs(eL[w]).mean(),biasL=eL.mean()))
    o.loc[te,'pH_co']=mH.predict(X[te]); o.loc[te,'pL_co']=mL.predict(X[te])
co=pd.DataFrame(rows); print(co.round(2)); co.to_csv(ROOT+'out/classout.csv',index=False)
o.to_csv(ROOT+'out/oof_gbr.csv',index=False)
# final models: ensemble of 5 seeds on all data
fz=[ElementProperty.from_preset('magpie'),Stoichiometry(),ValenceOrbital()]
def feat(c):
    c=Composition(c); r=[]
    for f in fz: r+=list(f.featurize(c))
    return np.nan_to_num(np.array(r,dtype=float))
targets=['Ti0.815Zr0.185Cr0.77Fe0.07Mn0.88','LaNi5','La0.9Ce0.1Ni5','La0.8Ce0.2Ni5','La0.75Ce0.25Ni5','La0.5Ce0.5Ni5',
 'LaNi4.9Al0.1','LaNi4.8Al0.2','LaNi4.7Al0.3','LaNi4.6Al0.4','LaNi4.45Al0.55','LaNi4.25Al0.75',
 'Ti0.8Zr0.2CrMn','TiFe0.9Mn0.1','Ti0.9Zr0.1Mn1.5Cr0.3V0.2']
XT=np.array([feat(t) for t in targets])
PH=[];PL=[]
for s in range(5):
    PH.append(gbr(s).fit(X,dH).predict(XT)); PL.append(gbr(s).fit(X,lnP).predict(XT))
PH=np.array(PH); PL=np.array(PL)
pred=pd.DataFrame(dict(comp=targets,dH=PH.mean(0),dH_sd=PH.std(0),lnP=PL.mean(0),lnP_sd=PL.std(0)))
pred['dS']=(R*pred.lnP+pred.dH/T0)*1000
pred['Peq25_bar']=np.exp(pred.lnP)*1.01325
# known values in dataset if present
kn=dict(zip([Composition(c).fractional_composition.formula for c in df.comp],zip(df.dH,df.dS,df.lnP)))
pred['in_data']=[kn.get(Composition(t).fractional_composition.formula) for t in targets]
print(pred.round(3).to_string()); pred.to_csv(ROOT+'out/pred_targets.csv',index=False)
# feature importance for lnP model
m=gbr().fit(X,lnP); names=json.load(open(ROOT+'out/featnames.json'))
imp=pd.Series(m.feature_importances_,index=names).sort_values(ascending=False)[:10]; print(imp.round(3)); imp.to_csv(ROOT+'out/imp_lnP.csv')
m=gbr().fit(X,dH); imp=pd.Series(m.feature_importances_,index=names).sort_values(ascending=False)[:10]; print(imp.round(3)); imp.to_csv(ROOT+'out/imp_dH.csv')
