import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, json, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fit_ts import DATA, topar, names
from tank2 import simulate_run, META
OUT=ROOT+'out/'
tag=sys.argv[1] if len(sys.argv)>1 else '10'
fit=json.load(open(OUT+f'fit_ts_{tag}.json')); z=np.array(fit['x']); p=topar(z)
rows=[]
fig,ax=plt.subplots(3,4,figsize=(14,9))
for i,F in enumerate([5,10,15]):
    a,d=DATA[F]; ta=a.index.values.astype(float); td=d.index.values.astype(float)
    A,D=simulate_run(p,F,tabs=ta,tdes=td)
    rmse=lambda x,y: float(np.sqrt(np.nanmean((x-y)**2)))
    w6=td<6*3600; w3=td<3*3600
    rows.append(dict(F=F,role='calibration' if str(F) in tag.split('_') else 'validation',
        abs_T=rmse(A['Tw'],a.Tm.values),abs_P=rmse(A['P'],a.P.values+1.013),
        des_T=rmse(D['Tw'][w6],d.Tm.values[w6]),des_P=rmse(D['P'][w3],d.P.values[w3]+1.013),des_g=rmse(D['g'][w6],d.g.values[w6]),
        g_abs_model=float(A['g'][-1]),g_abs_lo=META[F]['g_abs_lo'],g_abs_hi=META[F]['g_abs_hi'],
        g_des_model=float(D['g'].max()),g_des_obs=META[F]['g_des'],
        Tmax_model=float(A['Tw'].max()),Tmax_obs=float(a.Tm.max()),Tmin_model=float(D['Tw'][w6].min()),Tmin_obs=float(d.Tm.min()),
        Pdes0_model=float(D['P'][0]),Pdes0_obs=META[F]['P_des0']))
    h=ta/3600
    ax[i,0].plot(h,a.Tm,'k',lw=.8); ax[i,0].plot(h,A['Tw'],'r',lw=1); ax[i,0].set_title(f'{F} abs T')
    ax[i,1].fill_between(h,a.g_lo,a.g_hi,color='0.8'); ax[i,1].plot(h,A['g'],'r'); ax[i,1].set_title('abs g')
    hd=td/3600
    ax[i,2].plot(hd,d.Tm,'k',lw=.8); ax[i,2].plot(hd,D['Tw'],'r'); ax[i,2].set_title('des T'); ax[i,2].set_xlim(0,7)
    ax[i,3].plot(hd,d.g,'k'); ax[i,3].plot(hd,D['g'],'r'); a3=ax[i,3].twinx(); a3.plot(hd,d.P+1.013,'k:',lw=.7); a3.plot(hd,D['P'],'r:',lw=.7); ax[i,3].set_title('des g / P')
plt.tight_layout(); plt.savefig(OUT+f'val_ts_{tag}.png',dpi=70)
r=pd.DataFrame(rows); pd.set_option('display.width',250); print(r.round(2).to_string()); r.to_csv(OUT+f'val_ts_{tag}.csv',index=False)
print(dict(zip(names,np.round(z,3))))
