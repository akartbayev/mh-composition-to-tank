import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, json, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.7,'savefig.dpi':300})
OUT=ROOT+'out/'; FIG=ROOT+'fig/'; DD=ROOT+'tank_data/'
from fit_ts import topar, names
from tank2 import simulate_run
which=sys.argv[1]
best=np.array(json.load(open(OUT+'fit_ts_10.json'))['x']); p=topar(best)
files={5:('5_Absorb_5_SLPM','5_Desorb_5_SLPM'),10:('10_4_Absorb_10_SLPM','10_4_desorb_5_SLPM'),15:('15_Absorb_15_SLPM','15_Desorb_5_SLPM')}
if which=='fit':
    fig,ax=plt.subplots(4,3,figsize=(7.2,8.2))
    for j,F in enumerate([5,10,15]):
        A0=pd.read_csv(DD+files[F][0]+'.csv'); D0=pd.read_csv(DD+files[F][1]+'.csv')
        def rs(X,step):
            X=X.copy(); X['tmin']=X[['TC1','TC2','TC3','TC4','TC5']].min(axis=1); X['tmax']=X[['TC1','TC2','TC3','TC4','TC5']].max(axis=1); X['Tm']=X[['TC1','TC2','TC3','TC4','TC5']].mean(axis=1)
            X['b']=(X.t//step)*step; return X.groupby('b')[['tmin','tmax','Tm','P']].mean()
        a=pd.read_csv(DD+f'ds_abs_{F}.csv',index_col=0); d=pd.read_csv(DD+f'ds_des_{F}.csv',index_col=0)
        ra=rs(A0,300); rd=rs(D0[D0.t<7*3600],120)
        A,D=simulate_run(p,F,tabs=np.linspace(0,a.index.max(),500),tdes=np.linspace(0,7*3600,400))
        h=ra.index/3600; ax[0,j].fill_between(h,ra.tmin,ra.tmax,color='0.8',lw=0); ax[0,j].plot(h,ra.Tm,'k',lw=.8); ax[0,j].plot(A['t']/3600,A['Tw'],'C3',lw=1.1)
        ax[1,j].fill_between(a.index/3600,a.g_lo,a.g_hi,color='0.75',lw=0); ax[1,j].plot(a.index/3600,a.g_lo,'k',lw=.8); ax[1,j].plot(A['t']/3600,A['g'],'C3',lw=1.1)
        h=rd.index/3600; ax[2,j].fill_between(h,rd.tmin,rd.tmax,color='0.8',lw=0); ax[2,j].plot(h,rd.Tm,'k',lw=.8); ax[2,j].plot(D['t']/3600,D['Tw'],'C3',lw=1.1)
        ax[3,j].plot(d.index/3600,d.g,'k',lw=.9); ax[3,j].plot(D['t']/3600,D['g'],'C3',lw=1.1)
        a3=ax[3,j].twinx(); a3.plot(d.index/3600,d.P+1.013,'k:',lw=.8); a3.plot(D['t']/3600,D['P'],'C3',ls=':',lw=.9); a3.set_ylim(0,27)
        if j==2: a3.set_ylabel('Pressure (bar, dotted)')
        else: a3.set_yticklabels([])
        ax[0,j].set_title(f'{F} SLPM charge'+(' (calibration)' if F==10 else ' (validation)'),fontsize=8)
        for i in range(4): ax[i,j].set_xlim(0,None if i<2 else 7)
    ax[0,0].set_ylabel('Shell T, absorption (°C)'); ax[1,0].set_ylabel('H$_2$ absorbed (g)'); ax[2,0].set_ylabel('Shell T, desorption (°C)'); ax[3,0].set_ylabel('H$_2$ delivered (g)')
    for j in range(3): ax[1,j].set_xlabel('Time (h)'); ax[3,j].set_xlabel('Time (h)')
    from matplotlib.lines import Line2D; from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([],[],color='k',lw=.8,label='measured (TC mean / flow)'),Patch(color='0.78',label='TC range or flow-dropout band'),Line2D([],[],color='C3',lw=1.1,label='model (10 SLPM calibration)')],loc='upper center',ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(0.5,1.0))
    plt.tight_layout(rect=(0,0,1,0.97)); plt.savefig(FIG+'fig3_ts_fit.png')
if which=='plateau':
    prof=json.load(open(OUT+'profile_lnP0.json')); post=json.load(open(OUT+'posterior.json'))
    mu=np.array(post['ml_mu']); C=np.array(post['ml_cov'])
    ens=pd.read_csv(OUT+'ts_ensemble.csv')
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    v=np.array([r[0] for r in prof]); c=np.array([r[1] for r in prof])
    ax[0].plot(v,c-c.min(),'k.-',lw=.8); ax[0].set_yscale('symlog',linthresh=10); ax[0].set_xlabel(r'ln($P_{eq,0}$/atm), fixed'); ax[0].set_ylabel('Δ cost (profiled)')
    for t,col in zip(['5','10','15'],['C0','C3','C2']):
        f=json.load(open(OUT+f'fit_ts_{t}.json')); ax[0].axvline(f['x'][0],color=col,lw=.8,ls='--',label=f'{t} SLPM fit')
    ax[0].axvspan(mu[0]-np.sqrt(C[0,0]),mu[0]+np.sqrt(C[0,0]),color='C1',alpha=.12,label='GBR ±1σ (AB$_2$)'); ax[0].axvline(mu[0],color='C1',lw=1.2,label='GBR prediction')
    ax[0].legend(fontsize=6,frameon=True,framealpha=.9,loc='lower left'); ax[0].set_xlim(0,4)
    def ell(axx,mu,C,n,**kw):
        w,V=np.linalg.eigh(C); ang=np.degrees(np.arctan2(V[1,1],V[0,1])); axx.add_patch(Ellipse(mu,2*n*np.sqrt(w[1]),2*n*np.sqrt(w[0]),angle=ang,fill=False,**kw))
    ax[1].scatter(ens.lnP0,ens.dH,s=14,c='C3',label='time-series fits (3 runs × starts)',zorder=3)
    ax[1].plot(*mu,'k*',ms=9,label='GBR prediction'); ell(ax[1],mu,C,1,ec='k',lw=.8); ell(ax[1],mu,C,2,ec='k',lw=.8,ls='--')
    for lab,(H,L),off in [('Ti$_{0.8}$Zr$_{0.2}$CrMn',(24.6,3.17),(4,-9)),('Zr$_{0.2}$Ti$_{0.8}$MnCr',(30.0,1.80),(-72,4))]:
        ax[1].plot(L,H,'s',color='0.4',ms=5); ax[1].annotate(lab,(L,H),fontsize=6,xytext=off,textcoords='offset points')
    ax[1].set_xlim(-2.5,5.5); ax[1].set_ylim(10,46); ax[1].set_xlabel(r'ln($P_{eq,0}$/atm) at 25 °C'); ax[1].set_ylabel(r'$\Delta H$ (kJ mol$^{-1}$)')
    ax[1].legend(fontsize=6,frameon=True,framealpha=.9,loc='upper right',markerscale=.8)
    for x_,l in zip(ax,'ab'): x_.text(-0.2,1.03,f'({l})',transform=x_.transAxes,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG+'fig4_plateau.png')
