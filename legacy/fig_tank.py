import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, json, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.linewidth':0.8,'savefig.dpi':300})
OUT=ROOT+'out/'; FIG=ROOT+'fig/'
post=json.load(open(OUT+'posterior.json')); d=pd.read_csv(OUT+'abc_f10.csv'); a=pd.read_csv(OUT+'abc_accepted.csv')
mu=np.array(post['ml_mu']); C=np.array(post['ml_cov'])
which=sys.argv[1] if len(sys.argv)>1 else 'all'
def ellipse(ax,mu,C,n,**kw):
    v,w=np.linalg.eigh(C); ang=np.degrees(np.arctan2(w[1,1],w[0,1]))
    ax.add_patch(Ellipse(mu,2*n*np.sqrt(v[1]),2*n*np.sqrt(v[0]),angle=ang,fill=False,**kw))
def wq(x,w,p):
    i=np.argsort(x); c=np.cumsum(w[i]); return np.interp(p,c/c[-1],x[i])
if which in ('all','abc'):
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.5))
    ax[0].scatter(d.lnP0,d.dH,s=1,c='#cccccc',lw=0,label='prior samples')
    r2=pd.read_csv(OUT+'abc_pmc_r2.csv'); ax[0].scatter(r2.lnP0,r2.dH,s=1,c='#f4b6b6',lw=0,label='PMC proposals')
    ax[0].scatter(a.lnP0,a.dH,s=7,c='#d62728',lw=0,label='accepted')
    ax[0].plot(*mu,'k*',ms=8,label='GBR prediction')
    # ML prior uses (lnP, dH) ordering
    ellipse(ax[0],mu,C,1,ec='k',lw=.8,ls='-'); ellipse(ax[0],mu,C,2,ec='k',lw=.8,ls='--')
    ax[0].set_xlabel(r'ln($P_{eq,0}$/atm) at 25 °C'); ax[0].set_ylabel(r'$\Delta H$ (kJ mol$^{-1}$)')
    ax[0].set_xlim(-2.5,5.5); ax[0].set_ylim(12,46); ax[0].legend(fontsize=5.5,frameon=True,framealpha=.9,loc='upper right',markerscale=1.0,handletextpad=0.2,borderpad=0.3)
    bins=np.linspace(-1.5,4,30)
    ax[1].hist(a.lnP0,bins=bins,weights=a.w_abc,density=True,color='#d62728',alpha=.45,label='tank only')
    ax[1].hist(a.lnP0,bins=bins,weights=a.w_ml,density=True,histtype='step',color='k',lw=1.2,label='tank + GBR prior')
    xx=np.linspace(-1.5,4,200); s=np.sqrt(C[0,0]); ax[1].plot(xx,np.exp(-0.5*((xx-mu[0])/s)**2)/(s*np.sqrt(2*np.pi)),'k:',lw=1,label='GBR prior')
    ax[1].set_xlabel(r'ln($P_{eq,0}$/atm)'); ax[1].set_ylabel('density'); ax[1].legend(fontsize=6,frameon=False)
    bins=np.linspace(15,45,31)
    ax[2].hist(a.dH,bins=bins,weights=a.w_abc,density=True,color='#d62728',alpha=.45)
    ax[2].hist(a.dH,bins=bins,weights=a.w_ml,density=True,histtype='step',color='k',lw=1.2)
    xx=np.linspace(15,45,200); s=np.sqrt(C[1,1]); ax[2].plot(xx,np.exp(-0.5*((xx-mu[1])/s)**2)/(s*np.sqrt(2*np.pi)),'k:',lw=1)
    ax[2].set_xlabel(r'$\Delta H$ (kJ mol$^{-1}$)'); ax[2].set_ylabel('density')
    for x_,l in zip(ax,'abc'): x_.text(-0.22,1.03,f'({l})',transform=x_.transAxes,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG+'fig3_abc.png')
if which in ('all','cycle'):
    from tank import cycle
    names=['lnP0','dH','beta','zeta','eta','lam','hbw','hout','lkm','Psup','Pcut','Tamb']
    med={k:post['summ'][k]['post'][1] for k in names}; p=dict(med); p['km']=float(np.exp(p.pop('lkm')))
    # use accepted sample closest to the median in normalised space
    Z=(a[names]-a[names].median())/a[names].std(); r=a.iloc[int((Z**2).sum(1).values.argmin())]
    p={k:float(r[k]) for k in names}; p['km']=float(np.exp(p.pop('lkm')))
    json.dump(p,open(OUT+'representative.json','w'),indent=1)
    o=cycle(p,10,full=True); A,D=o['a'],o['d']
    ppc=pd.read_csv(OUT+'ppc.csv')
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.5))
    ta=A['t']/60; td=D['t']/60+ta[-1]+5
    ax[0].plot(ta,A['P'],'k',lw=1); ax[0].plot(td,D['P'],'k',lw=1); ax[0].set_ylabel('Pressure (bar)'); ax[0].set_xlabel('Time (min)')
    ax0=ax[0].twinx(); ax0.plot(ta,(A['ntot']-A['ntot'][0])*2.016,'C0',lw=1); ax0.plot(td,(D['ntot'][0]-D['ntot'])*2.016,'C0--',lw=1); ax0.set_ylabel('H$_2$ transferred (g)',color='C0')
    ax[1].plot(ta,A['Tw']-273.15,'C3',lw=1,label='shell'); ax[1].plot(ta,A['Tc']-273.15,'C3:',lw=1,label='bed centre')
    ax[1].plot(td,D['Tw']-273.15,'C3',lw=1); ax[1].plot(td,D['Tc']-273.15,'C3:',lw=1)
    ax[1].axhline(75,color='grey',lw=.6,ls='--'); ax[1].axhline(-18,color='grey',lw=.6,ls='--')
    ax[1].set_ylabel('Temperature (°C)'); ax[1].set_xlabel('Time (min)'); ax[1].legend(fontsize=6.5,frameon=False)
    obs={5:(78.03,65.22),10:(92.69,61.06),15:(91.29,65.77)}
    for i,F in enumerate([5,10,15]):
        s=ppc[ppc.F==F]
        for j,(k,c) in enumerate([('m_abs','C0'),('m_des','C2')]):
            q=np.nanpercentile(s[k],[5,50,95]); x=i+(j-0.5)*0.3
            ax[2].errorbar(x,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt='o',color=c,ms=4,capsize=2,label=('absorbed' if k=='m_abs' else 'released') if i==0 else None)
            ax[2].plot(x,obs[F][j],'kx',ms=6,mew=1.2,label='measured' if (i==0 and j==0) else None)
    ax[2].set_xticks([0,1,2]); ax[2].set_xticklabels(['5','10*','15']); ax[2].set_xlabel('Charge rate (SLPM)'); ax[2].set_ylabel('H$_2$ (g)')
    ax[2].set_ylim(38,112); ax[2].legend(fontsize=6,frameon=False,loc='lower left')
    for x_,l in zip(ax,'abc'): x_.text(-0.22,1.03,f'({l})',transform=x_.transAxes,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG+'fig4_cycle_ppc.png')
if which in ('all','scen'):
    s=pd.read_csv(OUT+'scen.csv')
    order=['A','B1','B2','C0','C']; lab=['A\ncalibrated','B1\nTi$_{0.8}$Zr$_{0.2}$CrMn','B2\nZr$_{0.2}$Ti$_{0.8}$MnCr','C$_0$\nGBR point','C\nGBR + error']
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.7))
    for i,(k,yl) in enumerate([('m_des','H$_2$ delivered at 5 SLPM (g)'),('m_abs','H$_2$ absorbed at 10 SLPM (g)'),('Tmin','Min. shell temperature (°C)')]):
        data=[s[s.scen==c][k].dropna().values for c in order]
        bp=ax[i].boxplot(data,widths=.55,patch_artist=True,showfliers=False,medianprops=dict(color='k'))
        for b,c in zip(bp['boxes'],['#d62728','#9467bd','#8c564b','#1f77b4','#aec7e8']): b.set_facecolor(c); b.set_alpha(.7)
        ax[i].set_xticks(range(1,6)); ax[i].set_xticklabels(['A','B1','B2','C$_0$','C']); ax[i].set_ylabel(yl)
        ref={'m_des':61.06,'m_abs':92.69,'Tmin':-18}[k]; ax[i].axhline(ref,color='grey',ls='--',lw=.7)
        ax[i].text(-0.24,1.03,f'({"abc"[i]})',transform=ax[i].transAxes,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG+'fig5_scenarios.png')
if which in ('all','window'):
    w=pd.read_csv(OUT+'window.csv'); g=w[w['var']=='grid']; sb=json.load(open(OUT+'sobol.json'))
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw=dict(width_ratios=[1.05,1]))
    G1=np.sort(g.g1.unique()); G2=np.sort(g.g2.unique())
    Z=g.pivot(index='g2',columns='g1',values='m_des').loc[G2,G1].values
    ref=np.nanmax(w[(w['var']=='lnP0')&(w.delta.abs()<1e-9)].m_des.values)
    cs=ax[0].contourf(G1,G2,Z,levels=np.linspace(0,max(100,np.nanmax(Z)),11),cmap='viridis')
    ax[0].contour(G1,G2,Z/ref,levels=[0.9],colors='w',linewidths=1.2)
    plt.colorbar(cs,ax=ax[0],label='H$_2$ delivered at 5 SLPM (g)')
    cov=np.array(post['ml_cov'])
    ellipse(ax[0],(0,0),cov,1,ec='r',lw=1); 
    o=pd.read_csv(OUT+'oof_gbr.csv'); o5=o[o.cls2=='AB5']; C5=np.cov(np.vstack([o5.pL-o5.lnP,o5.pH-o5.dH]))
    ellipse(ax[0],(0,0),C5,1,ec='orange',lw=1,ls='--')
    ax[0].plot(0,0,'w+',ms=8)
    ax[0].set_xlabel(r'$\delta$ ln $P_{eq,0}$ from calibrated value'); ax[0].set_ylabel(r'$\delta\Delta H$ (kJ mol$^{-1}$)')
    pr=pd.read_csv(OUT+'pred_targets.csv').iloc[0]; rep=json.load(open(OUT+'representative.json'))
    ax[0].plot(pr.lnP-rep['lnP0'],pr.dH-rep['dH'],'r*',ms=9,mec='w',mew=.6)
    from matplotlib.lines import Line2D
    hh=[Line2D([],[],color='0.6',lw=1.2,label='90% of calibrated (white)'),Line2D([],[],color='r',lw=1,label='AB$_2$ CV error, 1σ'),
        Line2D([],[],color='orange',lw=1,ls='--',label='AB$_5$ CV error, 1σ'),Line2D([],[],color='r',marker='*',ls='',ms=7,label='GBR prediction')]
    leg=ax[0].legend(handles=hh,fontsize=5.8,loc='upper left',bbox_to_anchor=(0,1.24),ncol=2,frameon=False)
    for t in leg.get_lines(): pass
    hh[0].set_color('k')
    names=sb['problem']['names']; nice={'lnP0':'ln $P_{eq,0}$','dH':'ΔH','beta':'β','eta':'η','lam':'λ$_{eff}$','hout':'$h_{out}$','lkm':'ln $k_m$','hbw':'$h_{bw}$'}
    x=np.arange(len(names)); wd=0.2
    for j,(k,c) in enumerate([('m_des','#1f77b4'),('m_abs','#ff7f0e'),('Tmin','#2ca02c'),('t_abs','#9467bd')]):
        ST=np.array(sb['res'][k]['ST']); ax[1].bar(x+(j-1.5)*wd,ST,wd,color=c,label={'m_des':'delivered','m_abs':'absorbed','Tmin':'min. T','t_abs':'charge time'}[k])
    ax[1].set_xticks(x); ax[1].set_xticklabels([nice[n] for n in names],fontsize=7.5); ax[1].set_ylabel('Total Sobol index $S_T$')
    ax[1].legend(fontsize=6.5,frameon=False,ncol=2)
    for x_,l in zip(ax,'ab'): x_.text(-0.2,1.03,f'({l})',transform=x_.transAxes,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG+'fig6_window_sobol.png',bbox_inches='tight')
if which in ('all','screen'):
    s=pd.read_csv(OUT+'screen.csv')
    comps=[c for c in s.comp.unique()]
    lbl={'Ti0.815Zr0.185Cr0.77Fe0.07Mn0.88':'AB$_2$ alloy (GBR)','LaNi5':'LaNi$_5$','La0.9Ce0.1Ni5':'Ce 0.1','La0.8Ce0.2Ni5':'Ce 0.2','La0.75Ce0.25Ni5':'Ce 0.25','La0.5Ce0.5Ni5':'Ce 0.5',
         'LaNi4.9Al0.1':'Al 0.1','LaNi4.8Al0.2':'Al 0.2','LaNi4.7Al0.3':'Al 0.3','LaNi4.6Al0.4':'Al 0.4','LaNi4.45Al0.55':'Al 0.55','LaNi4.25Al0.75':'Al 0.75'}
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
    m=s[s.src=='ML'].set_index('comp').loc[[c for c in lbl if c in set(s.comp)]]
    x=np.arange(len(m))
    ax[0].bar(x-0.2,m.m_abs,0.4,label='absorbed, 10 SLPM',color='#ff7f0e'); ax[0].bar(x+0.2,m.m_des,0.4,label='delivered, 5 SLPM',color='#1f77b4')
    db=s[s.src=='DB'].set_index('comp')
    for i,c in enumerate(m.index):
        if c in db.index: ax[0].plot(i+0.2,db.loc[c,'m_des'],'kx',ms=5,mew=1.1)
    ax[0].set_xticks(x); ax[0].set_xticklabels([lbl[c] for c in m.index],rotation=60,ha='right',fontsize=7); ax[0].set_ylabel('H$_2$ (g)')
    ax[0].legend(fontsize=6.5,frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2)
    ax[1].scatter(m.lnP0,m.m_des,c='#1f77b4',s=18)
    offs={'Ti0.815Zr0.185Cr0.77Fe0.07Mn0.88':(-60,6),'LaNi5':(4,-9),'La0.9Ce0.1Ni5':(5,-6),'La0.8Ce0.2Ni5':(-28,6),'La0.5Ce0.5Ni5':(-34,-3),'LaNi4.45Al0.55':(-6,7),'LaNi4.25Al0.75':(-4,7)}
    for c,r in m.iterrows():
        if c in offs: ax[1].annotate(lbl[c],(r.lnP0,r.m_des),fontsize=6,xytext=offs[c],textcoords='offset points')
    ax[1].set_xlabel(r'Predicted ln($P_{eq,0}$/atm)'); ax[1].set_ylabel('H$_2$ delivered at 5 SLPM (g)')
    ax[0].text(-0.18,1.03,'(a)',transform=ax[0].transAxes,fontweight='bold'); ax[1].text(-0.2,1.08,'(b)',transform=ax[1].transAxes,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG+'fig7_screen.png')
