import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.8,'savefig.dpi':300})
o=pd.read_csv(ROOT+'out/oof_gbr.csv')
cols={'AB5':'#1f77b4','AB2':'#d62728','AB':'#2ca02c','SS':'#9467bd','MIC':'#8c564b','Mg':'#e377c2','A2B':'#7f7f7f'}
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6))
for k,c in cols.items():
    s=o[o.cls2==k]
    ax[0].scatter(s.dH,s.pH,s=7,c=c,label=k,alpha=.75,lw=0)
    ax[1].scatter(s.lnP,s.pL,s=7,c=c,alpha=.75,lw=0)
ax[0].plot([0,100],[0,100],'k--',lw=.7); ax[0].set_xlim(0,100); ax[0].set_ylim(0,100)
ax[0].set_xlabel(r'Reported $\Delta H$ (kJ mol$^{-1}$ H$_2$)'); ax[0].set_ylabel(r'Predicted $\Delta H$')
ax[1].plot([-20,10],[-20,10],'k--',lw=.7); ax[1].set_xlim(-20,10); ax[1].set_ylim(-20,10)
ax[1].set_xlabel(r'Reported ln($P_{eq}$/atm), 25 °C'); ax[1].set_ylabel(r'Predicted ln $P_{eq}$')
ax[0].legend(fontsize=6.5,frameon=False,ncol=1,loc='upper left',handletextpad=0.1)
g=o.assign(eL=(o.pL-o.lnP).abs()).groupby('cls2').eL.mean().reindex(['AB5','SS','AB2','Mg','AB','MIC'])
co=pd.read_csv(ROOT+'out/classout.csv').set_index('held')
x=np.arange(len(g)); ax[2].bar(x,g.values,color=[cols[k] for k in g.index],alpha=.85)
for k in ['AB2','AB5']:
    i=list(g.index).index(k); ax[2].bar(i,co.loc[k,'maeL'],fill=False,edgecolor='k',hatch='///',lw=.8)
ax[2].set_xticks(x); ax[2].set_xticklabels(g.index,fontsize=8); ax[2].set_ylabel('MAE of ln $P_{eq}$'); ax[2].set_ylim(0,5)
ax[2].text(0.02,0.97,'filled: random 5-fold CV\nhatched: class held out',transform=ax[2].transAxes,va='top',fontsize=6.5)
for a,l in zip(ax,'abc'): a.text(-0.2,1.04,f'({l})',transform=a.transAxes,fontweight='bold')
plt.tight_layout(); plt.savefig(ROOT+'fig/fig2_ml.png'); print(g, co)
