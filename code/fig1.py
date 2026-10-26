import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5})
fig,ax=plt.subplots(figsize=(7.2,3.1)); ax.set_xlim(0,100); ax.set_ylim(0,44); ax.axis('off')
def box(x,y,w,h,title,body,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fc,ec='#333',lw=.8))
    ax.text(x+w/2,y+h-2.2,title,ha='center',va='top',fontweight='bold',fontsize=8.5)
    ax.text(x+w/2,y+h-6.3,body,ha='center',va='top',fontsize=7.2,linespacing=1.35)
def arr(x1,y1,x2,y2,t=''):
    ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=1,color='#333'))
    if t: ax.text((x1+x2)/2,(y1+y2)/2+1.2,t,ha='center',fontsize=6.8,style='italic')
box(1,22,20,20,'Composition','420 hydrides\n(ML-HydPARK)\nMagpie/stoichiometry/\nvalence descriptors','#e8f0fa')
box(25,22,22,20,'Equilibrium model','GBR: ΔH, ln P$_{eq}$(298 K)\nΔS from van ’t Hoff\nCV residuals → error\ndistribution','#eaf6ea')
box(51,22,22,20,'Tank model','1D radial bed + shell\nvan ’t Hoff + slope +\nhysteresis, Avrami-type\nkinetics, gas balance','#fdf0e6')
box(77,22,22,20,'Tank KPIs','H$_2$ absorbed and\ndelivered, charge time,\nbed/shell temperature\nextrema','#f3ecf8')
arr(21.8,32,24.2,32); arr(47.8,32,50.2,32,''); arr(73.8,32,76.2,32)
box(27,1,45,15,'Open AB2 tank data (7.8 kg alloy)','1 s logs of pressure, 5 thermocouples, flow\n→ least-squares calibration (10 SLPM);\nvalidation at 5 and 15 SLPM','#fff8d6')
arr(62,16.6,62,21.2,''); ax.text(63,18.3,'calibration',fontsize=6.8,style='italic')
box(74,1,25,15,'Analyses','scenarios A/B/C,\nSobol indices, accuracy\nwindow, AB5 screening','#eeeeee')
arr(88,21.2,88,16.6,'')
plt.savefig(ROOT+'fig/fig1_framework.png',dpi=300,bbox_inches='tight')
