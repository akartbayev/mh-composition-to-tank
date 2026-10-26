"""Time-series protocol for the open AB2 tank data, built on tank.Tank."""
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, json, pandas as pd
from scipy.integrate import solve_ivp
from tank import Tank, Rg, M, SLPM
DD=ROOT+'tank_data/'
META={int(k):v for k,v in json.load(open(DD+'meta.json')).items()}
TAMB={5:25.3,10:26.2,15:26.1}   # ambient = tank temperature at the end of the long desorption logs
def _sim(tk,mode,F,y0,tmax,t_eval):
    q=tk.p; N=q['N']
    if mode=='rest': f=lambda t,y: tk.rhs(t,y,'abs',0.0)[0]
    else: f=lambda t,y: tk.rhs(t,y,mode,F)[0]
    sol=solve_ivp(f,(0,tmax),y0,method='BDF',t_eval=t_eval,rtol=1e-4,
                  atol=np.r_[np.full(N,1e-6),np.full(N,1e-3),1e-3,1e-6],max_step=300)
    return sol
def outputs(tk,sol):
    q=tk.p; N=q['N']; ys=sol.y
    xt=(ys[:N]*tk.nmaxj[:,None]).sum(0)/tk.nmax
    ntot=xt*tk.nmax+ys[2*N+1]
    P=ys[2*N+1]*Rg*ys[N:2*N].mean(0)/q['Vg']/1e5
    return dict(t=sol.t,P=P,Tw=ys[2*N]-273.15,Tc=ys[N]-273.15,ntot=ntot,x=xt)
def simulate_run(p,F,tabs=None,tdes=None,des_hours=7.0,full=False):
    """absorption at set flow F with pressure limit Psup (measured), duration as measured; rest gap; 5 SLPM desorption to Pcut."""
    m=META[F]; pp=dict(p); pp.update(Psup=m['Psup'],Pcut=m['Pcut'],Tamb=TAMB[F]+273.15)
    tk=Tank(pp); q=tk.p
    T0=m['Tamb_abs']+273.15
    xs=np.linspace(1e-4,0.5,3000); ld=tk.lnpeq(xs,T0,-1); x0=xs[max(np.searchsorted(ld,np.log(0.05))-1,0)]
    y=np.concatenate([np.full(q['N'],x0),np.full(q['N'],T0),[T0,0.05e5*q['Vg']/(Rg*T0)]])
    dur=m['dur_abs']
    te=tabs if tabs is not None else np.linspace(0,dur,400)
    sa=_sim(tk,'abs',F,y,dur,te[te<=dur])
    if sa.status<0 or sa.y.shape[1]==0: raise RuntimeError('abs fail')
    A=outputs(tk,sa); y=sa.y[:,-1]
    if m['gap_h']>0.1:
        sr=_sim(tk,'rest',0,y,m['gap_h']*3600,None); y=sr.y[:,-1]
    td=tdes if tdes is not None else np.linspace(0,des_hours*3600,300)
    sd=_sim(tk,'des',5.0,y,td[-1],td)
    if sd.status<0: raise RuntimeError('des fail')
    D=outputs(tk,sd)
    A['g']=(A['ntot']-A['ntot'][0])*M*1e3; D['g']=(D['ntot'][0]-D['ntot'])*M*1e3
    return A,D

def design_run(p,F_abs=10.0,Psup=25.4,Tamb=26.0,abs_h=18.0,Pcut=1.4,des_h=7.0):
    """Design protocol used for scenarios: evacuated start at Tamb, charge at F_abs with supply limit Psup for abs_h,
    then immediately discharge at 5 SLPM down to Pcut for des_h. Returns KPIs."""
    pp=dict(p); pp.update(Psup=Psup,Pcut=Pcut,Tamb=Tamb+273.15)
    tk=Tank(pp); q=tk.p; T0=Tamb+273.15
    xs=np.linspace(1e-4,0.5,3000); ld=tk.lnpeq(xs,T0,-1); x0=xs[max(np.searchsorted(ld,np.log(0.05))-1,0)]
    y=np.concatenate([np.full(q['N'],x0),np.full(q['N'],T0),[T0,0.05e5*q['Vg']/(Rg*T0)]])
    ta=np.linspace(0,abs_h*3600,400); sa=_sim(tk,'abs',F_abs,y,ta[-1],ta); A=outputs(tk,sa)
    td=np.linspace(0,des_h*3600,300); sd=_sim(tk,'des',5.0,sa.y[:,-1],td[-1],td); D=outputs(tk,sd)
    gA=(A['ntot']-A['ntot'][0])*M*1e3; gD=(D['ntot'][0]-D['ntot'])*M*1e3
    t60=float(np.interp(60,gA,A['t'])/3600) if gA[-1]>=60 else np.nan
    return dict(m_abs=float(gA[-1]),m_des=float(gD[-1]),Tmax=float(A['Tw'].max()),Tmin=float(D['Tw'].min()),t60=t60,
                Tcmax=float(A['Tc'].max()),Tcmin=float(D['Tc'].min()))
