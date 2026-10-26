import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np
from scipy.integrate import solve_ivp
Rg=8.314462618; T0=298.15; M=2.016e-3
SLPM=1e-3/60*101325/(Rg*273.15)   # mol/s per SLPM
class Tank:
    """1D radial finite-volume model of a cylindrical MH bed with a lumped steel shell."""
    def __init__(s, p):
        s.p=dict(N=6, ri=0.04272, V=2.532e-3, ms=7.8, rho_s=6.1e3, cps=480., wmax=0.0176,
                 mw=10.0, cw=500., hbw=250., Vdead=1.0e-4, zeta=0.15,
                 Ca=59.2, Ea=21180., Cd=9.57, Ed=16450.)
        s.p.update(p); q=s.p
        q['L']=q['V']/(np.pi*q['ri']**2)
        q['eps']=1-q['ms']/q['rho_s']/q['V']
        q['Vg']=q['eps']*q['V']+q['Vdead']
        N=q['N']; r=np.linspace(0,q['ri'],N+1); s.r=r
        s.Vj=np.pi*(r[1:]**2-r[:-1]**2)*q['L']; s.mj=q['ms']*s.Vj/q['V']
        s.nmaxj=s.mj*q['wmax']/M; s.nmax=s.nmaxj.sum()
        rc=0.5*(r[1:]+r[:-1]); s.rc=rc
        s.Gint=q['lam']*2*np.pi*r[1:-1]*q['L']/(rc[1:]-rc[:-1])     # internal conductances W/K
        rw=q['ri']; dr=q['ri']-rc[-1]
        s.Gbw=1/(1/(q['lam']*2*np.pi*rw*q['L']/dr)+1/(q['hbw']*2*np.pi*rw*q['L']))
        s.Aout=2*np.pi*0.0508*q['L']+2*np.pi*0.0508**2
    def lnpeq(s,x,T,branch):
        q=s.p; xc=np.clip(x,1e-4,1-1e-4)
        return (q['lnP0']+np.log(1.01325)+q['dH']*1e3/Rg*(1/T0-1/T)+q['beta']*(xc-0.5)
                +q['zeta']*np.log(xc/(1-xc))+branch*q['eta']/2)
    def rates(s,x,T,P):
        q=s.p; lnP=np.log(max(P,1e-6))
        la=s.lnpeq(x,T,+1); ld=s.lnpeq(x,T,-1)
        ra=q['km']*q['Ca']*np.exp(-q['Ea']/(Rg*T))*np.clip(lnP-la,0,None)*(1-x)
        rd=q['km']*q['Cd']*np.exp(-q['Ed']/(Rg*T))*np.clip(ld-lnP,0,None)*x
        return ra-rd     # dx/dt
    def rhs(s,t,y,mode,F):
        q=s.p; N=q['N']
        x=y[:N]; T=y[N:2*N]; Tw=y[2*N]; ng=y[2*N+1]
        Tg=T.mean(); P=max(ng,1e-9)*Rg*Tg/q['Vg']/1e5
        dx=s.rates(x,T,P); nabs=s.nmaxj*dx
        if mode=='abs':
            fin=F*SLPM/(1+np.exp(np.clip((P-q["Psup"])/0.05,-50,50)))
        else:
            fin=-F*SLPM/(1+np.exp(np.clip((q["Pcut"]-P)/0.02,-50,50)))
        Q=nabs*q['dH']*1e3
        cond=np.zeros(N); f=s.Gint*(T[1:]-T[:-1]); cond[:-1]+=f; cond[1:]-=f
        qbw=s.Gbw*(T[-1]-Tw); cond[-1]-=qbw
        dT=(cond+Q)/(s.mj*q['cps'])
        dTw=(qbw-q['hout']*s.Aout*(Tw-q['Tamb']))/(q['mw']*q['cw'])
        dng=fin-nabs.sum()
        return np.concatenate([dx,dT,[dTw,dng]]), fin, P
    def eq_state(s,xbar):
        q=s.p; T=q['Tamb']
        P=np.exp(s.lnpeq(xbar,T,0)); ng=P*1e5*q['Vg']/(Rg*T)
        return np.concatenate([np.full(q['N'],xbar),np.full(q['N'],T),[T,ng]])
    def run(s,mode,F,y0,tmax=8*3600):
        q=s.p; N=q['N']
        f=lambda t,y: s.rhs(t,y,mode,F)[0]
        if mode=='abs':
            ev=lambda t,y: s.rhs(t,y,mode,F)[1]-0.10*F*SLPM
        else:
            ev=lambda t,y: s.rhs(t,y,mode,F)[2]-q['Pcut']*1.02
        ev.terminal=True; ev.direction=-1
        sol=solve_ivp(f,(0,tmax),y0,method='BDF',events=ev,rtol=1e-5,atol=np.r_[np.full(N,1e-6),np.full(N,1e-3),1e-3,1e-6],max_step=120)
        ys=sol.y; ts=sol.t
        xt=(ys[:N]*s.nmaxj[:,None]).sum(0)/s.nmax
        Tw=ys[2*N]; Tb=ys[N:2*N]
        ntot=xt*s.nmax+ys[2*N+1]
        P=ys[2*N+1]*Rg*Tb.mean(0)/q['Vg']/1e5
        return dict(t=ts,x=xt,Tw=Tw,Tc=Tb[0],P=P,ntot=ntot,yend=ys[:,-1],status=sol.status)
def cycle(p,F_abs,F_des=5.0,full=False,P_start=0.1):
    """evacuated/activated start (equilibrium at P_start, Tamb) -> absorb(F) -> rest -> desorb(5 SLPM) to Pcut"""
    tk=Tank(p); q=tk.p
    xs=np.linspace(1e-4,0.999,4000); ld=tk.lnpeq(xs,q['Tamb'],-1)
    x0=xs[max(np.searchsorted(ld,np.log(P_start))-1,0)]
    y=tk.eq_state(x0); y[-1]=P_start*1e5*q['Vg']/(Rg*q['Tamb'])
    a=tk.run('abs',F_abs,y)
    m_abs=(a['ntot'][-1]-a['ntot'][0])*M*1e3
    y=tk.eq_state(a['x'][-1]); y[-1]=a['yend'][-1]
    d=tk.run('des',F_des,y)
    m_des=(d['ntot'][0]-d['ntot'][-1])*M*1e3
    out=dict(m_abs=m_abs,m_des=m_des,Tmax=a['Tw'].max()-273.15,Tmin=d['Tw'].min()-273.15,
             Tcmax=a['Tc'].max()-273.15,Tcmin=d['Tc'].min()-273.15,
             t_abs=a['t'][-1]/60,t_des=d['t'][-1]/60,x0=x0,ok=(a['status']>=0 and d['status']>=0))
    if full: out.update(a=a,d=d)
    return out
