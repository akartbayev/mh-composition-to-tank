import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
import numpy as np, json, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from scipy.optimize import least_squares
from fit_ts import resid, names, lo, hi
OUT=ROOT+'out/'
best=np.array(json.load(open(OUT+'fit_ts_10.json'))['x'])
free=[i for i in range(len(names)) if names[i]!='lnP0']
def prof(v):
    def r(zf):
        z=best.copy(); z[0]=v; z[free]=zf; return resid(z,(10,))
    s=least_squares(r,best[free],bounds=(lo[free],hi[free]),x_scale=(hi-lo)[free]/10,diff_step=1e-3,max_nfev=200)
    return float(v),float(s.cost),s.x.tolist()
if __name__=='__main__':
    grid=np.round(np.linspace(0.6,3.6,13),3)
    with Pool(2) as pool: res=pool.map(prof,list(grid))
    json.dump(res,open(OUT+'profile_lnP0.json','w'))
    for v,c,_ in res: print(v,round(c,1))
