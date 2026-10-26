# Composition-to-tank modelling of metal hydride hydrogen storage

Code and processed data for the article *Physics-Informed Composition-to-Tank Modeling of Metal Hydride Hydrogen Storage* (A. Kartbayev, D. Chowdhury; submitted to *Hydrogen*, MDPI).

The study links composition-based machine-learning predictions of hydride thermodynamics to a physics-based model of a 7.8 kg AB2 tank, calibrated on the time series of an open test data set, and asks how accurate the predicted thermodynamics must be for tank-level decisions.

## Contents
| Path | Description |
|---|---|
| `code/ml.py`, `code/ml2.py` | Equilibrium models (gradient boosting, random forest) on ML-HydPARK; cross-validation, leave-class-out tests, predictions |
| `code/tank.py` | 1D radial metal hydride bed with lumped steel shell (van 't Hoff + slope + hysteresis, Avrami-type kinetics, gas balance) |
| `code/tank2.py` | Test protocol as logged (`simulate_run`) and design protocol (`design_run`) |
| `code/fit_ts.py` | Weighted least-squares calibration on the time series of one test: `python fit_ts.py 10` |
| `code/val_ts.py` | Validation of the 10 SLPM calibration on the 5 and 15 SLPM tests |
| `code/profile_ts.py` | Profile of the calibration objective over ln Peq,0 |
| `code/analysis3.py` | Stages `ens`, `scen`, `window`, `screen`, `sobol N` (scenarios, tolerance window, AB5 screening, Sobol indices) |
| `code/fig*.py` | Figures |
| `data/mlhydpark_extracted.csv` | ML-HydPARK v0.0.5, 420 entries |
| `tank_data/` | Downsampled time series and per-test boundary conditions derived from the open tank data |
| `out/` | Intermediate results (CSV/JSON) |
| `fig/` | Figures of the article |
| `legacy/` | Earlier run-level calibration (approximate Bayesian computation), kept for reference |

## Reproducing
```bash
pip install numpy pandas scipy scikit-learn matminer pymatgen SALib matplotlib
cd code
python ml.py && python ml2.py          # equilibrium models
python fit_ts.py 10                    # calibration (also 5, 15)
python val_ts.py 10 && python profile_ts.py
python analysis3.py ens && python analysis3.py scen && python analysis3.py window
python analysis3.py screen && python analysis3.py sobol 256
```
Paths are resolved relative to the repository root, so the scripts can be run from any location.

## Data sources
- Witman, M.; Allendorf, M.; Stavila, V. Database for Machine Learning of Hydrogen Storage Materials Properties, v0.0.5. Zenodo, 2024. https://doi.org/10.5281/zenodo.10680097
- Ramadhan, F.; Suwarno, S.; Hijriawan, M. Experimental Dataset on the Absorption–Desorption Dynamics of an AB2 Metal Hydride Hydrogen Storage Tank for a 300 Watt PEM Fuel Cell. Mendeley Data, V1, 2026. https://doi.org/10.17632/8km9z62tct.1 (CC BY 4.0)

## Licence
Code: MIT. Derived data: CC BY 4.0, with attribution to the original data sets above.
