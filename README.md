# Surface-water warning horizons

Core analysis code for **Premature recovery declarations and persistent breaches near poorly diluted wastewater in Chinese surface waters** (previous title: *Persistent breach regimes and recovery timing guide warning horizons in China's surface waters*).

This repository contains the data-processing, event-time modelling, recurrence analysis, calibration, evaluation, recovery-confirmation and station-pressure code underlying the study. It excludes the raw dataset, external geospatial data, fitted models, manuscript, figure exports and local working files.

## Data

Obtain `china_full.csv` from the fixed Figshare release **version 4**, file **43613973**: https://doi.org/10.6084/m9.figshare.22584742.v4. Retain the original file and its source attribution and licence. The code selects the China National Environmental Monitoring Centre weekly observations. The study network contains 149 stations and 53,900 station-weeks with all three endpoint indicators available.

Put the file in `data/china_full.csv`, set the `WATER_QUALITY_DATA` environment variable, or pass its absolute path with `--data`. Derived files are written to `results/`; reconciled summary tables are written to `tables/`. Neither directory is tracked by Git.

The `pressures` and `routing` stages also read public geospatial products. Place them under `data/global/` with the layout below, or set the `GLOBAL_DATA_DIR` environment variable to a folder with the same layout. Keep the original files and follow each provider's licence and citation terms.

| Path under `data/global/` | Product | Source |
|---|---|---|
| `hydrowaste/HydroWASTE_v10.zip` | HydroWASTE v1.0 treatment plants | https://figshare.com/articles/dataset/HydroWASTE_version_1_0/14847786 |
| `hydrorivers_as/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp` | HydroRIVERS v1.0, Asia (unzipped shapefile) | https://www.hydrosheds.org/products/hydrorivers |
| `hydrobasins_as_lev04/hybas_as_lev04_v1c.shp` | HydroBASINS v1c, Asia, level 4 (unzipped shapefile) | https://www.hydrosheds.org/products/hydrobasins |
| `population/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.zip` | GHS Urban Centre Database 2015, R2019A v1.2 | https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_STAT_UCDB2015MT_GLOBE_R2019A/V1-2/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.zip |
| `worldclim/wc2.1_10m_bio.zip` | WorldClim 2.1 bioclimatic variables, 10 arc-minutes | https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip |
| `gdp/GDP_per_capita_PPP_1990_2015_v2.nc` | Gridded GDP per capita (PPP), 1990–2015, v2 (Kummu et al.) | https://doi.org/10.5061/dryad.dk1j0 |

## Environment and execution

Use Python 3.12 with the packages in `requirements.txt`. These versions describe the environment used to check this export, rather than an archived original-training environment. Select an appropriate CPU or GPU PyTorch build for your platform. Tree learners use up to 24 threads in the retained settings; reduce their thread parameters on smaller machines. Full model fitting, repeated cross-fitting and resampling can take substantial time and memory.

```bash
python -m pip install -r requirements.txt
python run_analysis.py
python run_analysis.py --stage prepare --data /absolute/path/to/china_full.csv
python run_analysis.py --stage core
python run_analysis.py --stage recurrence
python run_analysis.py --stage calibration
python run_analysis.py --stage spatial
python run_analysis.py --stage sensitivity
python run_analysis.py --stage reconcile
python run_analysis.py --stage checks
python run_analysis.py --stage pressures
python run_analysis.py --stage routing
```

The command without arguments lists stages and does not fit a model. Run the stages in the displayed order; `--stage all` runs the entire sequence. Later stages depend on earlier outputs. Original script numbers are retained to make provenance traceable; missing numbers refer to superseded or presentation-only scripts that are not required here.

## Analysis map

| Stage | Purpose |
|---|---|
| prepare | Harmonize weekly observations, construct landmarks and independent monitoring windows |
| core | Cross-fitted concentration forecasts and censored event-time learners; hazard-level stacking and matched comparators |
| recurrence | Historical-state contrasts, recurrence timing, station and station-year memoryless comparisons |
| calibration | Marginal and state/history-stratified lower bounds, empirical coverage, discrimination and retrospective window capture |
| spatial | Later-year station holdouts and station-level summaries |
| sensitivity | Component attribution, missingness, sequence models, threshold and corrected individual-indicator analyses |
| reconcile | Reconcile event initialization and common-horizon summary metrics |
| checks | Alternative memoryless and autocorrelated-concentration nulls; calibration, comparator, censoring and capture checks; monthly onset composition |
| pressures | Station wastewater, urban and climate pressures within great-circle distances; recovery-confirmation rules; pressure associations with spatial block bootstrap, sensitivity, confounding, seasonal and robustness checks |
| routing | HydroRIVERS routing of plants to stations, routed dilution and basin assignment; dilution and period models, residual correlogram, common-window confirmation, indicator completeness, conditional censoring weights, larger spatial blocks and dilution-factor forms |

`feature_dictionary.csv` maps code feature names to scientific definitions, feature groups and information availability. It contains no monitoring measurements. Shared model and evaluation functions are in `survlib.py`, `nnhazard.py`, `tristate.py` and `msboost.py`. `rerun_per_indicator_corrected.py` computes event and monitoring censoring separately for each indicator and is the retained individual-indicator analysis. `reconcile_evidence.py` reconstructs the onset cohort used for the reported 1,191/1,687 recurrence share.

## Interpretation and reproducibility

The mean Brier score reported in the paper is the arithmetic mean over the stated horizons. Some original internal output keys retain `IBS`; these keys do not denote a continuous-time integral. Coverage uses complete 12-week follow-up windows, whereas bound-length summaries can include all eligible landmarks. The station-year null is retrospective. The 2014 sample was reused for learner selection and calibration, so coverage is evaluated empirically. Monitoring capture counts overlapping positive four-week windows, rather than distinct pollution events or realized field cost savings.

Run `python tests/smoke_checks.py` for small deterministic checks of hazard survival, right-censoring likelihood and person-period expansion. Export validation includes syntax checks and rerunning the preparation stage against the original input; it does not claim that every training and sensitivity stage was refitted for this code release. The `checks`, `pressures` and `routing` scripts (27a–30g) were added in the second release; they match the analysis versions except that the external-data folder is read from `GLOBAL_DATA_DIR`. Numerical reproduction may depend on package versions, parallel execution and stochastic learners. Consult the manuscript and Supplementary Information for the full cohort and evaluation definitions.
