"""Reconcile descriptive counts and evaluate common horizons without refitting models.

Original analysis files are read only. Outputs use the corrected initial observed
state in the relapse history and retain a row-level ledger for every new breach.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
DEST = Path(__file__).resolve().parents[1] / 'tables'
DEST.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
from survlib import surv_from_h, eval_ipcw_strat

w = pd.read_parquet(OUT / 'cnemc_weekly.parquet')
w = w[w.date.dt.year >= 2008]
rows = []
for site, g in w.groupby('site'):
    g = g.set_index('date').sort_index()
    idx = pd.date_range(g.index.min(), g.index.max(), freq='W-MON')
    g = g.reindex(idx)
    obs = g[['NH4N','DO','CODMn']].notna().all(axis=1).to_numpy()
    x = ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).to_numpy()
    last, old_last = -999, -999
    for i in range(len(idx)):
        if i > 0 and obs[i] and obs[i-1] and x[i] and not x[i-1]:
            lag = i-1-last
            rows.append(dict(site=site,date=idx[i],year=idx[i].year,
                relapse=int(1 <= lag <= 8),
                legacy_relapse=int(1 <= i-1-old_last <= 8),
                weeks_since_breach_at_landmark=lag))
        if obs[i] and x[i]:
            last = i
            if i > 0: old_last = i
events = pd.DataFrame(rows)
events.to_csv(DEST / 'reconciled_onset_ledger.csv', index=False, encoding='utf-8-sig')
by = events.groupby('year').agg(n_onsets=('relapse','size'), n_relapses=('relapse','sum'),
    n_relapses_legacy=('legacy_relapse','sum'))
by['relapse_share'] = by.n_relapses / by.n_onsets
by['legacy_relapse_share'] = by.n_relapses_legacy / by.n_onsets
by.to_csv(DEST / 'reconciled_relapse_share_by_year.csv', encoding='utf-8-sig')
rng = np.random.default_rng(20261022)
tab = events.groupby('site').relapse.agg(['sum','count'])
bs = []
for _ in range(500):
    b = tab.loc[rng.choice(tab.index, len(tab), replace=True)]
    bs.append(b['sum'].sum() / b['count'].sum())
headline = dict(n_onsets=len(events),n_relapses=int(events.relapse.sum()),
    relapse_share=events.relapse.mean(),relapse_share_lo=float(np.quantile(bs,.025)),
    relapse_share_hi=float(np.quantile(bs,.975)),bootstrap_replicates=500,
    bootstrap_seed=20261022,n_relapses_legacy=int(events.legacy_relapse.sum()),
    legacy_share=float(events.legacy_relapse.mean()))
pd.DataFrame([headline]).to_csv(DEST / 'reconciled_headlines.csv',index=False,encoding='utf-8-sig')
events[events.relapse != events.legacy_relapse].to_csv(DEST / 'reconciled_initialization_differences.csv',index=False,encoding='utf-8-sig')
assert headline['n_relapses'] == 1191 and len(events) == 1687

L = pd.read_parquet(OUT / 'landmarks.parquet').reset_index(drop=True)
H = pd.read_pickle(OUT / 'core' / 'hazards.pkl')
BF = pd.read_pickle(OUT / 'core' / 'baseline_F_test.pkl')
rr = []
for trans, state in [('onset',0),('recovery',1)]:
    test = (L.date >= '2015-01-01') & (L.state == state)
    Lt = L[test].reset_index(drop=True)
    all_f = {m:1-surv_from_h(v['te']) for m,v in H[trans].items()}
    all_f.update({'Forecast-threshold (max)':BF[trans+'_FTmax'],
        'Forecast-threshold (indep.)':BF[trans+'_FTind'], 'Horizon classifiers':BF[trans]})
    for per, mask in [('2015-2016',Lt.year<=2016),('2017-2018',Lt.year>=2017)]:
        for name,F in all_f.items():
            r=eval_ipcw_strat(F[mask],Lt[mask],Lt.year.to_numpy()[mask],horizons=(1,2,4))
            rr.append(dict(trans=trans,model=name,period=per,horizons='1;2;4',**r))
common=pd.DataFrame(rr)
common.to_csv(DEST/'reconciled_common_horizon_metrics.csv',index=False,encoding='utf-8-sig')
print(json.dumps(headline,indent=2))
print(common[(common.trans=='onset') & common.model.isin(['BRR-SL','Forecast-threshold (max)','Station KM'])][['model','period','IBS','AUCw@4']].to_string(index=False))
