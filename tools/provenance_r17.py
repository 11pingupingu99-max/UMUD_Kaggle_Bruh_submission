import sys, json, re, itertools
import numpy as np, pandas as pd
B = sys.argv[1]
c2 = ''.join(json.load(open('/home/claude/r83/UMUD_R83_KAGGLE.ipynb'))['cells'][2]['source'])
T = {n: json.loads(re.search(n + r" = json\.loads\('(.*?)'\)", c2).group(1)) for n in ('R17DEV', 'R17ENS', 'V2MT')}
IDS = sorted(T['V2MT'])
L = pd.read_csv(f'{B}/csv/R50T_LATEST_test.csv').set_index('image_id').reindex(IDS)
rf = pd.read_csv(f'{B}/files/assets/r17c_refit_test.csv').set_index('image_id').reindex(IDS)
dv = pd.read_csv(f'{B}/files/assets/r17c_develop_test.csv').set_index('image_id').reindex(IDS)
C = {}
for c in L.columns:
    if pd.api.types.is_numeric_dtype(L[c]): C['L:' + c] = L[c].values.astype(float)
for n, d in (('refit', rf), ('develop', dv)):
    for c in ('pa_deg', 'fl_mm', 'mt_mm'): C[f'r17c_{n}:{c}'] = d[c].values.astype(float)
models = ('develop', 'refit', 'seed1', 'seed2'); modes = ('plain', 'flip', 'sq')
for t in ('pa_deg', 'fl_mm'):
    for m in models:
        for k in (2, 3):
            for sub in itertools.combinations(modes, k):
                C[f'mean_{m}_{"+".join(sub)}_{t}'] = np.nanmean([L[f'{m}_{s}_{t}'].values for s in sub], 0)
                C[f'median_{m}_{"+".join(sub)}_{t}'] = np.nanmedian([L[f'{m}_{s}_{t}'].values for s in sub], 0)
    for k in (2, 3, 4):
        for ms in itertools.combinations(models, k):
            for sub in [('plain',), ('flip',), ('plain', 'flip'), ('plain', 'flip', 'sq')]:
                arr = [L[f'{m}_{s}_{t}'].values for m in ms for s in sub]
                C[f'mean_{"+".join(ms)}_{"+".join(sub)}_{t}'] = np.nanmean(arr, 0)
                C[f'median_{"+".join(ms)}_{"+".join(sub)}_{t}'] = np.nanmedian(arr, 0)
S = pd.DataFrame(C, index=IDS); M = S.values
def best(v, tol=2e-3):
    ok = np.isfinite(v)
    agree = (np.abs(M[ok] - v[ok, None]) <= tol * np.maximum(1, np.abs(v[ok, None]))).sum(0)
    j = np.argsort(-agree)[:3]
    return [(S.columns[i], int(agree[i])) for i in j], int(ok.sum())
for nm in ('R17DEV', 'R17ENS'):
    Lk = max(len(v) for v in T[nm].values() if v)
    for k in range(Lk):
        v = np.array([np.nan if (T[nm][i] is None or len(T[nm][i]) <= k or T[nm][i][k] is None) else T[nm][i][k] for i in IDS], float)
        b, n = best(v); print(f'{nm}[{k}] n={n}:', b)
print('brakujace R17DEV / R17ENS:', [i for i in IDS if not T['R17DEV'][i]][:12], [i for i in IDS if not T['R17ENS'][i]][:12])
