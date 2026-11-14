"""Faza C: pochodzenie wklejonych tabel R81/R83 -> kolumny eksportow (D5 R71T, R76T test_features, R80T test_r80).
Dla kazdej tabeli skalarnej i kazdej kolumny FX szukamy kolumny eksportu o najwiekszej zgodnosci (|roznica| <= tol)."""
import json, re
import numpy as np, pandas as pd
nb = json.load(open('/home/claude/r83/UMUD_R83_KAGGLE.ipynb')); c2 = ''.join(nb['cells'][2]['source'])
T = {n: json.loads(re.search(n + r" = json\.loads\('(.*?)'\)", c2).group(1)) for n in
     ('BAND', 'PADEC', 'FL22', 'OR_INS', 'V2MT', 'V2OV', 'R17DEV', 'R17ENS', 'FX_COLS', 'FX')}
IDS = sorted(T['V2MT'])
src = {}
for tag, path in (('d5', '/home/claude/r71t/real/D5/test_decoders.csv'), ('r76', '/home/claude/r76t/dl/test_features.csv'),
                  ('r80', '/home/claude/r80o_in/x/export_d81469c478/test_r80.csv')):
    d = pd.read_csv(path).set_index('image_id').reindex(IDS)
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            src[f'{tag}:{c}'] = d[c].values.astype(float)
S = pd.DataFrame(src, index=IDS)
print('kolumn eksportow:', S.shape[1])
M = S.values

def best(v, tol):
    v = np.asarray(v, float); ok = np.isfinite(v)
    if ok.sum() == 0:
        return None
    agree = (np.abs(M[ok] - v[ok, None]) <= tol * np.maximum(1, np.abs(v[ok, None]))).sum(0)
    j = int(np.argmax(agree))
    return S.columns[j], int(agree[j]), int(ok.sum())

rows = []
for n in ('BAND', 'FL22', 'OR_INS', 'V2MT', 'V2OV'):
    v = [T[n].get(i) for i in IDS]; v = np.array([np.nan if x is None else x for x in v], float)
    b = best(v, 1e-3); rows.append((n, *b) if b else (n, None, 0, 0))
for k in range(3):
    v = np.array([np.nan if (T['PADEC'][i] is None or T['PADEC'][i][k] is None) else T['PADEC'][i][k] for i in IDS], float)
    b = best(v, 1e-3); rows.append((f'PADEC[{k}]', *b))
for nm in ('R17DEV', 'R17ENS'):
    L = len(next(v for v in T[nm].values() if v))
    for k in range(L):
        v = np.array([np.nan if (T[nm][i] is None or T[nm][i][k] is None) else T[nm][i][k] for i in IDS], float)
        b = best(v, 1e-3); rows.append((f'{nm}[{k}]', *b))
for k, c in enumerate(T['FX_COLS']):
    v = np.array([np.nan if T['FX'][i][k] is None else T['FX'][i][k] for i in IDS], float)
    b = best(v, 2e-3); rows.append((f'FX[{k}] {c}', *b) if b else (f'FX[{k}] {c}', None, 0, 0))
R = pd.DataFrame(rows, columns=['tabela', 'zrodlo', 'zgodne', 'n'])
R['frac'] = (R.zgodne / R.n.replace(0, np.nan)).round(3)
R.to_csv('/home/claude/c3/provenance.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(R.to_string(index=False))
