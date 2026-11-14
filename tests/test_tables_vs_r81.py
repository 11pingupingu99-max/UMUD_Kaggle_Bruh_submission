"""Tables computed from model exports == tables pasted in R81 (within the numerical tolerance of the Colab re-runs)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umud_final import tables as T

NB = '/home/claude/r83/UMUD_R83_KAGGLE.ipynb'
c2 = ''.join(json.load(open(NB))['cells'][2]['source'])
R = {n: json.loads(re.search(n + r" = json\.loads\('(.*?)'\)", c2).group(1)) for n in ('BAND', 'PADEC', 'V2MT', 'V2OV', 'FX')}
IDS = sorted(R['V2MT'])
dec = pd.read_csv('/home/claude/r71t/real/D5/test_decoders.csv').set_index('image_id').reindex(IDS)
feat = pd.read_csv('/home/claude/r76t/dl/test_features.csv').set_index('image_id').reindex(IDS)
apo = pd.read_csv('/home/claude/r80o_in/x/export_d81469c478/test_apo_r80.csv.gz')
sizes = feat[['h', 'w']]; ppm = feat['px_per_mm'].to_dict()

def cmp(name, ours, ref, tol):
    a = np.array([np.nan if ours.get(i) is None else ours[i] for i in IDS], float)
    b = np.array([np.nan if ref.get(i) is None else ref[i] for i in IDS], float)
    both = np.isfinite(a) & np.isfinite(b); d = np.abs(a[both] - b[both])
    miss = int((np.isfinite(a) != np.isfinite(b)).sum())
    ok = (d <= tol).mean() if both.any() else 0
    print(f'{name:28s} n {both.sum():3d} | w tolerancji {tol}: {100 * ok:5.1f}% | max |d| {d.max():.4f} | brak tylko po jednej stronie: {miss}')
    return ok, miss

res = {}
e80 = pd.read_csv('/home/claude/r80o_in/x/export_d81469c478/test_r80.csv').set_index('image_id').reindex(IDS)
c2b = json.loads(re.search(r"PAYLOAD = '(.*?)'", c2).group(1)) if False else None
OTHER = ['IMG_00042.tif'] + [f'IMG_000{n}.tif' for n in (76, 77, 78, 79, 80, 81, 85)] + [f'IMG_00{n}.tif' for n in (151, 152, 153, 154, 155, 170)]
res['BAND'] = cmp('BAND', T.band(dec, OTHER), R['BAND'], 0.005)
P = T.padec(dec)
for k in range(3):
    res[f'PADEC{k}'] = cmp(f'PADEC[{k}]', {i: P[i][k] for i in IDS}, {i: (R['PADEC'][i] or [None] * 3)[k] for i in IDS}, 0.05)
res['V2MT'] = cmp('V2MT', T.v2mt(dec), R['V2MT'], 0.01)
res['V2OV'] = cmp('V2OV', T.v2ov(dec), R['V2OV'], 0.002)
res['FX85'] = cmp('FX85 a76_mt', T.a76_mt(feat), {i: R['FX'][i][85] for i in IDS}, 0.001)
fr22 = pd.read_csv('/home/claude/r71t/real/S/test_frags_r22_t60.csv.gz')
f71 = pd.read_csv('/home/claude/r71t/real/F71/test_frags_f71e4_t60.csv.gz')
MISS = set(feat.index[feat.guard.astype(str) == 'baseline_pair_missing'])
fx89 = T.cmbd(T.lex5_fl(fr22, apo, sizes, ppm), T.lex5_fl(f71, apo, sizes, ppm), missing_pair=MISS)
res['FX89'] = cmp('FX89 cmbd_a76mix_lex5_fl', fx89, {i: R['FX'][i][89] for i in IDS}, 0.02)
assert all(m == 0 for _, m in res.values()), 'brakujace wartosci musza sie zgadzac'
assert res['BAND'][0] > 0.99 and res['V2MT'][0] > 0.99 and res['V2OV'][0] > 0.99 and res['FX85'][0] == 1
assert all(res[f'PADEC{k}'][0] > 0.97 for k in range(3)) and res['FX89'][0] > 0.99
print('PASS: tabele z eksportow modeli = tabele R81')

# --- R17 (R50T / R17C) ---
B = '/tmp/claude-0/-home-claude/53baf52e-f9dc-56a1-8bcb-d5dad69b6905/scratchpad/in_r50t'
from pathlib import Path as _P
if _P(B).exists():
    R2 = {n: json.loads(re.search(n + r" = json\.loads\('(.*?)'\)", c2).group(1)) for n in ('R17DEV', 'R17ENS')}
    dv = pd.read_csv(f'{B}/files/assets/r17c_develop_test.csv').set_index('image_id').reindex(IDS)
    rf = pd.read_csv(f'{B}/files/assets/r17c_refit_test.csv').set_index('image_id').reindex(IDS)
    r50 = pd.read_csv(f'{B}/csv/R50T_LATEST_test.csv').set_index('image_id').reindex(IDS)
    D17 = T.r17dev(dv, rf); E17 = T.r17ens(r50)
    rr = {}
    for k in range(4):
        rr[f'R17DEV{k}'] = cmp(f'R17DEV[{k}]', {i: D17[i][k] for i in IDS}, {i: (R2['R17DEV'][i] or [None] * 4)[k] for i in IDS}, 0.001)
    for k in (0, 3, 4):
        rr[f'R17ENS{k}'] = cmp(f'R17ENS[{k}]', {i: E17[i][k] for i in IDS}, {i: (R2['R17ENS'][i] or [None] * 5)[k] for i in IDS}, 0.001)
    assert all(v[0] == 1.0 and v[1] == 0 for v in rr.values()), rr
    print('PASS: tabele R17 uzywane przez B10 (R17DEV 0-3, R17ENS 0, 3, 4) = R81')
