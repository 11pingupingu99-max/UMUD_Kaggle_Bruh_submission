"""Faza C: baza odtwarzalna (PA/FL/MT kazdego obrazu z regul i eksportow modeli) zamiast payloadu R23A.
Reguly v1: para rozciegien = najwezsze pasmo z >= 50% fragmentow (R12B vs v2, gdy MT rozne o > 2 mm; brak R12B -> v2);
MT = srodek-srodek R12B - 1,3629 (para R12B) albo MT v2 (para v2); PA = mediana dekoderow R22 z wybrana para;
FL = R17 refit (para R12B, status OK) | lex5 z para v2 (obraz dwuwarstwowy) | geometria Q_g*MT/sin PA (brak pary R12B / R17).
Klip 151-155 (srodkowe rozciegno) i PA bazy PNG (R21: r20_geometry na mapach R17 refit) - czekaja na C2: zostaja R23A."""
import sys, json, re, math
sys.path.insert(0, '/home/claude/r77')
import numpy as np, pandas as pd, nbrun
B = sys.argv[1]
R83 = '/home/claude/r83/UMUD_R83_KAGGLE.ipynb'
B10 = json.load(open('/home/claude/r82/probes_AU.json'))['B10']
s, g, _ = nbrun.run(R83, B10)
base = g['base'].set_index('image_id'); IDS = list(base.index); G = pd.Series(np.asarray(g['GROUP']), index=list(s.image_id))
d5 = pd.read_csv('/home/claude/r71t/real/D5/test_decoders.csv').set_index('image_id').reindex(IDS)
e80 = pd.read_csv('/home/claude/r80o_in/x/export_d81469c478/test_r80.csv').set_index('image_id').reindex(IDS)
apo = pd.read_csv('/home/claude/r80o_in/x/export_d81469c478/test_apo_r80.csv.gz')
feat = pd.read_csv('/home/claude/r76t/dl/test_features.csv').set_index('image_id').reindex(IDS)
rf = pd.read_csv(f'{B}/files/assets/r17c_refit_test.csv').set_index('image_id').reindex(IDS)
fr = pd.read_csv('/home/claude/r71t/real/S/test_frags_r22_t60.csv.gz'); F = {i: q for i, q in fr.groupby('image_id')}
A = {(r.image_id, r.cfg): r for r in apo.itertuples(index=False)}
LAYER = pd.read_csv('/home/claude/r82/test_fl_r82.csv').set_index('image_id')['layer']
PEND = [f'IMG_00{n}.tif' for n in range(151, 156)]
R12DEC = ('r22_r12b_L_pa_r17_median', 'r22_r12b_L_pa_local_support_baseline', 'r22_r12b_L_pa_local_median')
V2DEC = ('r22_v2_L_pa_r17_median', 'r22_v2_L_pa_local_support_baseline', 'r22_v2_L_pa_local_median')

def pair_choice(i):
    r12, v2 = A.get((i, 'r12b')), A.get((i, 'v2'))
    if r12 is None:
        return 'v2' if v2 is not None else 'none'
    if v2 is None:
        return 'r12b'
    ppm, w = feat.loc[i, 'px_per_mm'], feat.loc[i, 'w']; x = 0.5 * w
    mt = lambda p: ((p.l_top_a * x + p.l_top_b) - (p.u_bottom_a * x + p.u_bottom_b)) / ppm
    if abs(mt(r12) - mt(v2)) <= 2:
        return 'r12b'
    q = F.get(i)
    if q is None or len(q) < 3:
        return 'r12b'
    xm = (0.5 * (q.x0 + q.x1)).values; ym = (q.a * xm + q.b).values
    ins = lambda p: (ym >= p.u_bottom_a * xm + p.u_bottom_b) & (ym <= p.l_top_a * xm + p.l_top_b)
    i12, iv = ins(r12), ins(v2); anyin = i12 | iv
    if anyin.sum() < 3:
        return 'r12b'
    for name, m, inn in sorted([('r12b', mt(r12), i12), ('v2', mt(v2), iv)], key=lambda t: t[1]):
        if inn[anyin].mean() >= 0.5:
            return name
    return 'r12b'

rows = []
for i in IDS:
    pr = pair_choice(i)
    mt = (e80.loc[i, 'r12b_mt_center_mean_mm'] - 1.3629) if pr == 'r12b' else d5.loc[i, 'r22_v2_mt_inner_mean_mm']
    dec = d5.loc[i, list(R12DEC if pr == 'r12b' else V2DEC)].values.astype(float)
    if not np.isfinite(dec).any():
        dec = d5.loc[i, list(V2DEC)].values.astype(float)
    # PA bazy = dekoder local_support (tak jak R23A dla TIFF: pa_med glosuje R23A + r17_median + local_median);
    # gdy go brak - mediana dostepnych dekoderow z wybrana para
    pa = float(dec[1]) if np.isfinite(dec[1]) else (float(np.nanmedian(dec)) if np.isfinite(dec).any() else np.nan)
    if i.endswith('.png'):
        pa = float(base.loc[i, 'pa_deg'])   # PNG: R23A = R21 (fragmenty wspierane R17 refit) - kod r20_geometry odtworzy C2
    two_layer = pr == 'v2' and A.get((i, 'r12b')) is not None
    if pr == 'r12b' and rf.loc[i, 'status'] == 'OK':
        fl, src = rf.loc[i, 'fl_mm'], 'r17_refit'
    elif two_layer:
        fl, src = LAYER[i], 'lex5_v2'
    else:
        fl, src = np.nan, 'geometria'
    rows.append(dict(image_id=i, grp=G[i], pair=pr, mt=mt, pa=pa, fl=fl, fl_src=src, dec=list(np.round(dec, 4))))
D = pd.DataFrame(rows).set_index('image_id')
# geometria dla brakujacych FL: Q_g = mediana q = FL sin PA / MT w grupie (obrazy z FL R17)
D['q'] = D.fl * np.sin(np.radians(D.pa)) / D.mt
for gg in D.grp.unique():
    m = (D.grp == gg); Qg = float(np.nanmedian(D.loc[m & (D.fl_src == 'r17_refit'), 'q']))
    mm = m & (D.fl_src == 'geometria')
    D.loc[mm, 'fl'] = Qg * D.loc[mm, 'mt'] / np.sin(np.radians(D.loc[mm, 'pa']))
for i in PEND:
    D.loc[i, ['pa', 'fl', 'mt']] = base.loc[i, ['pa_deg', 'fl_mm', 'mt_mm']].values; D.loc[i, 'fl_src'] = 'R23A (C2)'
assert np.isfinite(D[['pa', 'fl', 'mt']].values).all()
cmp = D[['pa', 'fl', 'mt']].values - base.loc[IDS, ['pa_deg', 'fl_mm', 'mt_mm']].values
D['dPA'], D['dFL'], D['dMT'] = cmp[:, 0], cmp[:, 1], cmp[:, 2]
print(D.pair.value_counts().to_dict(), D.fl_src.value_counts().to_dict())
print('|d| wzgledem R23A (mediana / q90 / max): PA', np.round(np.quantile(np.abs(D.dPA), [.5, .9, 1]), 3),
      'FL', np.round(np.quantile(np.abs(D.dFL), [.5, .9, 1]), 3), 'MT', np.round(np.quantile(np.abs(D.dMT), [.5, .9, 1]), 3))
big = D[(np.abs(D.dPA) > 1) | (np.abs(D.dFL) > 3) | (np.abs(D.dMT) > 0.5)]
pd.set_option('display.width', 220)
print(big[['grp', 'pair', 'fl_src', 'pa', 'dPA', 'fl', 'dFL', 'mt', 'dMT']].round(2).to_string())
out = {'BASE': {i: [round(float(r.pa), 4), round(float(r.fl), 3), round(float(r.mt), 4)] for i, r in D.iterrows()},
       'PADEC_V2': {i: [None if not np.isfinite(x) else float(x) for x in r.dec] for i, r in D.iterrows() if r.pair == 'v2'},
       'FX89': {i: round(float(LAYER[i]), 2) for i, r in D.iterrows() if r.pair == 'v2' and r.fl_src == 'lex5_v2'},
       'OTHER_PAIR': [i for i, r in D.iterrows() if r.pair == 'v2'] + PEND}
json.dump(out, open('/home/claude/c3/cbase.json', 'w'))
D.to_csv('/home/claude/c3/cbase_table.csv')
print('zapisano cbase.json; PADEC_V2', len(out['PADEC_V2']), 'FX89', len(out['FX89']), 'OTHER_PAIR', len(out['OTHER_PAIR']))
