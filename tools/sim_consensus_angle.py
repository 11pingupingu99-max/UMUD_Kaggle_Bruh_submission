"""R84: symulator protokolu (trening DL_Track, slady GT) - nowe estymatory FL: kat konsensusu zamiast kata fragmentu."""
import sys, math
sys.path.insert(0, '/home/claude/r71t/src')
import numpy as np, pandas as pd
import r71_geom as G
S = '/home/claude/r71t/real/S'
im = pd.read_csv(f'{S}/train_images.csv'); ap = pd.read_csv(f'{S}/train_apo.csv.gz', float_precision='round_trip')
gt = pd.read_csv(f'{S}/train_gt.csv.gz', float_precision='round_trip')
print('cfg w train_apo:', ap.cfg.value_counts().to_dict())
MODELS = {'r22': f'{S}/train_frags_r22oof_t60.csv.gz'}
import glob
f71 = sorted(glob.glob('/home/claude/r71t/real/F71/train_fold*_frags_f71oof_t60.csv.gz'))
if f71:
    MODELS['f71'] = f71

def load_frags(p):
    if isinstance(p, list):
        return pd.concat([pd.read_csv(x, float_precision='round_trip') for x in p], ignore_index=True)
    return pd.read_csv(p, float_precision='round_trip')

def fl_with_angle(f, ang, au, bu, al, bl):
    xm = .5 * (f['x0'] + f['x1']); ym = f['a'] * xm + f['b']
    af = math.tan(math.radians(ang)); bf = ym - af * xm
    pts = []
    for a2, b2 in ((au, bu), (al, bl)):
        den = af - a2
        if abs(den) < 1e-6:
            return np.nan
        x = (b2 - bf) / den; pts.append((x, af * x + bf))
    return float(np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))

def table(fr, cfg='r12b', k=5):
    apd = {r['image_id']: r for r in ap[ap.cfg == cfg].to_dict('records')}
    frd = {i: g.to_dict('records') for i, g in fr.groupby('image_id')}
    gtd = {i: g.to_dict('records') for i, g in gt.groupby('image_id')}
    out = []
    for r in im.itertuples(index=False):
        a = apd.get(r.image_id)
        if a is None or r.status != 'OK' or r.image_id not in gtd or r.image_id not in frd:
            continue
        pg = G.protocol_fragments(gtd[r.image_id], a, r.w, r.h, 'angle')
        pf = G.protocol_fragments(frd[r.image_id], a, r.w, r.h, 'angle_fit')
        if len(pg) < 3 or len(pf) < 3:
            continue
        d = {'image_id': r.image_id, 'n_gt': len(pg), 'n_fr': len(pf)}
        for kk in (1, 3, 5):
            s = G.lex_select(pg, kk); d[f'gt_lex{kk}'] = np.mean([f['fl_px'] for f in s])
        d['gt_all'] = np.median([f['fl_px'] for f in pg])
        au, bu, al, bl = a['u_bottom_a'], a['u_bottom_b'], a['l_top_a'], a['l_top_b']
        deep = math.degrees(math.atan(al))
        sel = G.lex_select(pf, k)
        d['lex5'] = np.mean([f['fl_px'] for f in sel])
        # consensus angles (absolute line angle, same convention as angle_fit)
        med_all = float(np.median([f['pa'] for f in pf]))
        coh = [f for f in pf if abs(f['pa'] - med_all) <= 8.0] or pf
        ang_c = float(np.median([f['angle_fit'] for f in coh]))
        ang_s = float(np.median([f['angle_fit'] for f in sel]))
        for lam in (0.25, 0.5, 0.75, 1.0):
            v = [fl_with_angle(f, (1 - lam) * f['angle_fit'] + lam * ang_c, au, bu, al, bl) for f in sel]
            d[f'lex5_cons{int(lam * 100)}'] = np.nanmean(v)
            v = [fl_with_angle(f, (1 - lam) * f['angle_fit'] + lam * ang_s, au, bu, al, bl) for f in sel]
            d[f'lex5_sel{int(lam * 100)}'] = np.nanmean(v)
        # all coherent fragments with consensus angle (positions of all fragments)
        d['allc_cons'] = np.nanmedian([fl_with_angle(f, ang_c, au, bu, al, bl) for f in coh])
        # weighted mean angle by fragment length
        wl = np.array([f['length'] for f in coh]); angs = np.array([f['angle_fit'] for f in coh])
        ang_w = float((wl * angs).sum() / wl.sum())
        d['lex5_consw'] = np.nanmean([fl_with_angle(f, ang_w, au, bu, al, bl) for f in sel])
        # larger k
        for kk in (3, 8, 10):
            s = G.lex_select(pf, kk); d[f'lex{kk}'] = np.mean([f['fl_px'] for f in s])
            d[f'lex{kk}_cons100'] = np.nanmean([fl_with_angle(f, ang_c, au, bu, al, bl) for f in s])
        d['lex5_med'] = np.median([f['fl_px'] for f in sel])
        out.append(d)
    return pd.DataFrame(out)

def score(T, target):
    y = T[target].values.astype(float); res = []
    for c in T.columns:
        if c.startswith(('gt_', 'image_id', 'n_')):
            continue
        v = T[c].values.astype(float); ok = np.isfinite(v) & np.isfinite(y) & (v > 0) & (y > 0)
        r = np.log(v[ok] / y[ok]); r = r - np.median(r)
        res.append((c, int(ok.sum()), float(np.mean(np.abs(r))), float(np.quantile(np.abs(r), .9))))
    return pd.DataFrame(res, columns=['feature', 'n', 'mae_rel', 'q90']).set_index('feature')

if __name__ == '__main__':
    allres = {}
    for m, p in MODELS.items():
        T = table(load_frags(p))
        T.to_csv(f'/home/claude/r84/sim_{m}.csv', index=False)
        print(f'\n== {m}: obrazow {len(T)}')
        R = pd.concat({t: score(T, t)['mae_rel'] for t in ('gt_lex3', 'gt_lex1', 'gt_lex5', 'gt_all')}, axis=1)
        print((R * 100).round(2).sort_values('gt_lex3').to_string())
