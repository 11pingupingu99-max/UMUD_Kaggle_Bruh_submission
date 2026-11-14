"""Prototyp reguly wyboru pary rozciegien z fragmentow peczkow: para R12B albo v2 (gdy MT roznia sie > 2 mm).
Wybieramy najwezsze pasmo, ktore zawiera >= 50% fragmentow lezacych w ktorymkolwiek z pasm (miesien z peczkami)."""
import numpy as np, pandas as pd
apo = pd.read_csv('/home/claude/r80o_in/x/export_d81469c478/test_apo_r80.csv.gz')
feat = pd.read_csv('/home/claude/r76t/dl/test_features.csv').set_index('image_id')
fr = pd.read_csv('/home/claude/r71t/real/S/test_frags_r22_t60.csv.gz')
F = {i: g for i, g in fr.groupby('image_id')}
A = {(r.image_id, r.cfg): r for r in apo.itertuples(index=False)}
rows = []
for i in feat.index:
    r12, v2 = A.get((i, 'r12b')), A.get((i, 'v2'))
    ppm = feat.loc[i, 'px_per_mm']; w = feat.loc[i, 'w']
    if v2 is None:
        continue
    g = F.get(i)
    xm = (0.5 * (g.x0 + g.x1)).values if g is not None else np.array([])
    ym = (g.a * xm + g.b).values if g is not None else np.array([])
    def inside(p):
        if p is None or not len(xm):
            return np.zeros(len(xm), bool)
        return (ym >= p.u_bottom_a * xm + p.u_bottom_b) & (ym <= p.l_top_a * xm + p.l_top_b)
    def mt(p):
        if p is None:
            return np.nan
        x = 0.5 * w
        return ((p.l_top_a * x + p.l_top_b) - (p.u_bottom_a * x + p.u_bottom_b)) / ppm
    m12, mv = mt(r12), mt(v2)
    in12, inv = inside(r12), inside(v2); anyin = in12 | inv
    n = int(anyin.sum())
    choice = 'r12b'
    if r12 is None:
        choice = 'v2 (brak R12B)'
    elif np.isfinite(m12) and np.isfinite(mv) and abs(m12 - mv) > 2 and n >= 3:
        cand = sorted([('r12b', m12, in12), ('v2', mv, inv)], key=lambda t: t[1])   # najwezsze pierwsze
        for name, m, ins in cand:
            if ins[anyin].mean() >= 0.5:
                choice = name; break
    rows.append(dict(image_id=i, mt_r12b=m12, mt_v2=mv, n_frag=n, f_in_r12b=in12[anyin].mean() if n else np.nan,
                     f_in_v2=inv[anyin].mean() if n else np.nan, choice=choice))
D = pd.DataFrame(rows).set_index('image_id')
ch = D[D.choice != 'r12b']
pd.set_option('display.width', 200)
print(ch.round(2).to_string())
dis = D[(np.abs(D.mt_r12b - D.mt_v2) > 2)]
print('\nobrazy z rozna para (|dMT| > 2 mm):', len(dis)); print(dis.round(2).to_string())
