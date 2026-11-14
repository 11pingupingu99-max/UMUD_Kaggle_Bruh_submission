"""R86T: export of ALL aponeurosis components (not only the selected pair) of the R12B and v2 maps, so that the pair
can be chosen offline by a rule (the layer that contains the fascicles; e.g. the middle aponeurosis in two-muscle images).
Each component: top / center / bottom line fits (y = a*x + b, native pixels), x-range, width, area, mean depth.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import numpy as np
import r45_core

LINE = ('a', 'b', 'x0', 'x1', 'residual', 'angle')


def components_rows(image_id, cfg, prob, thr=.5):
    mask = np.asarray(prob) >= thr
    _, comps = r45_core.aponeurosis_components(mask)
    pair = r45_core.aponeurosis_lines(mask)
    sel = set()
    if pair is not None:
        sel = {pair['upper']['label'], pair['lower']['label']}
    rows = []
    for k, c in enumerate(sorted(comps, key=lambda c: c['mid'])):
        r = dict(image_id=image_id, cfg=cfg, comp=k, label=c['label'], mid=c['mid'], width=c['width'], area=c['area'],
                 in_default_pair=c['label'] in sel, h=int(mask.shape[0]), w=int(mask.shape[1]))
        for part in ('top', 'center', 'bottom'):
            z = c.get(part) or {}
            for key in LINE:
                r[f'{part}_{key}'] = z.get(key)
        rows.append(r)
    if not rows:
        rows.append(dict(image_id=image_id, cfg=cfg, comp=-1, h=int(mask.shape[0]), w=int(mask.shape[1])))
    return rows
