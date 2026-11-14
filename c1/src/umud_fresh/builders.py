"""Recovered table builders. Inputs are model measurements, never a saved submission.

The lost R94 fallback builder is reconstructed from PROVENANCE.md. Its rules are
documented here and compared separately against R96; identity is NOT assumed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from umud_final import tables
from .metadata import sequence_blocks

PA_KEYS = ("pa_r17_median", "pa_local_support_baseline", "pa_local_median")
FX_NAMES = {72: "pa_med6_r12b", 85: "a76_safe_mt_inner_mean_mm", 89: "cmbd_a76mix_lex5_fl",
            110: "mean3_a76mix_lex5_fl", 111: "f86_a76mix_lex5_fl",
            130: "mean3_nontelemed", 132: "folds_late_mean3_nontelemed"}


def positive(x):
    a = np.asarray(x, float)
    return np.isfinite(a) & (a > 0)


def geometric(values, require_all=True):
    a = np.asarray(values, float)
    ok = positive(a)
    if not ok.any() or (require_all and not ok.all()):
        return np.nan
    return float(np.exp(np.log(a[ok]).mean()))


def finite_round(v, nd):
    return round(float(v), nd) if v is not None and np.isfinite(v) else None


def _column(frame, name):
    return pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)


def _calibrate(candidates, target, groups, log=False):
    """Choose an offset-corrected decoder by agreement with available PRIMARY model outputs.

    No ground truth or leaderboard score enters this step. Offset = median difference; score = MAE after offset;
    tie breaking follows the candidate order. A group with <5 paired values uses the
    full batch's fit. Insufficient global support leaves that candidate unavailable.
    """
    target = np.asarray(target, float); groups = np.asarray(groups)
    out = np.full(len(target), np.nan); source = np.full(len(target), "", object); fits = {}
    for gg in np.unique(groups):
        mask = groups == gg; ranked = []
        for name, values in candidates.items():
            a = np.asarray(values, float)
            ok = positive(a) & positive(target)
            if (ok & mask).sum() >= 5:
                ok &= mask; scope = str(gg)
            else:
                scope = "whole_batch"
            if ok.sum() < 5:
                continue
            delta = np.log(a[ok] / target[ok]) if log else a[ok] - target[ok]
            shift = float(np.median(delta)); residual = float(np.mean(np.abs(delta - shift)))
            fitted = a * np.exp(-shift) if log else a - shift
            ranked.append((residual, name, fitted, shift, scope))
        ranked.sort(key=lambda a: a[0])
        fits[str(gg)] = []
        for residual, name, fitted, shift, scope in ranked:
            use = mask & ~positive(out) & positive(fitted)
            out[use] = fitted[use]; source[use] = name + "_cal"
            fits[str(gg)].append({"source": name, "shift": shift, "residual": residual, "scope": scope})
    return out, source, fits


def build_base(meta, dec, r20, r20two, mt, r17_refit):
    ids = meta.image_id.tolist(); mm = meta.set_index("image_id").loc[ids]
    ds = [d.reindex(ids) for d in (dec, r20, r20two, mt, r17_refit)]
    dec, r20, r20two, mt, r17_refit = ds
    groups = mm.group.to_numpy(); is_png = mm.is_png.to_numpy(bool)
    pa = np.full(len(ids), np.nan); fl = pa.copy(); thickness = pa.copy()
    src_pa = np.full(len(ids), "", object); src_fl = src_pa.copy(); src_mt = src_pa.copy()
    def take(dest, sources, values, label, mask=None):
        a = np.asarray(values, float); use = ~positive(dest) & positive(a)
        if mask is not None:
            use &= mask
        dest[use] = a[use]; sources[use] = label
    take(pa, src_pa, _column(dec, "r22_r12b_L_pa_local_support_baseline"), "r22_late", ~is_png)
    take(pa, src_pa, _column(r20, "pa_local_support_baseline"), "r20_support")
    take(pa, src_pa, _column(r20two, "pa_local_support_baseline"), "r20_support_mtpair")
    take(fl, src_fl, _column(r17_refit, "fl_mm"), "r17_refit",
         (r17_refit.status == "OK").to_numpy() if "status" in r17_refit else None)
    take(fl, src_fl, _column(r20two, "fl_mm"), "r17_refit_mtpair")
    # R12B vertical_mean: two largest connected components, not the R20 pair selector.
    take(thickness, src_mt, _column(mt, "mt_r12c_mm"), "r12b_rule")
    v2 = _column(dec, "r22_v2_mt_inner_mean_mm").to_numpy()
    ok = positive(thickness) & positive(v2)
    mt_shift = float(np.median(v2[ok] - thickness[ok])) if ok.sum() >= 5 else np.nan
    take(thickness, src_mt, v2 - mt_shift, "v2_inner_cal")
    primary_pa = pa.copy()
    pa_candidates = {"r20_all": _column(r20, "pa_r17_median")}
    for label, key in (("v2_r17med", "pa_r17_median"), ("v2_support", "pa_local_support_baseline"),
                       ("v2_local", "pa_local_median")):
        pa_candidates[label] = _column(dec, f"r22_v2_L_{key}")
    fallback, labels, pa_fit = _calibrate(pa_candidates, primary_pa, groups)
    use = ~positive(pa) & positive(fallback); pa[use] = fallback[use]; src_pa[use] = labels[use]
    # Group shape FL*sin(PA)/MT is another image-derived fallback, not a label.
    geom_fl = np.full(len(ids), np.nan)
    q = fl * np.sin(np.deg2rad(pa)) / thickness
    for gg in np.unique(groups):
        g = groups == gg; okq = g & positive(q)
        if okq.sum() < 5:
            okq = positive(q)
        if okq.sum() >= 5:
            geom_fl[g] = np.median(q[okq]) * thickness[g] / np.sin(np.deg2rad(pa[g]))
    fl_candidates = {"v2_fl": _column(dec, "r22_v2_L_fl_mm"), "geometry": geom_fl}
    fallback_fl, labels_fl, fl_fit = _calibrate(fl_candidates, fl, groups, log=False)
    use = ~positive(fl) & positive(fallback_fl); fl[use] = fallback_fl[use]; src_fl[use] = labels_fl[use]
    base = pd.DataFrame({"image_id": ids, "pa_deg": pa, "fl_mm": fl, "mt_mm": thickness})
    audit = base.assign(group=groups, src_pa=src_pa, src_fl=src_fl, src_mt=src_mt)
    if not positive(base[["pa_deg", "fl_mm", "mt_mm"]].to_numpy()).all():
        bad = audit[~positive(base[["pa_deg", "fl_mm", "mt_mm"]].to_numpy()).all(1)]
        raise ValueError(f"No reliable model-derived fallback for {len(bad)} images: {bad.to_dict('records')}")
    return base, audit, {"pa": pa_fit, "fl": fl_fit, "mt_v2_shift_mm": mt_shift}


def pa_med6(dec, base, groups, padec=None):
    """Actual FX72 contract recovered against all 309 saved values.

    TIFF: base + two R22 decoders (r17/local) + all three F71 decoders = 6 votes.
    PNG: base + R22 r17/support + F71 r17/support = 5 votes despite the historical name.
    Each vote is median-offset aligned to the base within DEVICE group. If any selected
    F71 vote is missing, use base. R22 votes may be missing (nanmedian). In fresh mode
    both the base and the R22 votes are computed again, not read from the old tables.
    """
    ids = base.image_id.tolist(); b = base.set_index("image_id").loc[ids, "pa_deg"]
    groups = pd.Series(np.asarray(groups), index=ids)
    padec = tables.padec(dec) if padec is None else padec
    out = b.copy()
    for gg in groups.unique():
        ix = groups.index[groups == gg]; rsel = (0, 1) if gg == "png" else (0, 2)
        fsel = (0, 1) if gg == "png" else (0, 1, 2)
        cols = {"base": b.loc[ix]}
        for j in rsel:
            cols[f"r{j}"] = pd.Series({i: (padec.get(i) or [None] * 3)[j] for i in ix}, dtype=float)
        for j in fsel:
            cols[f"f{j}"] = _column(dec, f"f71_r12b_L_{PA_KEYS[j]}").reindex(ix)
        v = pd.DataFrame(cols).astype(float)
        v = v - v.sub(b.loc[ix], axis=0).median(axis=0)
        good = v[[f"f{j}" for j in fsel]].notna().all(axis=1)
        med = v.median(axis=1)
        out.loc[ix[good]] = med.loc[ix[good]]
    return {i: finite_round(v, 4) for i, v in out.items()}


def build_fx(meta, dec, apo, feat, fragments, base):
    ids = meta.image_id.tolist(); mm = meta.set_index("image_id").loc[ids]
    dec = dec.reindex(ids); feat = feat.reindex(ids)
    ppm = mm.px_per_mm.to_dict(); sizes = mm[["h", "w"]]
    missing = set(ids) - set(apo.loc[apo.cfg == "r12b", "image_id"])
    native = {}
    for tag, fr in fragments.items():
        native[tag] = tables.lex5_fl(fr, apo, sizes, ppm, cfg="a76_v2mix")
    def vals(tag):
        return [native.get(tag, {}).get(i) for i in ids]
    g22 = np.array(vals("r22avg"), float); g71 = np.array(vals("f71avg"), float)
    f86 = np.array(vals("f86avg"), float)
    f86[np.array([i in missing for i in ids])] = np.nan
    c89 = np.sqrt(g22 * g71)
    c89[np.array([i in missing for i in ids])] = np.nan
    per22 = [f"r22k{k}" for k in range(5)]
    per71 = [f"f71k{k}" for k in (0, 2, 3, 4)]
    # R96 uses available folds when a fragment list is empty, like pandas.mean(skipna=True).
    pf22 = np.array([geometric([native.get(t, {}).get(i) for t in per22], require_all=False) for i in ids])
    pf71 = np.array([geometric([native.get(t, {}).get(i) for t in per71], require_all=False) for i in ids])
    c89pf = np.sqrt(pf22 * pf71)
    c110 = np.cbrt(c89 * c89 * f86)
    c132 = np.cbrt(c89pf * c89pf * f86)
    c132[~positive(c89)] = np.nan
    tel = mm.group.isin(["telr", "teln"]).to_numpy()
    c130 = np.where(tel, c89, c110); c132 = np.where(tel, c89, c132)
    pa6 = pa_med6(dec, base, mm.group)
    med6 = np.array([pa6[i] for i in ids])
    a76 = tables.a76_mt(feat)
    fx_cols = [f"unused_{j}" for j in range(134)]
    for j, name in FX_NAMES.items():
        fx_cols[j] = name
    data = {72: med6, 85: [a76.get(i) for i in ids], 89: c89, 110: c110, 111: f86, 130: c130, 132: c132}
    fx = {}
    for k, i in enumerate(ids):
        row = [None] * len(fx_cols)
        for j, values in data.items():
            row[j] = finite_round(values[k], 4 if j in (72, 85) else 2)
        fx[i] = row
    detail = pd.DataFrame({"image_id": ids, "c89": c89, "c89pf": c89pf, "f86": f86, "c110": c110,
                           "FX130": c130, "FX132": c132, "PA6": med6})
    return fx_cols, fx, detail


def build_inputs(meta, dec, apo, feat, fragments, r20, r20two, mt, r17, r50):
    dec = dec.set_index("image_id") if "image_id" in dec else dec
    feat = feat.set_index("image_id") if "image_id" in feat else feat
    def index(d):
        return d.set_index("image_id") if "image_id" in d else d
    r20, r20two, mt, r50 = [index(d) for d in (r20, r20two, mt, r50)]
    r17 = {k: index(d) for k, d in r17.items()}
    base, audit, calibration = build_base(meta, dec, r20, r20two, mt, r17["refit"])
    # Same pair as R12B MT: baseline code originally masked |centre MT difference|>1mm.
    centre = _column(dec, "r22_r12b_L_mt_center_mm")
    reference = _column(mt, "vertical_mean_mm")
    other_pair = set(centre.index[(centre - reference).abs() > 1.])
    fx_cols, fx, fx_detail = build_fx(meta, dec, apo, feat, fragments, base)
    ids = base.image_id; mm = meta.set_index("image_id").loc[ids]
    inputs = {"base": base, "BAND": tables.band(dec, other_pair), "PADEC": tables.padec(dec),
              "V2MT": tables.v2mt(dec), "V2OV": tables.v2ov(dec),
              "R17DEV": tables.r17dev(r17["develop"], r17["refit"]), "R17ENS": tables.r17ens(r50),
              "FX_COLS": fx_cols, "FX": fx, "PNG_MUSCLE": mm.muscle.to_dict(),
              "GROUP": mm.group.to_numpy(), "GRYSUB": mm.grysub.to_numpy(),
              "IS_PNG": mm.is_png.to_numpy(bool), "BID": sequence_blocks(base, meta),
              "FL22": {}, "OR_INS": {}}
    return inputs, audit, calibration, fx_detail
