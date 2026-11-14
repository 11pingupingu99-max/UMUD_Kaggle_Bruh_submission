"""Numerical R96 operations. Recovered from the SHA-pinned final notebook.
No predictions, image-ID maps, file discovery or output side effects are embedded here.
GPL-3.0-or-later.
"""
import re
import numpy as np
import pandas as pd
for _name in ("BAND", "PADEC", "V2MT", "V2OV", "R17DEV", "R17ENS", "FX_COLS", "FX", "PNG_MUSCLE", "FL22", "OR_INS"):
    globals()[_name] = INPUTS[_name]
base = INPUTS["base"][["image_id", "pa_deg", "fl_mm", "mt_mm"]].reset_index(drop=True).copy()
assert len(base) > 0 and base.image_id.is_unique
assert np.isfinite(base[["pa_deg", "fl_mm", "mt_mm"]]).all().all()
GROUP = np.asarray(INPUTS["GROUP"], str)
dom = np.where(np.asarray(INPUTS["IS_PNG"], bool), "png", "tif")
GPIN = False
GXT_IN = None
GROUPS = ("tif", "png", "sie", "telr", "teln", "gry", "tel")
TOK = re.compile(r"(pa_fx|fl_fx|mt_fx|fxp|fxf|fxm|png_arpa|png_arfl|png_mus|gry_split|fl_gx|pa_gx|fl_guard|fl_phys|fl_seq|pa_seq|mt_band|arpa|arfl|pa_med|pa_pg|pa_or|fl_pg|fl_med|fl_k|fl_mt|fl_f|fl_s|pa_d|pa_s|pa_f|pa_g|mt_d|mt_f|mtvo|mtva|mtv|fl_dev|pa_dev|fl_ens|fl_enq|pa_ens|pa_mix|mt_pair)"
                 r"([+-]?\d+(?:\.\d+)?)(?:@(tif|png|sie|telr|teln|gry|tel|pgm|pvl|prf|pxx|grya|gryb))?")
def parse(v):
    v = v.replace(" ", "")
    if v == "base":
        return []
    out, pos = [], 0
    while pos < len(v):
        m = TOK.match(v, pos)
        assert m, f"niepoprawny wariant od znaku {pos}: {v[pos:]!r}"
        out.append((m.group(1), float(m.group(2)), m.group(3)))
        pos = m.end()
        if pos < len(v):
            assert v[pos] == "+", f"oczekiwany '+' w {v!r}"
            pos += 1
            assert pos < len(v), f"pusty czlon po '+' w {v!r}"
    kinds = [(k, d) for k, _, d in out]
    assert len(kinds) == len(set(kinds)), "ten sam czlon podany dwa razy"
    return out

ops = parse(VARIANT)
sub = base.copy()
MT_OFFSET_R12C = 1.3629          # MT w R23A = MT surowe (srodek-srodek) - 1.3629 mm
GUARD_PA_SHIFT = 0.58            # guard i fl_phys licza MT/sin(PA) zawsze z PA konfiguracji B (R23A + 0.58 st)
PA_G = base.pa_deg.values + GUARD_PA_SHIFT
if GPIN:   # R83: guard z PA bazy sprzed ebase* -> te same obrazy przyciete co bez podmiany
    PA_G = BASE_PA0 + GUARD_PA_SHIFT
    print('gpin: guard liczy MT/sin(PA) z PA bazy sprzed podmiany')
PA_RAW = base.pa_deg.values.copy()
PA_SRC = PA_RAW.copy()   # start toru PA (pa_med moze go zmienic); FL i guard licza zawsze z PA_RAW
IS_PNG = dom == "png"

BID = np.asarray(INPUTS["BID"], int)
GRYSUB = np.asarray(INPUTS["GRYSUB"], str)
MUS = np.array([("png_" + {"gm":"gm", "vl":"vl", "rf":"rf"}.get(PNG_MUSCLE.get(i, "?"), "x")) if d == "png" else ""
                for i, d in zip(base.image_id, dom)])
_sw = {k: v for k, v, _ in ops if k in ("png_mus", "gry_split")}
for k, v in _sw.items():
    assert v in (0.0, 1.0), f"{k}: 0 albo 1"
MUS_ON, GRY_ON = _sw.get("png_mus", 0) == 1, _sw.get("gry_split", 0) == 1
GROUP2 = GROUP.copy().astype(object)
if MUS_ON:
    GROUP2[IS_PNG] = MUS[IS_PNG]
if GRY_ON:
    GROUP2[GROUP == "gry"] = GRYSUB[GROUP == "gry"]
GROUP2 = GROUP2.astype(str)
print("podgrupy:", pd.Series(GROUP2).value_counts().to_dict())
PA_FOR_FL = None   # pa_pg zapisuje tu PA sprzed swojej zmiany, zeby fl_pg liczyl geometrie z tych samych danych
def m_of(d):
    if d is None:
        return np.ones(len(sub), bool)
    if d in ("tif", "png"):
        return dom == d
    if d == "tel":
        return np.isin(GROUP, ("telr", "teln"))
    if d in ("pgm", "pvl", "prf", "pxx"):
        return MUS == {"pgm": "png_gm", "pvl": "png_vl", "prf": "png_rf", "pxx": "png_x"}[d]
    if d in ("grya", "gryb"):
        return GRYSUB == {"grya": "gry_a", "gryb": "gry_b"}[d]
    return GROUP == d
def seq_shrink(col, w, mask):
    s = pd.Series(sub[col].values)
    med = s.groupby(BID).transform("median").values
    ok = mask & (BID >= 0)
    sub.loc[ok, col] = med[ok] + w * (sub.loc[ok, col].values - med[ok])
    return int(ok.sum())

fl_scale = [o for o in ops if o[0] in ("fl_k", "fl_mt")]
_cover = np.zeros(len(sub), int)
for _, _, d in fl_scale:
    _cover += m_of(d)
assert (_cover <= 1).all(), "skale FL (fl_k/fl_mt) nie moga sie nakladac: albo globalna, albo rozlaczne grupy"
for _kind in ("fl_f", "pa_f", "arpa", "arfl", "pa_g", "fl_s", "fl_pg", "pa_med", "fl_med", "pa_pg", "pa_or", "mtv", "mt_f", "fl_dev", "pa_dev", "fl_ens", "fl_enq", "pa_ens", "pa_mix", "pa_fx", "fl_fx", "mt_fx"):
    _cover = np.zeros(len(sub), int)
    for k, _, d in ops:
        if k == _kind:
            _cover += m_of(d)
    assert (_cover <= 1).all(), f"{_kind}: grupy nie moga sie nakladac"
for k, val, d in ops:
    if k in ("arpa", "arfl"):
        assert d in ("sie", "telr", "teln", "gry", "tel"), f"{k}: podaj grupe TIFF (@sie/@telr/...), PNG ma png_{k}"
    if k == "mtvo":
        assert d is None and 0 <= val <= 1, "mtvo: waga w [0, 1], bez grupy"
        assert any(k2 == "mtv" for k2, _, _ in ops), "mtvo dziala tylko razem z mtv"
    if k == "mt_pair":
        assert d is None and val in (0.0, 1.0), "mt_pair: 0 albo 1, bez grupy"
    if k == "fl_gx":
        assert d is None and 0.8 <= val <= 3.0, "fl_gx: prog w [0.8, 3], bez grupy"
        assert any(k2 == "fl_guard" for k2, _, _ in ops), "fl_gx dziala tylko razem z fl_guard"
    if k == "pa_gx":
        assert d is None and 0 <= val <= 1, "pa_gx: waga w [0, 1], bez grupy"
        assert any(k2 == "fl_guard" for k2, _, _ in ops) and any(k2 == "pa_pg" for k2, _, _ in ops), "pa_gx dziala tylko z fl_guard i pa_pg"
    if k == "mtva":
        assert d is None and val in (0.0, 1.0), "mtva: 0 albo 1, bez grupy"
        assert any(k2 == "mtv" for k2, _, _ in ops), "mtva dziala tylko razem z mtv"
    if k == "pa_dev":
        assert d == "png", "pa_dev tylko @png (PA PNG pochodzi z R21, czyli z modelu R17)"
    if k == "pa_ens":
        assert d == "png", "pa_ens tylko @png (PA PNG pochodzi z R21, czyli z modelu R17)"
    if k == "pa_mix":
        assert d in ("tif", "sie", "telr", "teln", "gry", "tel", "grya", "gryb"), "pa_mix tylko dla grup TIFF (PNG ma pa_ens)"
_kinds = {k for k, _, _ in ops}
# R53: zespoly R17 moga dzialac w rozlacznych grupach (np. fl_dev0.5@tif + fl_ens1@png), ale nie na tych samych obrazach
for _fam, _txt in (({"fl_dev", "fl_ens", "fl_enq"}, "fl_dev, fl_ens i fl_enq"), ({"pa_dev", "pa_ens"}, "pa_dev i pa_ens")):
    _cover = np.zeros(len(sub), int)
    for k, _, d in ops:
        if k in _fam:
            _cover += m_of(d)
    assert (_cover <= 1).all(), f"{_txt} wykluczaja sie na tych samych obrazach (ten sam zespol) - podaj rozlaczne grupy"
MTVO = [v for k, v, _ in ops if k == "mtvo"]
MTVO = MTVO[0] if MTVO else None
MTVA_ON = any(k == "mtva" and v == 1.0 for k, v, _ in ops)
OV_MIN = 0.2    # mtvo: v2 widzi obie krawedzie na co najmniej 20 pp szerokosci obrazu wiecej niz R12B
PAIR_Q = 1.5    # mt_pair: para R12B niemozliwa geometrycznie (q > 1.5), para v2 mozliwa (1/1.5 <= q <= 1.5)
PAIR = np.zeros(len(sub), bool)   # obrazy, w ktorych mt_pair wybral pare v2 (mt_band i mtv ich nie ruszaja)
GX_T = 2.0      # R54: q = FL sin PA / MT > 2 przy guardzie -> FL niemozliwe geometrycznie (eksperci OSF: q <= 1.8)
if GXT_IN is not None:   # R83: gxt<prog> - inny prog q dla fl_gx
    GX_T = GXT_IN
    print('gxt: prog q dla fl_gx =', GX_T)
EXT = np.zeros(len(sub), bool)    # obrazy z q > GX_T (wyznaczane w kroku fl_guard)
GX_CAP = next((v for k, v, _ in ops if k == "fl_gx"), None)
PA_GX = next((v for k, v, _ in ops if k == "pa_gx"), None)
FXV = np.array([[np.nan if v is None else v for v in FX.get(i, [None] * len(FX_COLS))] for i in sub.image_id], float)
def _fx_sel(sel, user):
    v = [x for k, x, d in ops if k == sel]
    if not any(k == user for k, _, _ in ops):
        assert not v, f"{sel} bez {user}"
        return None
    assert len(v) == 1, f"{user} wymaga dokladnie jednego {sel}<indeks> (indeks w FX_COLS)"
    assert all(d is None for k, _, d in ops if k == sel), f"{sel}: bez grupy"
    k_ = int(v[0]); assert v[0] == k_ and 0 <= k_ < len(FX_COLS), f"{sel}: indeks 0..{len(FX_COLS) - 1}"
    print(f"{user}: cecha {FX_COLS[k_]} (dostepna dla {int(np.isfinite(FXV[:, k_]).sum())} obrazow)")
    return k_
FXP, FXF, FXM = _fx_sel("fxp", "pa_fx"), _fx_sel("fxf", "fl_fx"), _fx_sel("fxm", "mt_fx")
ORDER = ["mt_d", "pa_med", "pa_fx", "pa_or", "pa_dev", "pa_ens", "pa_mix", "png_arpa", "pa_d", "pa_s", "pa_seq", "arpa", "pa_f",
         "fl_k", "fl_mt", "fl_dev", "fl_ens", "fl_enq", "fl_fx", "fl_seq", "fl_phys", "mt_pair", "fl_guard", "png_arfl", "arfl", "fl_f", "mt_band", "mtv", "mt_fx", "mt_f", "pa_g", "pa_pg", "fl_pg", "fl_med", "fl_s"]
log = []
for kind in ORDER:
    for k, val, d in ops:
        if k != kind:
            continue
        mask = m_of(d)
        if not mask.any() or (k.startswith("png_ar") and not IS_PNG.any()):
            continue
        if k == "mt_d":
            assert -3 <= val <= 3, "przesuniecie MT poza zakresem"
            sub.loc[mask, "mt_mm"] = base.loc[mask, "mt_mm"] + val
        elif k == "mt_f":
            # R59: koncowe przesuniecie MT grupy (po mt_band i mtv, przed geometria pa_pg / fl_pg). mt_d dziala na poczatku,
            # wiec przy mtva1 (MT w 70% z poziomu v2) jego zmiana w grupie zostaje w wyniku tylko w ok. 30%.
            assert -1 <= val <= 1, "mt_f: przesuniecie MT w [-1, 1] mm"
            sub.loc[mask, "mt_mm"] = sub.loc[mask, "mt_mm"].values + val
            print(f"mt_f{val:+}{'@' + d if d else ''}: {int(mask.sum())} obrazow")
        elif k == "pa_d":
            assert -10 <= val <= 10, "przesuniecie PA poza zakresem"
            sub.loc[mask, "pa_deg"] = sub.loc[mask, "pa_deg"] + val
        elif k == "pa_s":
            # shrinkage PA wokol sredniej grupy (po przesunieciach): PA <- m + s * (PA - m)
            assert 0.5 <= val <= 1.5, "s dla pa_s w [0.5, 1.5]"
            if d == "png" and MUS_ON:
                for gg in np.unique(GROUP2[mask]):
                    mg = mask & (GROUP2 == gg); m_ = float(sub.loc[mg, "pa_deg"].mean())
                    sub.loc[mg, "pa_deg"] = m_ + val * (sub.loc[mg, "pa_deg"].values - m_)
                print(f"pa_s{val}@png: osobno w podgrupach miesni")
            else:
                m_ = float(sub.loc[mask, "pa_deg"].mean())
                sub.loc[mask, "pa_deg"] = m_ + val * (sub.loc[mask, "pa_deg"].values - m_)
                print(f"pa_s{val}{'@' + d if d else ''}: srednia {m_:.3f} st, {int(mask.sum())} obrazow")
        elif k == "pa_seq":
            assert 0 <= val <= 1, "w dla pa_seq w [0, 1]"
            log.append(("pa_seq", seq_shrink("pa_deg", val, mask)))
        elif k == "arpa":
            # korekta proporcji na koncu toru PA (po przesunieciach i shrinkage, ktore dzialaja w pikselach):
            # obraz z pozioma skala r * (pionowa): tan(PA) = r * tan(PA_px)
            assert 0.5 <= val <= 1.5, "arpa: r w [0.5, 1.5]"
            before = sub.loc[mask, "pa_deg"].values.copy()
            sub.loc[mask, "pa_deg"] = np.degrees(np.arctan(val * np.tan(np.radians(before))))
            print(f"arpa{val}@{d}: {int(mask.sum())} obrazow, PA srednio {float((sub.loc[mask, 'pa_deg'].values - before).mean()):+.2f} st")
        elif k == "mt_band":
            # MT <- MT + w * (srednie pasmo - pasmo obrazu): grubsze rozciegna -> wieksza roznica srodek-krawedz.
            # Srednia MT po obrazach z pasmem zostaje bez zmian. Na OSF-35: MAE MT 0,49 -> 0,38 mm przy w = 1.
            assert 0 <= val <= 2, "w dla mt_band w [0, 2]"
            bnd = np.array([np.nan if BAND.get(i) is None else BAND[i] for i in sub.image_id], float)
            ok_ = mask & np.isfinite(bnd) & ~PAIR
            bm = float(np.nanmean(bnd[np.isfinite(bnd)]))
            adj = np.where(ok_, val * (bm - np.nan_to_num(bnd, nan=bm)), 0.0)
            sub["mt_mm"] = sub.mt_mm.values + adj
            print(f"mt_band{val}{'@' + d if d else ''}: {int(ok_.sum())} obrazow, srednie pasmo {bm:.4f} mm, "
                  f"srednia |korekta| {np.abs(adj[ok_]).mean():.3f} mm, zakres {adj.min():+.2f}..{adj.max():+.2f}")
        elif k == "pa_med":
            # PA startowe = mediana trzech niezaleznych rodzin dekoderow (patrz nizej), kazdy dekoder przesuniety
            # o swoja mediane roznicy do R23A w grupie (srednia grupy zostaje). Chroni przed grubymi bledami jednego dekodera.
            # val = waga: PA_SRC <- (1 - val) * R23A + val * mediana
            assert 0 <= val <= 1 and d is not None, "pa_med: waga w [0, 1], podaj grupe (np. @png)"
            dec = np.array([[np.nan if PADEC.get(i) is None or PADEC[i][j] is None else PADEC[i][j] for j in (0, 1, 2)] for i in sub.image_id], float)
            new = PA_SRC.copy()
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg)
                cols = [PA_RAW[mg]]
                # PNG: R23A (= R21) + r17_median + local_support; TIFF: R23A (= local_support) + r17_median + local_median
                for j in ((0, 1) if gg.startswith("png") else (0, 2)):
                    dj = dec[mg, j]; bias = np.nanmedian(dj - PA_RAW[mg])
                    cols.append(dj - bias)
                V = np.vstack(cols)
                med = np.where(np.isfinite(V).sum(0) == 3, np.nanmedian(V, axis=0), PA_RAW[mg])   # bez trzech glosow: R23A
                new[mg] = (1 - val) * PA_RAW[mg] + val * med
            ch = mask & (np.abs(new - PA_SRC) > 1e-9)
            PA_SRC[mask] = new[mask]
            sub.loc[mask, "pa_deg"] = PA_SRC[mask]
            print(f"pa_med{val}@{d}: zmienione {int(ch.sum())} obrazow, srednio |{np.abs(new - PA_RAW)[mask].mean():.2f}| st, "
                  f"max {np.abs(new - PA_RAW)[mask].max():.1f} st")
        elif k == "pa_fx":
            # R64: PA startowe ku cesze R63T (FX_COLS[FXP]) przesunietej w (pod)grupie o mediane roznicy do biezacego PA
            # (poziom grupy zostaje), roznica obcieta do +-8 st. Piksele, przed png_arpa / arpa. Brak cechy -> bez zmian.
            assert 0 <= val <= 1, "pa_fx: waga w [0, 1]"
            fv = FXV[:, FXP]; new = PA_SRC.copy()
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); ok_ = mg & np.isfinite(fv)
                if ok_.sum() < 3:
                    continue
                sh = fv - float(np.median(fv[ok_] - PA_SRC[ok_]))
                new[ok_] = PA_SRC[ok_] + val * np.clip(sh[ok_] - PA_SRC[ok_], -8, 8)
                new[mg] += float(np.median(PA_SRC[mg]) - np.median(new[mg]))   # mediana (pod)grupy dokladnie bez zmian
            print(f"pa_fx{val}{'@' + d if d else ''} ({FX_COLS[FXP]}): zmienione {int((mask & (np.abs(new - PA_SRC) > 1e-9)).sum())} obrazow, "
                  f"srednio |dPA| {np.abs(new - PA_SRC)[mask].mean():.3f} st, max {np.abs(new - PA_SRC)[mask].max():.2f} st")
            PA_SRC[mask] = new[mask]
            sub.loc[mask, "pa_deg"] = PA_SRC[mask]
        elif k == "fl_fx":
            # R64: FL ku cesze R63T (FX_COLS[FXF]) w skali log; mediana log-roznicy w (pod)grupie odjeta (poziom FL zostaje),
            # roznica obcieta do +-0,4. Po zespolach R17, przed fl_seq i guardem.
            assert 0 <= val <= 1, "fl_fx: waga w [0, 1]"
            fv = FXV[:, FXF]; cur = sub.fl_mm.values.copy(); fac = np.ones(len(sub))
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); ok_ = mg & np.isfinite(fv) & (fv > 0)
                if ok_.sum() < 3:
                    continue
                r = np.log(np.where(ok_, fv, 1.0) / cur)
                fac[ok_] = np.exp(val * np.clip(r[ok_] - float(np.median(r[ok_])), -0.4, 0.4))
                fac[mg] *= float(np.median(cur[mg]) / np.median(cur[mg] * fac[mg]))   # mediana FL (pod)grupy bez zmian
            sub["fl_mm"] = cur * fac
            print(f"fl_fx{val}{'@' + d if d else ''} ({FX_COLS[FXF]}): {int((mask & (fac != 1)).sum())} obrazow, "
                  f"srednio |dFL| {np.abs(sub.fl_mm.values - cur)[mask].mean():.2f} mm, max {np.abs(sub.fl_mm.values - cur).max():.1f} mm")
        elif k == "mt_fx":
            # R64: MT ku cesze R63T (FX_COLS[FXM]) przesunietej o mediane roznicy w (pod)grupie (poziom zostaje); tylko
            # |roznica| <= 2 mm (inaczej to inna para rozciegien). Po mtv, przed mt_f.
            assert 0 <= val <= 1, "mt_fx: waga w [0, 1]"
            fv = FXV[:, FXM]; cur = sub.mt_mm.values.copy(); new = cur.copy()
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); ok_ = mg & np.isfinite(fv) & ~PAIR
                if ok_.sum() < 3:
                    continue
                sh = fv - float(np.median(fv[ok_] - cur[ok_]))
                same = ok_ & (np.abs(sh - cur) <= 2.0)
                new[same] = cur[same] + val * (sh[same] - cur[same])
                new[mg] += float(np.median(cur[mg]) - np.median(new[mg]))   # mediana MT (pod)grupy bez zmian
            sub["mt_mm"] = new
            print(f"mt_fx{val}{'@' + d if d else ''} ({FX_COLS[FXM]}): {int((mask & (new != cur)).sum())} obrazow, "
                  f"srednio |dMT| {np.abs(new - cur)[mask].mean():.3f} mm, max {np.abs(new - cur).max():.2f} mm")
        elif k == "pa_or":
            # PA startowe ku dekoderowi z pola kierunkow R22 (glebokie 40% miesnia), przesunietemu w kazdej (pod)grupie
            # o mediane roznicy do biezacego PA (srednia grupy zostaje): PA_SRC <- PA_SRC + w * (OR - PA_SRC).
            # Dziala w pikselach, jak pa_med (przed png_arpa/arpa). Brak dekodera -> obraz bez zmian.
            assert 0 <= val <= 1, "pa_or: waga w [0, 1]"
            orv = np.array([np.nan if OR_INS.get(i) is None else OR_INS[i] for i in sub.image_id], float)
            new = PA_SRC.copy()
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); ok_ = mg & np.isfinite(orv)
                if ok_.sum() < 3:
                    continue
                sh = orv - float(np.median(orv[ok_] - PA_SRC[ok_]))
                new[ok_] = PA_SRC[ok_] + val * (sh[ok_] - PA_SRC[ok_])
            ch = mask & (np.abs(new - PA_SRC) > 1e-9)
            print(f"pa_or{val}{'@' + d if d else ''}: zmienione {int(ch.sum())} obrazow, srednio |dPA| {np.abs(new - PA_SRC)[mask].mean():.3f} st, "
                  f"max {np.abs(new - PA_SRC)[mask].max():.2f} st")
            PA_SRC[mask] = new[mask]
            sub.loc[mask, "pa_deg"] = PA_SRC[mask]
        elif k == "mtv":
            # MT ku MT z rozciegien v2 (R45T), tylko gdy obie metody widza te sama pare rozciegien:
            # V2 przesuniete o mediane roznicy w grupie (mediana grupy zostaje), |V2 - MT| <= 2 mm; inaczej bez zmian
            # (wieksze roznice to inna para rozciegien, np. glebsze rozciegno w blokach 71-75 i 116-120).
            # mtvo<w>: inna waga (zwykle 1) dla obrazow, na ktorych v2 widzi rozciegna na >= 20 pp szerokosci wiecej niz R12B
            # (R12B wykryl rozciegno tylko czesciowo, wiec jego MT jest mniej pewne); przesuniecie grupy liczone jak wyzej.
            assert 0 <= val <= 1, "mtv: waga w [0, 1]"
            v2 = np.array([np.nan if V2MT.get(i) is None else V2MT[i] for i in sub.image_id], float)
            dov = np.array([np.nan if V2OV.get(i) is None else V2OV[i] for i in sub.image_id], float)
            wv = np.full(len(sub), float(val))
            gate = np.isfinite(dov) & (dov >= OV_MIN)
            if MTVO is not None:
                wv[gate] = MTVO
            cur = sub.mt_mm.values.copy(); new = cur.copy(); used = np.zeros(len(sub), bool)
            # mtva1: jedno przesuniecie dla wszystkich grup (mediana po calej masce) - roznice poziomu MT miedzy grupami
            # bierze wtedy z v2 (pomiar bezwzgledny krawedzi), a nie z dotychczasowego MT
            okall = mask & np.isfinite(v2) & ~PAIR
            sh_all = float(np.median(v2[okall] - cur[okall]))
            for gg in np.unique(GROUP[mask]):
                mg = mask & (GROUP == gg); ok_ = mg & np.isfinite(v2) & ~PAIR
                if ok_.sum() < 3:
                    continue
                sh = v2 - (sh_all if MTVA_ON else float(np.median(v2[ok_] - cur[ok_])))
                same = ok_ & (np.abs(sh - cur) <= 2.0)
                new[same] = cur[same] + wv[same] * (sh[same] - cur[same]); used |= same
            sub["mt_mm"] = new
            print(f"mtv{val}{'@' + d if d else ''}: zmienione {int(used.sum())} obrazow (ta sama para rozciegien), "
                  f"srednio |dMT| {(np.abs(new - cur)[used].mean() if used.any() else 0.):.3f} mm, max {np.abs(new - cur).max():.2f} mm")
            if MTVO is not None:
                go = used & gate
                print(f"mtvo{MTVO}: waga {MTVO} zamiast {val} dla {int(go.sum())} obrazow (v2 widzi rozciegna na >= {OV_MIN:.0%} szerokosci wiecej): "
                      + ", ".join(f"{i}: {c:.2f} -> {n:.2f}" for i, c, n in zip(sub.image_id[go], cur[go], new[go])))
        elif k == "fl_dev":
            # FL ku sredniej geometrycznej dwoch modeli R17: FL <- FL * exp(w * (d - med_g d)), d = log(FL_dev / FL_refit).
            # Tylko obrazy, w ktorych FL R23A pochodzi z modelu refit (303/309). Mediana d w grupie odjeta, wiec poziom FL
            # w grupie (skalibrowany na LB) zostaje - zmieniaja sie tylko roznice miedzy obrazami (redukcja szumu modelu).
            assert 0 <= val <= 1, "fl_dev: waga w [0, 1]"
            arr = np.array([[np.nan if x is None else x for x in R17DEV.get(i, [None] * 4)] for i in sub.image_id], float)
            fdv, frf = arr[:, 0], arr[:, 1]
            okd = mask & np.isfinite(fdv) & np.isfinite(frf) & (frf > 0) & (np.abs(base.fl_mm.values / np.where(frf > 0, frf, 1) - 1) < 1e-3)
            dd = np.log(np.where(okd, fdv / np.where(okd, frf, 1), 1.0))
            fac = np.ones(len(sub))
            for gg in np.unique(GROUP[okd]):
                mg = okd & (GROUP == gg)
                fac[mg] = np.exp(val * (dd[mg] - float(np.median(dd[mg]))))
            before = sub.fl_mm.values.copy()
            sub["fl_mm"] = before * fac
            print(f"fl_dev{val}{'@' + d if d else ''}: {int(okd.sum())} obrazow, srednio |dFL| {np.abs(sub.fl_mm.values - before)[okd].mean():.2f} mm, "
                  f"max {np.abs(sub.fl_mm.values - before).max():.1f} mm")
        elif k in ("fl_ens", "fl_enq"):
            # FL ku zespolowi 4 modeli R17 (R50T): FL <- FL * exp(w * (d - med_g d)), d = log(FL_zespol / FL_refit), oba z tego
            # samego przebiegu R50T. fl_enq: zespol z widokiem o kwadratowych pikselach (8 pomiarow). Tylko obrazy, w ktorych FL R23A
            # pochodzi z modelu refit (R17C, 303/309). Mediana d w grupie odjeta - poziom FL grupy (skalibrowany na LB) zostaje.
            assert 0 <= val <= 1, f"{k}: waga w [0, 1]"
            col = 1 if k == "fl_ens" else 2
            arr = np.array([[np.nan if x is None else x for x in R17ENS.get(i, [None] * 5)] for i in sub.image_id], float)
            r17c = np.array([[np.nan if x is None else x for x in R17DEV.get(i, [None] * 4)] for i in sub.image_id], float)[:, 1]
            fre, fen = arr[:, 0], arr[:, col]
            okd = (mask & np.isfinite(fre) & np.isfinite(fen) & (fre > 0) & np.isfinite(r17c) & (r17c > 0)
                   & (np.abs(base.fl_mm.values / np.where(r17c > 0, r17c, 1) - 1) < 1e-3))
            dd = np.log(np.where(okd, fen / np.where(okd, fre, 1), 1.0))
            fac = np.ones(len(sub))
            for gg in np.unique(GROUP[okd]):
                mg = okd & (GROUP == gg)
                fac[mg] = np.exp(val * (dd[mg] - float(np.median(dd[mg]))))
            before = sub.fl_mm.values.copy()
            sub["fl_mm"] = before * fac
            print(f"{k}{val}{'@' + d if d else ''}: {int(okd.sum())} obrazow, srednio |dFL| {np.abs(sub.fl_mm.values - before)[okd].mean():.2f} mm, "
                  f"max {np.abs(sub.fl_mm.values - before).max():.1f} mm")
        elif k == "pa_ens":
            # PA PNG (R21 = fragmenty modelu R17 refit) ku zespolowi 4 modeli R17: PA_SRC <- PA_SRC + w * (dp - med dp),
            # dp = PA_zespol - PA_refit (oba z R50T, piksele). Mediana w podgrupie odjeta.
            assert 0 <= val <= 1, "pa_ens: waga w [0, 1]"
            arr = np.array([[np.nan if x is None else x for x in R17ENS.get(i, [None] * 5)] for i in sub.image_id], float)
            dp = arr[:, 4] - arr[:, 3]
            okp = mask & np.isfinite(dp)
            new = PA_SRC.copy()
            for gg in np.unique(GROUP2[okp]):
                mg = okp & (GROUP2 == gg)
                new[mg] = PA_SRC[mg] + val * (dp[mg] - float(np.median(dp[mg])))
            print(f"pa_ens{val}@{d}: {int(okp.sum())} obrazow, srednio |dPA| {np.abs(new - PA_SRC)[okp].mean():.3f} st, max {np.abs(new - PA_SRC).max():.2f}")
            PA_SRC[mask] = new[mask]
            sub.loc[mask, "pa_deg"] = PA_SRC[mask]
        elif k == "pa_mix":
            # PA TIFF (R23A = local_support, po pa_med mediana trzech dekoderow R22) usrednione z PA zespolu 4 modeli R17
            # (z odbiciem, piksele - jak PA_SRC), przesunietym w kazdej (pod)grupie o mediane roznicy:
            # B = PA_SRC + w * (E - PA_SRC). Potem odchylenia od sredniej grupy przeskalowane tak, zeby srednia i SD grupy
            # zostaly jak przed usrednieniem: zmienia sie tylko to, ktore obrazy maja PA wyzsze, a ktore nizsze.
            # Usrednienie dwoch rodzin modeli zmniejsza szum; sciaganie do sredniej zostaje w pa_s / pa_g.
            # OSF-35 (ten sam rozrzut, sciaganie 0.85-0.9): w = 0.4 -> MAE PA 1.07-1.09 -> 1.03-1.06 st.
            assert 0 <= val <= 1, "pa_mix: waga w [0, 1]"
            arr = np.array([[np.nan if x is None else x for x in R17ENS.get(i, [None] * 5)] for i in sub.image_id], float)
            e = arr[:, 4]
            new = PA_SRC.copy(); n_mix = 0
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); ok_ = mg & np.isfinite(e)
                if ok_.sum() < 5:
                    continue
                x = PA_SRC[ok_]
                b = x + val * ((e[ok_] - float(np.median(e[ok_] - x))) - x)
                sd0, sd1 = float(x.std()), float(b.std())
                new[ok_] = x.mean() + (sd0 / sd1 if sd1 > 1e-9 else 1.0) * (b - b.mean())
                n_mix += int(ok_.sum())
            print(f"pa_mix{val}@{d}: {n_mix} obrazow, srednio |dPA| {np.abs(new - PA_SRC)[mask].mean():.3f} st, max {np.abs(new - PA_SRC).max():.2f}")
            PA_SRC[mask] = new[mask]
            sub.loc[mask, "pa_deg"] = PA_SRC[mask]
        elif k == "pa_dev":
            # PA PNG (R21 = fragmenty modelu R17 refit) ku sredniej dwoch modeli R17: PA_SRC <- PA_SRC + w * (dp - med dp),
            # dp = PA_dev - PA_refit (piksele, jak PA_SRC). Mediana w podgrupie odjeta.
            assert 0 <= val <= 1, "pa_dev: waga w [0, 1]"
            arr = np.array([[np.nan if x is None else x for x in R17DEV.get(i, [None] * 4)] for i in sub.image_id], float)
            dp = arr[:, 2] - arr[:, 3]
            okp = mask & np.isfinite(dp)
            new = PA_SRC.copy()
            for gg in np.unique(GROUP2[okp]):
                mg = okp & (GROUP2 == gg)
                new[mg] = PA_SRC[mg] + val * (dp[mg] - float(np.median(dp[mg])))
            print(f"pa_dev{val}@{d}: {int(okp.sum())} obrazow, srednio |dPA| {np.abs(new - PA_SRC)[okp].mean():.3f} st, max {np.abs(new - PA_SRC).max():.2f}")
            PA_SRC[mask] = new[mask]
            sub.loc[mask, "pa_deg"] = PA_SRC[mask]
        elif k == "pa_g":
            # shrinkage PA w kazdej grupie zrodla osobno, wokol sredniej grupy (po wszystkich korektach)
            assert 0.2 <= val <= 1.5, "pa_g w [0.2, 1.5]"   # R49: dolna granica 0.5 -> 0.2 (teln chce mocnego sciagania)
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); m_ = float(sub.loc[mg, "pa_deg"].mean())
                sub.loc[mg, "pa_deg"] = m_ + val * (sub.loc[mg, "pa_deg"].values - m_)
            print(f"pa_g{val}{'@' + d if d else ''}: {int(mask.sum())} obrazow")
        elif k == "fl_s":
            # shrinkage FL w kazdej grupie osobno, wokol mediany grupy (po wszystkich korektach FL)
            assert 0.5 <= val <= 1.5, "fl_s w [0.5, 1.5]"
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); m_ = float(np.median(sub.loc[mg, "fl_mm"].values))
                sub.loc[mg, "fl_mm"] = m_ + val * (sub.loc[mg, "fl_mm"].values - m_)
            print(f"fl_s{val}{'@' + d if d else ''}: {int(mask.sum())} obrazow")
        elif k == "fl_med":
            # FL = mediana trzech oszacowan w grupie: nasze FL, FL z dekodera R22 (przeskalowane do naszej mediany w grupie)
            # i geometria Q_g * MT / sin PA (Q_g = mediana q w grupie). FL <- (1 - w) * FL + w * mediana.
            assert 0 <= val <= 1, "fl_med w [0, 1]"
            fl22 = np.array([np.nan if FL22.get(i) is None else FL22[i] for i in sub.image_id], float)
            cur = sub.fl_mm.values.copy()
            geo = sub.mt_mm.values / np.sin(np.radians(sub.pa_deg.values))
            new = cur.copy()
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); ok_ = mg & np.isfinite(fl22)
                if ok_.sum() < 3:
                    continue
                k22 = float(np.median(cur[ok_] / fl22[ok_])); Qg = float(np.median(cur[mg] / geo[mg]))
                med = np.median(np.vstack([cur[ok_], fl22[ok_] * k22, geo[ok_] * Qg]), axis=0)
                new[ok_] = (1 - val) * cur[ok_] + val * med
            sub["fl_mm"] = new
            print(f"fl_med{val}{'@' + d if d else ''}: zmienione {int((np.abs(new - cur) > 1e-9).sum())} obrazow, "
                  f"srednio |{np.abs(new - cur)[mask].mean():.2f}| mm, max {np.abs(new - cur).max():.1f} mm")
        elif k == "pa_pg":
            # PA ku geometrii wlasnej grupy z FL i MT: PA_geo = asin(Q_g * MT / FL), Q_g = mediana FL*sin(PA)/MT w grupie;
            # PA <- PA + w * (PA_geo - PA) (roznica obcieta do +-8 st). fl_pg liczy potem geometrie z PA sprzed tej zmiany.
            # Na OSF-35: w = 0.2 -> MAE PA 1.174 -> 1.129 st.
            assert 0 <= val <= 1, "pa_pg w [0, 1]"
            cur = sub.pa_deg.values.copy(); fl = sub.fl_mm.values; mt = sub.mt_mm.values
            new = cur.copy()
            wv = np.full(len(sub), float(val))
            if PA_GX is not None:
                wv[EXT] = PA_GX                      # pa_gx: inna waga dla obrazow z q > 2 (z kroku fl_guard)
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg)
                Qg = float(np.median(fl[mg] * np.sin(np.radians(cur[mg])) / mt[mg]))
                pg = np.degrees(np.arcsin(np.clip(Qg * mt[mg] / fl[mg], 0.05, 0.9)))
                new[mg] = cur[mg] + wv[mg] * np.clip(pg - cur[mg], -8, 8)
            PA_FOR_FL = cur
            sub["pa_deg"] = new
            print(f"pa_pg{val}{'@' + d if d else ''}: srednio |dPA| {np.abs(new - cur)[mask].mean():.3f} st, max {np.abs(new - cur).max():.2f}")
        elif k == "fl_pg":
            # FL ku geometrii wlasnej grupy: q = FL / (MT / sin PA) z koncowych PA i MT, Q_g = mediana q w grupie,
            # FL <- FL * (Q_g / q)^w. Na OSF-35 (jedno zrodlo) w = 0.25 obnizylo MAE FL o 0.54 mm.
            assert 0 <= val <= 1, "fl_pg w [0, 1]"
            pa_geo_src = sub.pa_deg.values if PA_FOR_FL is None else PA_FOR_FL
            geo = sub.mt_mm.values / np.sin(np.radians(pa_geo_src))
            q = sub.fl_mm.values / geo
            before = sub.fl_mm.values.copy()
            for gg in np.unique(GROUP2[mask]):
                mg = mask & (GROUP2 == gg); Qg = float(np.median(q[mg]))
                sub.loc[mg, "fl_mm"] = before[mg] * (Qg / q[mg]) ** val
            print(f"fl_pg{val}{'@' + d if d else ''}: {int(mask.sum())} obrazow, srednia |dFL| {np.abs(sub.fl_mm.values - before)[mask].mean():.2f} mm")
        elif k == "pa_f":
            # koncowe przesuniecie PA w stopniach fizycznych (po arpa); nie rusza sredniej innych grup
            assert -5 <= val <= 5, "pa_f w [-5, 5]"
            sub.loc[mask, "pa_deg"] = sub.loc[mask, "pa_deg"] + val
            print(f"pa_f{val:+}{'@' + d if d else ''}: {int(mask.sum())} obrazow")
        elif k == "fl_f":
            # koncowy mnoznik FL (po guard, png_arfl i arfl)
            assert 0.5 <= val <= 1.5, "fl_f w [0.5, 1.5]"
            sub.loc[mask, "fl_mm"] = sub.loc[mask, "fl_mm"].values * val
            print(f"fl_f{val}{'@' + d if d else ''}: {int(mask.sum())} obrazow")
        elif k == "fl_k":
            assert 0.5 <= val <= 1.5, "skala FL poza zakresem"
            sub.loc[mask, "fl_mm"] = base.loc[mask, "fl_mm"] * val
        elif k == "fl_mt":
            # FL srodek-srodek -> krawedz-krawedz, podobienstwo trojkatow; MT_sr = MT_R23A + 1.3629
            assert 0.0 <= val <= 3.0, "grubosc pasma poza zakresem"
            mt_c = base.loc[mask, "mt_mm"] + MT_OFFSET_R12C
            sub.loc[mask, "fl_mm"] = base.loc[mask, "fl_mm"] * (mt_c - val) / mt_c
        elif k == "fl_seq":
            assert 0 <= val <= 1, "w dla fl_seq w [0, 1]"
            log.append(("fl_seq", seq_shrink("fl_mm", val, mask)))
        elif k == "png_arpa":
            # obraz PNG sciesniony poziomo o r: kat w pikselach jest zawyzony, tan(PA) = tan(PA_px) * r
            assert 0.5 <= val <= 1.0 and d is None, "png_arpa: r w [0.5, 1], bez domeny"
            sub.loc[IS_PNG, "pa_deg"] = np.degrees(np.arctan(val * np.tan(np.radians(PA_SRC[IS_PNG]))))
            print(f"png_arpa{val}: PNG PA srednio {float((sub.pa_deg.values - PA_SRC)[IS_PNG].mean()):+.2f} st wzgledem startu PA")
        elif k == "png_arfl":
            # dlugosc odcinka pod katem phi (w pikselach) przy skali pionowej: FL * sqrt(cos^2/r^2 + sin^2)
            assert 0.5 <= val <= 1.0 and d is None, "png_arfl: r w [0.5, 1], bez domeny"
            phi = np.radians(PA_RAW[IS_PNG])
            fac = np.sqrt(np.cos(phi) ** 2 / val ** 2 + np.sin(phi) ** 2)
            sub.loc[IS_PNG, "fl_mm"] = sub.loc[IS_PNG, "fl_mm"].values * fac
            print(f"png_arfl{val}: mnoznik FL dla PNG srednio {fac.mean():.4f} (min {fac.min():.4f}, max {fac.max():.4f})")
        elif k == "arfl":
            # jak png_arfl, dla grupy TIFF; kat w pikselach = PA R23A; po guard (guard liczy w pikselach)
            assert 0.5 <= val <= 1.5, "arfl: r w [0.5, 1.5]"
            phi = np.radians(PA_RAW[mask])
            fac = np.sqrt(np.cos(phi) ** 2 / val ** 2 + np.sin(phi) ** 2)
            before = sub.loc[mask, "fl_mm"].values.copy()
            sub.loc[mask, "fl_mm"] = before * fac
            print(f"arfl{val}@{d}: {int(mask.sum())} obrazow, mnoznik FL srednio {fac.mean():.4f}, "
                  f"srednio {float((sub.loc[mask, 'fl_mm'].values - before).mean()):+.2f} mm")
        elif k == "fl_phys":
            # sciaganie FL do geometrii: q = FL / (MT / sin PA), Q = mediana q (bez etykiet), FL <- FL * (Q / q)^w
            assert 0 <= val <= 1, "w dla fl_phys w [0, 1]"
            phys = sub.mt_mm.values / np.sin(np.radians(PA_G))
            q = sub.fl_mm.values / phys
            Q = float(np.median(q))
            before = sub.fl_mm.values.copy()
            sub.loc[mask, "fl_mm"] = before[mask] * (Q / q[mask]) ** val
            print(f"fl_phys{val}: Q = mediana q = {Q:.4f} | srednia |dFL| {np.abs(sub.fl_mm.values - before).mean():.2f} mm")
        elif k == "mt_pair":
            # Wybor pary rozciegien zgodnej z geometria, przed guard. R12B i v2 czasem widza rozne pary (|dMT| > 2 mm,
            # np. glebsze rozciegno). Gdy FL modelu peczkow (R17) nie miesci sie miedzy rozciegnami R12B
            # (q12 = FL sin PA / MT > 1.5, ten sam test co guard), a miesci sie miedzy rozciegnami v2 (1/1.5 <= q_v2 <= 1.5),
            # wybieramy pare v2: MT <- MT v2 (przesuniete o mediane roznicy w grupie, jak mtv), a guard nie obcina FL,
            # bo sprzecznosc wyjasnia para rozciegien, a nie FL. Reguła z obrazu (trzy modele, bez etykiet).
            if val == 1.0:
                v2 = np.array([np.nan if V2MT.get(i) is None else V2MT[i] for i in sub.image_id], float)
                cur = sub.mt_mm.values.copy()
                v2s = np.full(len(sub), np.nan)
                for gg in np.unique(GROUP):
                    mg = (GROUP == gg) & np.isfinite(v2)
                    if mg.sum() >= 3:
                        v2s[mg] = v2[mg] - float(np.median(v2[mg] - cur[mg]))
                sp = np.sin(np.radians(PA_G)); flc = sub.fl_mm.values
                q12 = flc * sp / cur
                qv2 = flc * sp / np.where(np.isfinite(v2s), v2s, np.nan)
                PAIR[:] = (np.isfinite(v2s) & (np.abs(v2s - cur) > 2.0) & (q12 > PAIR_Q)
                           & (qv2 >= 1.0 / PAIR_Q) & (qv2 <= PAIR_Q))
                sub.loc[PAIR, "mt_mm"] = v2s[PAIR]
                ch = pd.DataFrame({"image_id": sub.image_id[PAIR], "grupa": GROUP[PAIR], "MT_R12B": cur[PAIR].round(2),
                                   "MT_v2": v2s[PAIR].round(2), "q_R12B": q12[PAIR].round(2), "q_v2": qv2[PAIR].round(2)})
                print(f"mt_pair1: para v2 w {int(PAIR.sum())} obrazach")
                print(ch.to_string(index=False))
        elif k == "fl_guard":
            # prosta powiez miedzy rozciegnami: FL * sin(PA) ~ MT -> FL <= hi * MT / sin(PA)
            assert 1.0 <= val <= 3.0, "hi dla fl_guard w [1, 3]"
            phys = sub.mt_mm.values / np.sin(np.radians(PA_G))
            EXT[:] = mask & (sub.fl_mm.values / phys > GX_T)
            cap = val * phys
            if GX_CAP is not None:
                cap[EXT] = GX_CAP * phys[EXT]      # fl_gx: osobny prog dla obrazow z q > 2
            hit = mask & (sub.fl_mm.values > cap)
            before = sub.fl_mm.values.copy()
            sub.loc[hit, "fl_mm"] = cap[hit]
            print(f"obrazy z q > {GX_T} (FL ponad 2x dluzsze niz pozwala geometria): {int(EXT.sum())}"
                  + (f", prog fl_gx{GX_CAP}" if GX_CAP is not None else "") + (f", PA do geometrii z waga pa_gx{PA_GX}" if PA_GX is not None else ""))
            ch = pd.DataFrame({"image_id": sub.image_id[hit], "FL_przed": before[hit].round(1),
                               "FL_po": sub.fl_mm.values[hit].round(1), "q": (before[hit] / phys[hit]).round(2)})
            print(f"fl_guard{val}: zmienione {int(hit.sum())} obrazow, suma zmian {float((before - sub.fl_mm.values).sum()):.0f} mm")
            print(ch.to_string(index=False))
if not any(k in ("mt_d", "mt_band", "mtv", "mtvo", "mtva", "mt_pair", "mt_f", "mt_fx") for k, _, _ in ops):
    assert (sub.mt_mm.values == base.mt_mm.values).all(), "MT musi zostac bez zmian"
assert np.isfinite(sub[["pa_deg", "fl_mm", "mt_mm"]].values).all() and (sub[["pa_deg", "fl_mm", "mt_mm"]] > 0).all().all()
