# Provenance of the per-image tables used by the final submissions F1 and F3

The final notebooks (`final/UMUD_FINAL_F1_KAGGLE.ipynb`, `final/UMUD_FINAL_F3_KAGGLE.ipynb`) combine per-image tables with fixed constants (`VARIANT`).
This file lists, for every table that F1 or F3 actually uses, the notebook and export it comes from and the code that builds it.
Run folders are on Google Drive under `UMUD_Challenge/outputs/<RUN>`; each records the SHA-256 of its checkpoints.

Builder status: ✔ = code in this repository; ✎ = formula given here, builder script to be restored (the original scripts
were lost in a workspace reset on 12–13 Nov 2026; the tables embedded in the final notebook are the original outputs).

| table / column | used by | content | source notebook → export | builder |
|---|---|---|---|---|
| `RBASE['BASEK']` (`rbasek`) | F1, F3 | base PA / FL / MT per image | R93T (`test_detail.csv`, `test_mt.csv`): R20 geometry on R12B aponeuroses (p ≥ 0.5) + R17-refit fascicles (side 768, p ≥ 0.65) → PA (local support), FL; MT = R12B rule (two largest components, mean vertical inner distance, −1.3629 mm OSF-35 median offset, clip 10–50 mm). PA of 238 TIFF from the late R22 decoder (R45T). | ✎ priority PA: R22 late (TIFF) → R20 support (standard pair) → R20 support on the MT pair → calibrated fallback (per group, best decoder minus its median offset); FL: R17 refit → R20 FL on the MT pair → calibrated `v2_L_fl` (or `Q_g·MT/sin PA` where better); MT: R12B rule → v2 inner − 0.123 mm. Rules decide only 14 images (00019, 00151–00155, 00170, 00256, 00261, 00271, 00272, 00275, 00303, 00307); all other values are the original pipeline outputs. |
| `CTAB['BAND']` | F1, F3 (`mt_band`) | aponeurosis band correction of the R22 decoder with the R12B pair | R71T stage D5 decoder export | ✔ `src/umud_final/tables.py` (`band`) |
| `CTAB['PADEC']` | F1, F3 (`pa_med`) | 3 PA decoders of the R22 maps with the R12B pair | R71T stage D5 decoder export | ✔ `tables.py` (`padec`) |
| `CTAB['V2MT']`, `CTAB['V2OV']` | F1, F3 (`mtv`, `mtva`) | MT of the v2 aponeurosis pair; how much wider v2 sees both aponeuroses than R12B | R71T stage D5 decoder export (v2 = R45T aponeurosis model) | ✔ `tables.py` (`v2mt`, `v2ov`) |
| `FX[85]` (`CTAB['FX85']`) | F1, F3 (`fxm85`, `mt_fx`) | MT of the A76 aponeurosis maps guarded to the R12B pair | R76T export | ✔ `tables.py` (`a76_mt`) |
| `FX[89]` (`CTAB['FX89']`) | F1, F3 (Telemed part of the FL feature) | lex5 FL: geometric mean of R22 (R71T stage S) and F71 (R71T stage E, folds E4) between `a76_v2mix` edges (R80T); missing when the R12B baseline pair is missing | R71T + R80T exports | ✔ `tables.lex5_fl`, `tables.cmbd` |
| `FX[111]` | via 110/130/132 | lex5 FL of F86 fragments, same edges | R86T export | ✎ `tables.lex5_fl(F86 fragments)` |
| `FX[110]` | via 130 | `(FX89² · FX111)^(1/3)` (R22 · F71 · F86) | — | ✎ (agrees with the embedded column to 0.009 mm) |
| `FX[130]` | F1 (`fxf130`, `fl_fx0.85`) | `FX110` outside Telemed screens, `FX89` on Telemed | — | ✎ |
| `FX[132]` | F3 (`fxf132`, `fl_fx0.85`) | `(c89pf² · FX111)^(1/3)` outside Telemed, `FX89` on Telemed; `c89pf` = geometric mean over R22 folds and over F71 folds (E4) of lex5 from each fold decoded separately | R96T (`test_frags_r22k*`, `test_frags_f71k*`) | ✎ (`tables.lex5_fl` per fold, then geometric means) |
| `FX[72]` | F3 (`fxp72`, `pa_fx1.0` @sie/png/gry) | PA from 6 decoder votes (3 decoders × R22, F71) | R71T stage D5 (`test_decoders.csv`) | ✎ median of the 6 votes |
| `R17DEV`, `R17ENS` | F1, F3 (`fl_dev`, `pa_ens@png`) | FL / PA of the R17 models (develop, refit; ensemble of 4) | R17C predictions, R50T export | ✔ `tables.py` (`r17dev`, `r17ens`) |
| groups, `gry_b`, PNG muscle label, scale | F1, F3 (`img`) | read from the images at run time | — | ✔ `src/embedded/scale.py`, `src/embedded/png_muscle.py` (templates `TPL`: 6×24×64, SHA-256 `834aeb668427…`, cut once from six Lumify screenshots) |
| video blocks | F1, F3 (`fl_seq`, `pa_seq`) | consecutive file numbers + agreement of base measurements | — | ✔ code in the final notebook |
| constants | F1, F3 | ~60 `VARIANT` tokens, set by hand | — | ✔ `final/VARIANTS.json` |

Kept in the notebook but **not used** by F1 / F3: `PAYLOAD` (legacy base, replaced by `RBASE` under `rbasek`), `CBASE`,
`CBASE2`, `EBLAY`, `EBASE1`, `EBASE2`, `ANCLEX`, `ANCGEO`, `PAV2`, `COV`, `FL22`, `OR_INS`, all other `FX` columns; the frozen
`GROUP` / `PNG_MUSCLE` / `REG_PX_PER_CM` maps only for printed comparisons (and the registry scale as fallback for an
unreadable ruler).

## Checkpoints (SHA-256)

| model | file (Drive) | SHA-256 |
|---|---|---|
| R12B aponeuroses | `UMUD_R12B/best_dinov2_frozen.pt` | `21ec34d0f5dd7664ecab0ca608f976fdf9d2a161e2af2873c21d513cd5ae19ef` |
| R17 refit | `UMUD_Challenge/outputs/R17_v1_0_0/refit/best.pt` | `6957c5228c1cf5c2067096d5f0214f6c2a7f54b0aeca03b8f664051022859d31` |
| R22 (5 folds) | `UMUD_Challenge/outputs/R22_v1_0_0/fold_k/final.pt` | listed in `R22_v1_0_0/TRAINING_COMPLETE.json` |
| F71 (5 folds; E4 = 0, 2, 3, 4) | `UMUD_Challenge/outputs/R71T_v1_0_0/T/fold_k/final.pt` | listed in each `DONE.json` |
| v2, A76, F86, R17 ensemble | R45T / R76T / R86T / R50T run folders | listed in each run's `DONE.json` |

Every inference notebook checks these sums before loading a checkpoint.
