# UMUD Challenge — muscle architecture (PA, FL, MT) from B-mode ultrasound

Solution code for the Kaggle competition *UMUD Challenge: Muscle Architecture in Ultrasound Data*: pennation angle (PA, °),
fascicle length (FL, mm) and muscle thickness (MT, mm) for every test image.

Every number in the submission is computed from the images by code and trained networks. No hand labels or hand
measurements of test images. No leaderboard access at run time. The combination constants were set by hand, which the host
allows (thread 739660: hand-chosen weights and constants are permitted; automatic weighting by leaderboard feedback is not).

## Final submissions

| name | public LB | what it is |
|---|---:|---|
| **F1** | 0.20497 | reproducible baseline: token pipeline B12 + model tables from code (`ctab`) + base measurements without legacy values (`rbasek`) + device group, PNG muscle label and physical scale read from the images (`img`) |
| **F3** | 0.20479 | F1 + FL feature with the folds of the fascicle models R22 / F71 decoded separately (late fusion, FX column 132) and PA from 6 decoder votes (FX column 72), both outside Telemed screens; group levels held at F1 by fixed constants |

Final notebooks (English, one per submission, `VARIANT` preset in cell 1): `final/UMUD_FINAL_F1_KAGGLE.ipynb` and
`final/UMUD_FINAL_F3_KAGGLE.ipynb`. They run the same engine as the notebook that produced the competition submissions
(`final/original/UMUD_R96_KAGGLE.ipynb`, SHA-256 `8776f8cc…`, Polish messages, `VARIANT` pasted by hand). Only comments and
printed messages differ: the parsed code with all string constants masked is identical, and the CSVs are byte-identical
(table below). Exact `VARIANT` strings and notebook checksums: `final/VARIANTS.json`.

Repository: <REPO_URL>

## Reproduce the final CSV on the competition data (CPU, ~3 min)

1. Kaggle → New notebook → File → Import → `final/UMUD_FINAL_F1_KAGGLE.ipynb` (or `_F3_`).
2. Add Input → Competitions → *UMUD Challenge*. The notebook finds the 309 test images by file name anywhere under
   `/kaggle/input`.
3. Run all → `/kaggle/working/submission.csv`. The log prints the checks read from the images (groups 309/309, PNG labels,
   scale) and the SHA-256 of the CSV.

### Checks without the competition images

`python tools/run_final_local.py F1 --no-img` (or `F3`) runs a final notebook locally without the `img` token, so the
embedded tables alone determine the CSV:

| run | SHA-256 of `submission.csv` |
|---|---|
| F1 without `img` | `73d6a7123683d04a3206affcd70b4e9f92382361565d0f13d941524ef993a5ef` (same CSV as Kaggle probe BE5, public 0.20497 = F1) |
| F3 without `img` | `4b535c1e5a93dab65d69dd4ccdfd2ec7527a65b2691975a055e31fb656411dd8` |

With `img`, FL and MT are rescaled only on images whose ruler reading differs from the frozen scale registry. On the test set
that is expected for one image (×1.0012), which is why F1 and BE5 have the same public score.

## Method in one paragraph

Screen layout gives the device group (Siemens, Telemed ×2, grey-scale, Lumify PNG). The on-screen ruler gives the physical scale
(px/cm), and template matching reads the muscle label on PNG screenshots. Consecutive 5-frame video blocks are detected
too. Segmentation networks trained on the challenge training data (DL_Track-style masks) give:
- aponeuroses: R12B (DINOv2-frozen encoder + decoder), v2 (5 folds) and A76 (5 folds);
- fascicles: R22 (ConvNeXt-T U-Net with an axial orientation field, 5 folds), F71 (5 folds) and F86 (ConvNeXt-S);
- R17: 4 models for an independent PA/FL estimate.

Fascicle maps are reduced to straight fragments (skeleton + line fit). FL is the mean of the 5 least-extrapolated, PA-coherent
fragments extended to the inner edges of the aponeurosis pair (lex5). PA comes from decoders at the deep aponeurosis.
MT is the inner-edge distance. The per-image measurements are stored as tables and combined by a fixed token pipeline:
group levels, aspect corrections, shrinkage towards group means, video-block smoothing and FL/PA/MT geometry coupling.

## Pipeline (training → inference → tables → submission)

All stages are Colab notebooks (GPU L4/A100) with code packed inside, SHA-256 checks of every checkpoint and resumable runs.
Inputs are the challenge training data and test images on Google Drive.

| # | notebook (`pipeline/`) | trains / runs | feeds |
|---|---|---|---|
| 1 | `UMUD_R12B_MT_OVERNIGHT_…_v1_0_3_RUNTIME_HARDENED.ipynb` | R12B aponeurosis model | MT base, aponeurosis pairs |
| 2 | `UMUD_R17B_TRENING_COLAB.ipynb`, `UMUD_R17C_OCENA_I_PREDYKCJE_COLAB.ipynb` | R17 fascicle/aponeurosis models (develop, refit) | FL base (R17 refit) |
| 3 | `UMUD_R20_PA_DIAGNOSTYKA.ipynb` | R20 geometry (PA local support) | PA base (PNG and 6 TIFF) |
| 4 | `UMUD_R22_A_TRAIN_CONVNEXT_5FOLD.ipynb` (+ `_RECOVERY_AMP`, `UMUD_R22_B_ENSEMBLE_AND_SUBMISSIONS.ipynb`) | R22 fascicle model, 5 folds | fragments, PA decoders, BAND |
| 5 | `UMUD_R45T_ROZCIEGNA_KIERUNEK_COLAB.ipynb` | v2 aponeurosis model (5 folds); late R22 PA decoder | PA base (238 TIFF) |
| 6 | `UMUD_R50T_ZESPOL_R17_COLAB.ipynb` | R17 ensemble (4 models) on test | R17DEV, R17ENS |
| 7 | `UMUD_R71T_PECZKI_TRENING_COLAB_v1_1.ipynb` | F71 fascicle model (5 folds); exports: R22 / F71 fragments (stages S, E), decoders with R12B and v2 pairs (stage D5) | FX89, BAND, PADEC, V2MT, V2OV, FX72 |
| 8 | `UMUD_R76T_A76_F76_TRENING_COLAB.ipynb` | A76 aponeurosis model (5 folds) | FX85 (MT), aponeurosis edges |
| 9 | `UMUD_R80T_KRAWEDZIE_COLAB.ipynb` | A76 + v2 map-mixture edges (`a76_v2mix`) | FX89, FX110, FX132 geometry |
| 10 | `UMUD_R86T_TRENING_F86_COLAB.ipynb` | F86 fascicle model | FX110/111 (F86 part) |
| 11 | `UMUD_R93T_ODZYSK_BAZY_COLAB.ipynb` | re-runs the R20 / R17 / R12B base code on test | RBASE (base PA/FL/MT, `rbasek`) |
| 12 | `UMUD_R96T_FOLDY_RADON_COLAB.ipynb` | R22 / F71 folds decoded separately (no training) | FX132 (F3) |
| 13 | `final/UMUD_FINAL_F1_KAGGLE.ipynb`, `final/UMUD_FINAL_F3_KAGGLE.ipynb` | token pipeline + image reading (CPU) | `submission.csv` |

Notebooks 1–4, 7 (v1.1) and 8 are kept next to this package in the Drive folder `pipeline_z_Drive` (numbered as in the
table) and go into `pipeline/` when the repository is published; `pipeline/` here holds stages 5, 6, 9–12 and the
first R71T version.

Every notebook writes its run folder on Drive (`UMUD_Challenge/outputs/<RUN>`), with `ENVIRONMENT.json`, a log and
`DONE.json` / `TRAINING_COMPLETE.json` holding the SHA-256 of each checkpoint. Table builders are in `src/umud_final/`
(`tables.py` = BAND, PADEC, V2MT, V2OV, FX85, FX89 from the exports; `scale.py`, `layout.py` = image source).
`src/embedded/` holds the exact image-reading code that the final notebook runs (`scale.py`, `png_muscle.py`).
Which table comes from which export and which code: `PROVENANCE.md`.

## Running on a new dataset

The final notebook combines per-image tables keyed by image name. For new images:
1. Run inference stages 5–12 on the new images; the trained weights from stages 1–10 are reused.
2. Build the tables with `src/umud_final` and the builders listed in `PROVENANCE.md`.
3. Embed the tables in the final notebook and run it with the same `VARIANT`.

All constants stay fixed. The status of each builder (in the repository or still to be restored) is listed in
`PROVENANCE.md` and `TODO_NEW_DATASET.md`.

## Licence

Code: GPL-3.0-or-later (`LICENSE`). Model weights were trained only on the challenge training data and are shared under the
challenge data terms. No third-party pretrained weights beyond ImageNet / DINOv2 backbones (timm / torchvision).
