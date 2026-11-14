# UMUD Challenge — final C1 image-inference submission

Repository: https://github.com/11pingupingu99-max/UMUD_Kaggle_Bruh_submission

The current method computes pennation angle (PA), fascicle length (FL) and muscle thickness (MT) from the supplied ultrasound images and the exact 30 own trained checkpoints. No host labels, fixed test-prediction tables or old image-ID registry are read by the C1 inference path.

## Current profiles

| profile | reported public score | notebook | CSV SHA-256 from the completed C1 run |
|---|---:|---|---|
| C1_F1 | 0.20486 | `final/UMUD_FINAL_F1_KAGGLE.ipynb` | `e936f2231644be55e2630e8b198b3fdb70461617f75dfc944495604f7aeca8f8` |
| C1_F3 | 0.20494 | `final/UMUD_FINAL_F3_KAGGLE.ipynb` | `8e1cfd98cb8af629f5098f9e073344cceca0e3c440efd83cabc3800e2847c613` |

Scores were reported by the user on 14 November 2026 for the complete 309-image C1 T4 run. The recommended pair is C1_F1 + C1_F3; actual final-selection IDs were not supplied and are not asserted here. The CSV hashes beginning `73d6a712` and `4b535c1e` belong to historical R96 **without `img`** replays, not C1. Their original public versions remain available in Git history and `final/original/UMUD_R96_KAGGLE.ipynb`.

## Run full inference

**Kaggle:** import either final notebook, select a GPU (the reference run used T4), and Add Input for the competition data and the unpacked `UMUD_C1_WEIGHTS` Dataset. Run all; the selected profile is saved as `/kaggle/working/submission.csv`. Both profiles are computed in one pass. Internet is needed only for missing dependencies; local wheels may be supplied. The notebook does not replace the installed torch/torchvision CUDA pair.

**Colab:** the same notebook can run on a GPU with Drive mounted. If no portable weight package exists, it uses the original exact checkpoint paths from the verified C1 model registry. No training is needed.

**Weights:** run `final/UMUD_WAGI_FINAL_COLAB.ipynb` on CPU to collect and SHA-verify all 30 existing Drive checkpoints. Create a Kaggle Dataset from the unpacked folder, preserving its subdirectories, and add it as Input. Binary checkpoint files are not included in this source archive, and no public Dataset URL has been supplied. The packaging notebook does not upload anything.

**CLI:** the complete, byte-frozen C1 runtime is isolated under `c1/` to avoid mixing it with historical R96 modules:

```bash
python c1/tools/configure_inference.py --images-dir /data/images --weights-dir /data/weights --output-dir /work/c1
python c1/tools/run_fresh.py --config /work/c1/config_c1.json --smoke 6
python c1/tools/run_fresh.py --config /work/c1/config_c1.json
```

Install a matching torch/torchvision pair and the requirements in `c1/requirements.txt`. Versions from the successful T4 run are recorded in `c1/environment_c1_gpu.json`. Different numerical/CUDA versions can change decoding; the new wrappers were not rerun on a Kaggle GPU.

## Code, checks and provenance

C1 includes all base/table builders and the neural driver. Its source hash remains exactly `af622512b9883b87aa1fe95f026ec882f95791266467abbc4acf49e6f93771c9`, matching the full successful run. `c1/final/VARIANTS.json` is the frozen runtime input; the root `final/VARIANTS.json` records current publication metadata and notebook hashes and is not read by the runtime.

```bash
python -m unittest discover -s c1/tests_public -v
```

Public tests verify frozen source/model metadata, both notebook packages and exact-byte CSV handling on synthetic inputs. They need no GPU, images, labels or private prediction tables. The completed C1 run was checked privately; its raw logs, comparison tables and prediction exports are not published in this repository.

There are **16 historical training/inference notebooks in `pipeline/`**, including all nine numbered notebooks previously held only in the Drive folder `pipeline_z_Drive`: R12B v1.0.3, R17B, R17C, R20, three R22, R71T v1.1 and R76T. Their exact names, byte sizes and hashes are in `pipeline/RESTORED_FROM_DRIVE.json`. No training was rerun during this update.

Historical R96 root modules and replay tools are retained for provenance. They are not the C1 inference entry point. Additional historical vendor CSV exports are not part of this update. The fresh runtime does not consume them.

Fixed calibration constants were informed by public-LB feedback; they are not represented as OOF-derived values. PNG glyph templates originate from six Lumify test screenshots. Rejected manual/AI calibration annotations were not incorporated. Supported screen layouts and batch-dependent group statistics remain explicit. Organizer eligibility is not claimed to have been approved.

Code license: GPL-3.0-or-later. Metadata: `CITATION.cff`, `codemeta.json`. Training/data/checkpoint publication terms must be documented with the actual weight release. See `PROVENANCE_C1.md`, the preserved historical provenance and pipeline notebooks.

`release/FINALIZATION.json` records current scores and recommended profiles. `GITHUB_UPDATE_REVIEW.md` explains the R96-to-C1 correction. This source update is published in the repository linked above. Binary weight publication and actual Kaggle final selection remain external steps.
