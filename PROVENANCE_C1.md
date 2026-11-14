# C1 method provenance

C1 combines R12B/v2/A76 aponeurosis models with R17/R22/F71/F86 fascicle models, geometry and fixed calibration. Exact checkpoint hashes and fold membership are in `c1/models/WEIGHTS_MANIFEST.json`.

The complete local C1 run used T4. The frozen source hash is af622512b9883b87aa1fe95f026ec882f95791266467abbc4acf49e6f93771c9. Current wrappers embed that exact source and compute all measurements from supplied images and checkpoints; they contain no saved prediction CSVs.

F71 PA uses all five folds; FL uses E4 folds 0,2,3,4. Runtime profiles remain byte-frozen at `c1/final/VARIANTS.json`; publication metadata lives separately in root `final/VARIANTS.json`.

Public-LB feedback informed calibration constants and variant selection. Fixed PNG word-shape templates originate from six Lumify test screenshots; the method's layout support is explicit. Rejected manual/AI calibration annotations were not added.

Training/inference notebooks are under `pipeline/`. Source license: GPL-3.0-or-later. Binary checkpoint licensing and Dataset publication remain separate. Public tests use synthetic data and metadata; private run logs and prediction/comparison exports are not included. The new wrappers were not rerun on a Kaggle GPU.
