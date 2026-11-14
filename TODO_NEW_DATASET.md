# Current new-dataset status — C1, 14 November 2026

The complete driver and base/FX table builders are now supplied in `c1/`; they are no longer missing.
Use the current full-inference notebooks in `final/`, or `c1/tools/run_fresh.py`.
Runtime input may have new names and a different number of images. Supported device layouts/rulers are required; an explicit metadata manifest may supply independently known scale and clip ordering. Group statistics depend on batch composition.

Remaining external steps:
- Collect the 30 exact checkpoint binaries using `final/UMUD_WAGI_FINAL_COLAB.ipynb` and add them as a Dataset for Kaggle.
- Source is published in this repository. Publish the weights with their terms/provenance and record the Dataset version/URL.
- Select the two already measured C1 submissions on Kaggle and retain their IDs.

Historical R96 without-img CSVs are not the C1 control outputs. See root README for the correct hashes and scores.
The new wrappers have not been rerun on a Kaggle GPU.
