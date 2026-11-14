# C1 source update

This update replaces historical R96 replay presets with the complete C1 image-inference presets, adds the exact source runtime and all nine previously missing training notebooks, and records the public repository URL and current notebook checksums.

Current profiles: C1_F1 public 0.20486 and C1_F3 public 0.20494, as reported by the user. Historical hashes 73d6a712... / 4b535c1e... are R96 without-img replay hashes, not current C1 results.

Publication scope is source code, notebooks and method/model metadata. Private run logs, prediction CSVs, comparison tables, cached measurements and old vendor CSV exports are not included in this update. Public tests use synthetic rows and source/model integrity checks. Original public R96 files remain in Git history.

The frozen mathematical/neural source is unchanged from the completed local C1 run. Model binaries and actual Kaggle final selections remain external steps. The new notebook wrappers have not been rerun on a Kaggle GPU.
