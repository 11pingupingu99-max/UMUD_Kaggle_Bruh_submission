# Frozen C1 runtime

Active source SHA-256: af622512b9883b87aa1fe95f026ec882f95791266467abbc4acf49e6f93771c9.
CLI: `python c1/tools/configure_inference.py ...` then `python c1/tools/run_fresh.py --config ...`.
Tests: `python -m unittest discover -s c1/tests_public -v`. These use only synthetic rows and public source/model metadata.
Private run exports, logs, prediction CSVs and comparison tables are deliberately not included. Binary checkpoints must be supplied separately.
