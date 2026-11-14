# Public C1 tests

Run `python -m unittest discover -s c1/tests_public -v` from repository root. Tests use source checksums, checkpoint metadata, notebook source packages and synthetic CSV rows. They do not read or contain private run logs, predictions, labels or images. No GPU or checkpoints are required.
