# Historical CIFAR run recipes

These scripts preserve the configuration choices and artifact names used by
local CIFAR colour experiments. Run them with Bash from a checkout with `uv`.
Python commands use the repository lockfile through `uv run python`.
`TRAIN_DEVICE` overrides the historical `mps` training default.

These are long-running recipes, not smoke tests. Some acquisition/build steps
are commented out because those historical runs reused existing artifacts.
Review the script and its input paths before invoking it. They retain fixed run
IDs and output names for provenance; choose new names before starting a new
campaign. Running a script against an existing output tree can replace generated
artifacts. The visualization recipe also requires archived local recordings.

The old checkpoint comment describes selection on the training pipeline's test
loss. That selection split is not an independent final holdout, so its score
must not be presented as one. The publication audit did not rerun these
campaigns or change their neuron parameters or protocol.
