# Preserved July 2026 helper scripts

These scripts were recovered from the SNN/CIFAR debugging sessions. Their
experiment definitions and hard-coded historical findings are preserved for
inspection; publishing them does not reproduce or endorse those findings.

Run Python modules from `neuron-model/` with `uv run python -m`, followed by
`snn_classification_realtime.foveation.recovered_snn_debug_tools.scripts.NAME`.
The two artifact builders accept `--input-root` and `--output`, read existing
figures, and create a new report without overwriting one. Their reports visibly
identify the text as historical. They do not run simulations or recalculate
numbers. The original recovered HTML report is retained separately unchanged.

`exp_teacher_modes` and `probe_memory_content` start historical CIFAR experiments
when executed. They are opt-in and are not import-time jobs. The teacher probe
uses training data for both representation learning and its readout diagnostic,
selects the best regularization on the diagnostic holdout, and may reuse image
indices. Its printed score is exploratory separability, not independent test
accuracy. No change to that historical method was made during publication.

`progress` reads completed local records and reports only what it finds. It no
longer assumes that 15 unrelated historical arms have finished. It accepts an
explicit `--input-root` and does not start jobs.
