# Recovered SNN-debug tools — July 2026

These are the last recoverable versions of scratchpad tools created during the
Claude SNN/CIFAR-10 debugging sessions. They have been promoted from temporary
session storage into the foveation experiment tree.

- `diagnostics/` holds smoke tests and diagnostic probes.
- `launchers/` holds reproducible experiment-launch scripts.
- `reports/` holds the generated mini-brain findings report.
- `scripts/` holds artifact builders, experiment helpers, and analysis tools.

The primary implementation files from these sessions were already present at
their original repository paths. Only a missing active-inference decoder was
restored directly; existing files with later divergent content were retained.

The launchers resolve the repository root relative to their own location. The
attractor-class launcher is supported by compatibility aliases in
`minibrain/exp_attractor_class.py`, so its historical `reservoir_retino` and
`reservoir_random` arms run against the maintained mini-brain implementation.

All recovered diagnostics and launchers pass syntax checks. The maintained
attractor-class CLI also accepts both historical reservoir aliases; the
retinotopic alias maps to the existing retinotopic reservoir and the random
alias maps to its random-wiring counterpart. Full CIFAR experiments remain
deliberately opt-in because they are long-running and generate sizable output.
