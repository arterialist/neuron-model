# CIFAR comparison experiments

This directory preserves a local experiment design and its Python runners that
were previously excluded by the repository ignore rules. They compare fixed
patch dictionaries and learning rules before proposing a PAULA model change.
They do not modify the PAULA neuron equations or calibration.

Use the repository lockfile and run modules from `neuron-model/`:

```bash
uv sync
uv run python -m cifar_ceiling.run_kmeans_ceiling --help
uv run python -m cifar_ceiling.run_learned_ladder --help
uv run python -m cifar_ceiling.analyze_ceiling --help
```

A sweep requires a retained canonical calibration with matching settings and
source hashes, using the full training dataset and falling within the declared
reference tolerance.

The acquisition runners require an already downloaded CIFAR-10 dataset under
`--data-root`. Their loader sets `download=False`. Full runs are expensive and
are opt-in. Choose a new `--out` directory when changing a protocol. Cached
records must match the requested protocol; legacy records without that metadata
must be inspected independently instead of reused for a new run.

The design file is a historical proposal. Hypotheses, published reference
numbers, and the approximate 62 percent PAULA baseline in it are context from
that proposal, not newly reproduced results. One local calibration record was
found during the October 2026 publication audit. Its single-seed test accuracy
was 69.55 percent at K=400. It was not rerun, and no completed ladder campaign
was found in this directory. Generated records, figures, and local CIFAR data
remain ignored.

The historical implementation uses a regularized multinomial logistic readout;
the cited Coates and Ng implementation used an L2 SVM. A calibration tolerance
is an instrument check, not proof that the published protocol was replicated.
The Krotov arm applies the learning rule to whitened patches, not the original
whole-image protocol. The CNN arm is an unmatched reference. Do not infer a
PAULA mechanism or generalization result from these external comparison arms.

Offline checks use tiny synthetic arrays and temporary result directories:

```bash
uv run python -m unittest discover -s tests -p 'test_cifar_ceiling.py' -v
```

These checks cover split isolation, preprocessing agreement, encoding, actual
seed pairing, cached-protocol validation, and small runner paths. They do not
acquire CIFAR data or reproduce historical accuracy.
