# Historical network configurations

These 15 YAML files were local experiment inputs before the October 2026
publication audit. Their parameter values and layer definitions are preserved.
They are examples and historical recipes, not validated configurations for the
current embodied brain or recommendations to change its neuron model.

The CIFAR launchers in `../scripts/` refer to the five numbered colour configs.
The `cifar10colored_v5.yaml` file retains the historical internal name
`cifar10colored_v4`. Its launcher passes `--name cifar10colored_v5` when building.
Use an explicit `--name` and a new output directory if invoking the builder by
hand so this preserved naming mismatch does not overwrite a V4 network.

The `mnist_*` and `paula_*` files use `networks_agent/`; generated JSON networks
remain ignored. The grayscale 20-layer configuration may create a large network.
Listing these inputs does not assert that the archived runs completed or that
their reported accuracies were independently reproduced.

Build an explicitly selected config with the locked environment:

```bash
uv run python build_network.py --config configs/test_small.yaml --output-dir /tmp/new-neuron-network --name test_small
```
