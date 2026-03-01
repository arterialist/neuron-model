# Network YAML Config Guide

Reference for building spiking neuron networks from YAML configuration files. Used by `build_network.py`.

---

## Structure

```yaml
dataset: mnist
output_dir: networks
name: my_network

layers:
  - type: conv
    kernel_size: 4
    stride: 2
    connectivity: 0.8
  - type: conv
    kernel_size: 4
    stride: 2
    connectivity: 0.8
  - type: dense
    size: 144
    connectivity: 0.25
```

---

## Top-Level Fields

| Field | Required | Description |
|-------|----------|-------------|
| `dataset` | yes | Input dataset: `mnist`, `cifar10`, `cifar10_color`, `cifar100`, `usps`, `svhn`, `fashionmnist` |
| `layers` | yes | List of layer definitions (conv and/or dense) |
| `output_dir` | yes* | Output directory for the JSON file (*or provide via `--output-dir` CLI) |
| `name` | no | Output filename without extension (default: config filename stem) |
| `inhibitory_signals` | no | Enable inhibitory signals (default: false) |
| `rgb_separate_neurons` | no | Separate RGB channels in neurons (default: false) |
| `input_size` | no | Override input size (default: 100) |

---

## Layer Types

### Convolutional (`type: conv`)

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `kernel_size` | no | 3 | Convolution kernel size (also accepts `kernel`) |
| `stride` | no | 1 | Stride |
| `connectivity` | no | 0.8 | Fraction of possible synapses to create (0–1) |

**Important:** The `filters` parameter must not be used. Do not specify `filters` in convolutional layer definitions.

### Dense (`type: dense`)

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `size` | no | 128 | Number of neurons in the layer |
| `connectivity` | no | 0.5 | Fraction of possible synapses (0–1). Use ~0.25 for stable attractors |
| `synapses_per` | no | — | Override synapses per neuron (advanced) |

---

## Layer Ordering

- Conv layers must precede dense layers. Conv-after-dense is not supported.
- The first layer determines input handling: if first is conv, an input layer is created from the image; if first is dense, the image is flattened.

---

## CLI Overrides

```bash
python build_network.py --config path/to/config.yaml [--output-dir DIR] [--name NAME]
```

- `--output-dir` overrides YAML `output_dir`
- `--name` overrides YAML `name`
- At least one of `output_dir` (YAML or CLI) must be set

---

## Output

`{output_dir}/{name}.json` — network configuration suitable for activity recording and evaluation.
