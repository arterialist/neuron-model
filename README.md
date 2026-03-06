# PAULA: Predictive Adaptive Unsupervised Learning Agent

> **Empirical Validation of the ALERM Framework**
>
> _Part of the Artificial Life Research Initiative._
>
> 📄 **PAULA Paper:** [PAULA: A Computational Substrate for Self-Organizing Biologically-Plausible AI](https://al.arteriali.st/blog/paula-paper)  
> 📄 **ALERM Framework:** [The ALERM Framework: A Unified Theory of Biological Intelligence](https://al.arteriali.st/blog/alerm-framework)

This repository contains the implementation of **PAULA** (Predictive Adaptive Unsupervised Learning Agent)—a radically different approach to neural network modeling. It serves as the empirical validation for the **ALERM** (Architecture, Learning, Energy, Recall, Memory) mathematical framework. Unlike traditional deep learning (backpropagation, dense matrices), PAULA uses purely local learning rules, extreme sparsity, and temporal/spiking dynamics.

This isn't just another neural network implementation—it's a **virtual laboratory** for experimenting with the building blocks of mind.

---

## What This Repository Contains

### Core Implementation

- **Neuron model** (`neuron/`): Graph-based neurons with vector communication, adaptive plasticity, and temporal dynamics
- **Network builder** (`build_network.py`, `snn_classification_realtime/network_builder_direct.py`): Build sparse conv + dense architectures from YAML configs
- **Activity pipeline** (`snn_classification_realtime/`): Record network dynamics, train SNN readout classifiers, run real-time evaluation

### Virtual Laboratory

- **Interactive CLI** (`cli/`): Command-line interface for direct experimentation
- **Web visualization** (`cli/web_viz/`): Real-time network animation (WebSocket + Cytoscape.js)
- **Interactive training** (`interactive_training.py`): Build networks interactively and explore topologies

### Experimental Platform

- **Pipeline** (`pipeline/`): Batch experiment runner (partially functional)
- **Ablation testing** (`neuron/ablation_registry.py`): Isolate ALERM components for empirical study

---

## For Researchers: Paper → Code Entrypoints

If you are coming from the ALERM/PAULA papers, this repo contains the codebase used for the empirical studies. High-level mapping:

| ALERM Concept                     | Where to Look                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| **Architecture (A) ↔ Memory (M)** | `build_network.py`, `snn_classification_realtime/network_builder_direct.py`, `networks/` |
| **Learning & Plasticity (L)**     | `neuron/neuron.py`, `neuron/ablation_registry.py`                                        |
| **Energy (E) ↔ Recall (R)**       | `snn_classification_realtime/realtime_classifier/evaluation.py`, `evals/`                |

To reproduce the reported MNIST results or run ablations, see the **[Reproducibility Guide](REPRODUCIBILITY.md)**.

---

## Getting Started

```bash
git clone https://github.com/arterialist/neuron-model.git
cd neuron-model

# Install (recommended: use uv for fast dependency management)
uv sync
# or: pip install -e .
```

**Quick exploration:**

```bash
# Launch interactive CLI
python cli/neuron_cli.py

# Build a network from YAML and run the full pipeline
# See REPRODUCIBILITY.md for step-by-step instructions
```

---

## Legacy & Experimental Components (⚠️ Warning)

This repository evolved rapidly. Some tools are useful but **unstable/experimental**:

- **CLI & Web Viz** (`cli/`, `cli/web_viz/`): Functional for exploring topologies and attractor dynamics; may have bugs
- **Pipeline** (`pipeline/`): Half-functional batch runner; contributions welcome
- **Modality Processor** (`modality_processor/`): Untested stub for multi-modal input; not part of core validation

---

## Join the Journey

Here's the truth: this started as a one-man obsession, but it's grown into something bigger than what I can do alone. I've spent countless hours developing the theoretical foundations and turning abstract ideas into working models—but the really exciting discoveries, the ones that could change how we think about intelligence, will come from all of us working together.

If you're reading this and feeling that spark of curiosity, that's exactly what I was hoping for.

### How You Can Help

I'm looking for fellow travelers on this journey:

**If you're a researcher:** Run experiments I haven't thought of. Push the model to its limits. Find the edge cases that reveal new insights. See `REPRODUCIBILITY.md` for guidance on reproducible experiments.

**If you're a developer:** Help make this tool more powerful. There are performance optimizations waiting to be discovered, visualizations that could reveal hidden patterns, interfaces that could make exploration more intuitive.

**If you're curious:** Ask questions. Report what seems broken. Tell me what doesn't make sense. Some of my best insights have come from questions that seemed "obvious."

**If you're a theorist:** Challenge the foundations. Propose new experiments. Help me understand what this model is really telling us about intelligence.

- Found something broken? Open an issue. I want to know.
- Have an idea for an experiment? Start a discussion. Let's design it together.
- Built something cool? Share it. Show me what you discovered.
- See the bigger picture: [al.arteriali.st](https://al.arteriali.st)
