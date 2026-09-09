# Effective input-current dynamics

`InputCurrentNeuron` inherits PAULA's membrane, spike/reset, modulation,
postsynaptic adaptation and retrograde mechanisms. It adds decaying current
components at explicitly selected input ports. Ordinary `Neuron` instances
retain their previous equations and constructor defaults.

```python
from neuron.extensions.experimental.input_current import InputCurrentNeuron

# Construct the cell and its normal PAULA ports first.
cell = InputCurrentNeuron(neuron_id, params, log_level="CRITICAL")
# ... install postsynaptic points, distances, terminals and source bindings ...
cell.configure_current_kernel(
    ports=[0, 4], decay_ticks=[10.0, 48.0],
    peak_fractions=[0.82, 0.18], normalization="area",
)
```

The numbers above illustrate the API. They are not default biological
parameters. Other ports bypass the kernel, allowing a separate experimental
current-injection port. Signed arriving potentials remain signed.

Each queued arrival adds its original, pre-learning potential once, after
native dendritic attenuation. Between arrivals each component decays by
`exp(-dt/tau)`. The inherited membrane integrates their sum. A spike resets
the soma, not the input current. A current tail does not create extra receptor
inputs, plasticity updates or retrograde events. Changed firing can still
change subsequent native adaptation and the surrounding neural network.

Peak normalization preserves the impulse's initial current amplitude. Area
normalization divides the whole kernel by
`sum(fraction / (1 - exp(-dt/tau)))`, preserving its shape and the native
impulse's discrete integrated charge. With a long tail these interventions
can differ greatly in total charge. Neither normalization calibrates PAULA
current to amperes.

Inspect `arrived_port_impulse`, `last_port_current`, `current_state`,
`current_ports` and `total_current`. The state array has one row per filtered
port and one column per decay component. Calls require consecutive ticks and
constant positive `dt` until reset. The existing network reset hook clears
these states without erasing learned coefficients. Deepcopy preserves both
the current state and the inherited in-flight queue.

This is an experimental effective current kernel. It does not identify a
receptor subtype, include conductance driving force or short-term vesicle
depression, represent a dendritic cable, or simulate a voltage-clamp electrode.
Its instantaneous rise and time-unit convention must remain explicit in an
assay. A fit to a somatic EPSC tail cannot identify these missing mechanisms.

Verification includes analytic tails, discrete charge conservation, overlapping
signed arrivals, delayed pre-learning potentials, unchanged current-injection
responses, positive ongoing adaptation, reset and deepcopy. A separate
1,600-tick comparison against the preceding committed `Neuron` implementation
matched ordinary states, weights, queues and events exactly. A warmed,
alternating eight-pair benchmark of 20,000 ticks gave medians 0.283 s before
and 0.278 s after extracting the current hook. This small noisy difference is
not evidence of a speed improvement.
