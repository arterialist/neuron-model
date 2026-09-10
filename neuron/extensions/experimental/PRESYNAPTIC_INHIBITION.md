# Local presynaptic inhibition

`PresynapticInhibitionNeuron` extends ordinary PAULA without changing its
defaults. It adds a declared connection between a cell's inhibitory receiving
ports and selected release terminals. The neuron still integrates the ordinary
somatic currents, generates its own spikes, updates native receiving weights,
and emits/processes native retrograde events.

For each declared port, the incoming positive signal amplitude multiplies the
magnitude of its current negative receiving coefficient, including `u_i.plast`.
The mechanism uses that port's ordinary propagation delay and attenuation.
These delayed impulses drive one intracellular state:

```
x[t] = exp(-dt / decay_ticks) * x[t-1] + arriving_inhibitory_drive[t]
fraction[t] = 1 / (1 + gain * x[t])
selected_terminal_release = native_terminal_release * fraction[t]
```

This is an effective decaying inhibitory drive followed by an assumed
quasi-steady release-suppression law. The algebraic suppression law is supplied,
not learned or discovered by the neuron. Any circuit transfer curve follows
from composing this local hypothesis with actual synapses, membrane thresholds
and activity. It does not establish self-organized gain control.

The same input retains its original somatic effect. Gating is an additional
declared intracellular action. It does not replace the inhibitory current,
overwrite synaptic weights, impose source spike rates or consult population
activity, neuron labels, stimulus identity or experimental outcome.
Configuration selects existing ports and terminals before execution.

The extension rejects changing receptor polarity rather than silently rectifying
a learned excitatory coefficient into inhibition. Native learning remains active.
The gain-zero path preserves native event coefficients exactly. The reset hook
clears the intracellular state and pending gate arrivals, but not learned weights.
The common state modulates all declared terminals equally; this is not a spatial
terminal reconstruction. Other terminal amplitudes and all return events pass
through unchanged. Neuromodulatory vectors are not rescaled here.

## Biological motivation and limits

[Olsen and Wilson, 2008](https://doi.org/10.1038/nature06864) found substantial
presynaptic lateral inhibition at fly ORN terminals, involving ionotropic and
metabotropic receptors. That supports testing receptor-to-release regulation.
It does not identify this equation, its gain, its time constant, its spatial
scope, or the effect of every individual connectome edge.

The first linked assay declares gain 0, 0.1 or 1 and a decay of 100 nominal
ticks. None is a fitted physiological parameter. Native spike shape, calcium
influx, distinct receptor kinetics, vesicle pools and receptor adaptation are
not represented by this extension. Composition with the separate terminal
depression extension has not been tested; do not silently substitute one for
the other.

The matched zero-gain tests check native state, receiving learning and events.
Nonzero-gain tests check delayed suppression, selective terminals, decay,
retained spikes under matched input, preserved return events and failure on
invalid state. These are implementation checks. Functional evidence belongs
to the connected transfer assay in `active-inference/simulations/drosophila/`.
