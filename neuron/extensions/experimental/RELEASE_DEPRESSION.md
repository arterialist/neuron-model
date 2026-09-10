# Use-dependent terminal release

`DepressingReleaseNeuron` is an opt-in subclass of the ordinary spiking PAULA
neuron. It adds a recoverable available fraction to selected output terminals.
Native soma dynamics, receiving coefficients, retrograde updates and outgoing
modulatory metadata remain governed by PAULA. Existing constructors and default
neurons do not change.

After constructing the neuron's ordinary synapses, configure existing terminal
IDs before ticking:

```python
cell.configure_release_depression(
    terminals=[0, 3], depletion_fraction=0.22, recovery_ticks=893.0
)
```

Each terminal starts with `R=1`. Before a tick, it recovers by
`R += (1-R)*(1-exp(-dt/recovery_ticks))`. A native release of amplitude `w`
becomes `w*R`; that event then changes the available fraction to `R*(1-u)`.
Other terminals and all return events pass through unchanged. A depleted
terminal does not stop the soma firing or freeze long-term learning.

This is an effective single-resource model. The example values correspond to
the simple depression control in Figure 1 of
[Nagel, Hong and Wilson, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4289142/).
Their paper shows why that simple control is insufficient for sustained
olfactory transmission. This extension does not reproduce their separate
fast/slow components or presynaptic inhibitory regulation. Mapping a tick to a
millisecond, transferring parameters across cell types, and selecting terminals
are experiment-level assumptions, not properties inferred by the neuron.

The cell receives no odor identity, stimulus schedule, target behavior or global
error. Its added dynamics depend on its own releases and elapsed local time.
`u=0` passes native forward events through without changing their numeric type.
The regression test verifies exact native states and events with active
receiving plasticity and return events.

Inspection fields are `release_available`, `release_used_fraction`,
`release_native_amplitude` and `release_effective_amplitude`, indexed by
`release_terminals`. Used fraction and amplitudes are zero on ticks without
release. Available fraction is the value after recovery and any event depletion.

Consecutive tick indices and a constant positive `dt` are required. `dt` cannot
exceed the ordinary membrane integration constant. `reset_simulation()` restores
resources through `reset_additional_state()` while preserving learned weights.
Deep copies preserve the ongoing resource trajectory. Full checkpoint
serialization and combining this class with other experimental subclasses have
not been validated.

Tests cover exact native behavior at zero depletion, event-by-event analytical
recovery at two time steps, per-terminal selection, continued adaptation,
copy/reset behavior and invalid timing/configuration. The connected fly
experiment lives in `active-inference/simulations/drosophila/orn_train.py`.
Those tests establish implementation properties, not biological validity.
