# Passive cable with local graded release

`LocalCableGradedNeuron` is an opt-in subclass of PAULA's `GradedNeuron`.
It keeps one neuronal identity, native input/output ports, delayed local
potentials, synaptic coefficients, neuromodulation and retrograde events.
It adds a voltage state for every supplied tree node and derives terminal
release from specified local output sites. It has no stimulus labels, learned
decoder or external action policy.

This is an experimental cellular hypothesis. It does not implement ion channels,
calcium, vesicle dynamics, contact-specific learning or measured electrical
properties of a named neuron. Native neuromodulation remains neuron-wide.
Because the cell does not spike, inherited `t_last_fire` stays at negative
infinity and its timing-dependent plasticity is not a validated graded-cell
learning rule. Positive learning rates remain active.

## Geometry and equation

Pass `PassiveCable` an acyclic connected tree, coordinates and radii in µm,
and an assumed ratio of specific membrane to axial resistance, `Rm/Ra`, in µm.
Each edge is modeled as a uniform cylinder at the mean of its endpoint radii.
Half of each edge's lateral membrane area belongs to each endpoint. This makes
sealed ends and branch loading explicit, but is not a converged tapered-cable
model. A zero endpoint radius is retained; an edge with both radii zero or zero
length is rejected. No geometry is healed or deleted by this extension.

Let `C` be the diagonal matrix of normalized membrane areas, summing to one.
An edge of length `l` and mean radius `r` contributes
`(Rm/Ra) π r² / (l × total_area)` to the axial Laplacian `L`.
For `a = dt/lambda_param`, the update is

```text
(C + a L) v_next = (1-a) C v + a I
```

The explicit membrane leak is PAULA's existing update. Axial exchange is an
implicit passive solve. For `0 < a <= 1`, the homogeneous update is stable and
axial exchange conserves the area-weighted potential. At equilibrium the
equation is `(C + L) v = I`, independent of timestep. This is not a claim of
converged transient timing. `lambda_param` remains measured in simulator ticks;
no milliseconds-per-tick calibration is supplied.

The normalization preserves the aggregate model's whole-neuron current scale.
It does not calibrate current in amperes or voltage in millivolts. Concentrating
the same aggregate current into a small membrane region can produce large local
voltages and early release saturation. That is a prediction of these assumptions,
not evidence that the assigned current density is biologically correct.

## Neural interface

Configure the neuron after its ports are created and before its first tick.
`input_ports/input_nodes` list each input contact. The port's aggregate native
arriving potential is divided among its contact entries. Every anatomical input
must have sites. The extra `uniform_input_port` is reserved for an explicitly
uniform experimental current per membrane area, separate from anatomical ports.

`terminal_ports/terminal_nodes` list output sites. Each terminal reads the mean
of `graded_gain × max(v - graded_S0, 0)`, capped at `graded_max` when positive,
across its sites. Its native terminal coefficient then multiplies that mean.
Thus a receiving cell can retain its existing count-scaled weight without
double-counting the same contact multiplicity. Native return events continue
to update the existing terminal coefficient.

The branch update reads the native queued pre-learning potentials. It does not
recompute old inputs with newly learned weights or add a second propagation
queue. Inputs with zero dendritic delay enter during the same tick. A separate
ordinary network-cleft delay still applies to emitted release.

`S` is the membrane-area-weighted voltage. `O` is mean local release over all
declared output sites, including sites without a connected partner if supplied.
Neither is a spike flag or a measured calcium signal. Inspect `cable.voltage`,
`cable.last_current`, `arrived_port_current`, `terminal_ids` and
`terminal_release` for the actual distributed state. The native base uses
float32 arithmetic for queued potentials; the sparse cable solve uses float64.
Conservation and native/cable numerical differences are checked separately.
Local voltage has no hidden clipping; numerical nonfiniteness raises an error.

Deepcopy preserves voltage and native in-flight state while discarding only
the non-pickleable SuperLU factor, which is rebuilt on use. The network's
optional `reset_additional_state` hook clears compartment state during an
ordinary reset without changing geometry or erasing learned coefficients.
Ordinary neurons have no such hook and retain their existing reset path.

## Evidence

`tests/test_passive_cable.py` checks the dense equation, signed-current
conservation, dynamic spread and washout, the uniform-input limit, input-count
normalization, exact delayed potentials, ongoing learning and return events,
deepcopy continuation, invalid geometry/timesteps and reset. These tests do not
establish fly physiology.

The companion `active-inference` reconstruction binds the actual FlyWire m783
APL contact sites to these ports and records all node voltages and currents
every tick. Its 224-tick intact run replayed exactly without instrumentation.
With APL release blocked, replacing global APL with this cable left every
recorded KC and PN field identical. The intact replacement changed their
inputs and dynamics. See that repository's
`simulations/drosophila/LOCAL_APL_FINDINGS_2026-09-10.md` for the findings and
important saturation limits.

A conditional refinement of the first arriving current from exact rest used
1 through 64 substeps per original tick. The maximum voltage rose from 299.08
to 349.31 model units; capped output contact entries increased from 161 to 207.
Finer time integration therefore did not eliminate the initial saturation.
This is a held-current numerical experiment, not a rerun of the adapting
network. It also changes the explicit leak approximation. The conservation
check is not evidence of converged transient timing or calibrated release.

[Amin et al., 2020](https://doi.org/10.7554/eLife.56954) constrains the biological
question through measurements of local APL activity and inhibition. Its
phenomenological spatial fit is not validation of this dynamical cable model.
The specific resistance ratio, graded gain and compartment construction here
remain assumptions requiring their own tests.
