# Bounded signed-magnitude plasticity

Status: experimental local learning mechanism. It is not an accepted brain
architecture, a demonstrated associative-memory system or a cellular reconstruction.

`BoundedPlasticityNeuron` inherits `PlasticityRateNeuron`. It is opt-in through
`bounded_plasticity` metadata. Default behavior is unchanged. The experiment
keeps the PAULA timing window, delayed currents, retrograde signals and weak
positive basal learning. Neural modulation can still amplify the learning rate.

## Why it exists

An exact replay of the 1,152-cell regional audiovisual experiment found a
negative information weight clamping to -100 at tick 1872, then jumping to +100
at tick 1887. The native positive-update soft factor becomes 11 at weight -100.
That amplified the rebound to +164 before hard clipping. It was an arithmetic
sign reversal, not a modeled biological change in receptor or chloride state.

Preserving a synapse's sign here is a modeling assumption about that connection.
It does not impose Dale's principle on all outputs of each neuron, and does not
deny biological mechanisms that can change the sign of a synaptic effect.

## The candidate dynamics

Let `q = abs(w)`, `e` be the native local error magnitude, `d` the native timing
credit direction, and `s` accumulated adaptation time. For positive credit,
`dq/ds = q * [e * (1 - q/C) - decay]`. For negative credit,
`dq/ds = -q * (e + decay)`. Keep the original sign of `w`.

Each event uses the analytic flow over its effective learning-rate interval,
holding the locally measured error constant within that update. This avoids
Euler sign overshoot. It is not an exact integration of the coupled brain.
The cap defaults to 10, matching the native positive soft-bound scale; decay
defaults to .02. Neither value is a task label or a decoded performance target.

Zero weights remain zero. Extreme depression can underflow and is counted.
No hidden minimum weight, frozen learning period or external normalization
repairs the state. Stable bounds alone can still produce saturated, silent,
redundant or non-associative networks, so these remain failure conditions.

## Implementation boundary

The base neuron has no postsynaptic-update hook. The subclass captures local
pre-update information, runs the inherited tick, then replaces active incoming
information weights before returning. Propagating currents and retrograde
events use pre-update quantities in the base code. Tests verify that ordering.
Native DEBUG weight messages are provisional; separate messages report final
bounded updates. Network consumers see only the final state. A later shared
hook refactor should preserve this ordering and prove default equivalence.

The mechanism does not stabilize every PAULA state variable. In particular,
presynaptic terminal information and modulator amplitudes retain their existing
retrograde rules. The experiment observes terminal sign and range each tick.
It rejects passive weight decay and ablations when enabled rather than silently
combining incompatible semantics.

## Relation to literature

[van Rossum, Bi and Turrigiano, 2000](https://doi.org/10.1523/JNEUROSCI.20-23-08812.2000)
shows that weight dependence can stabilize correlation-based learning without
necessarily supplying strong competition. [Gutig et al., 2003](https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003)
examines the tradeoff between stable weight dynamics and sensitivity to input
correlations. These motivate testing magnitude dependence. Neither paper
derives the equation above, uses PAULA's timing-credit rule, or validates the
present hierarchical or consciousness claims.

## Experiments

The shared `multimodal_pairing_probe` accepts `--weight-dynamics native|bounded`.
Paired and swapped real-video training keeps sensory amplitude marginals equal.
Fresh-state probes test stored incoming weights with adaptation still active.
Every tick records cellular variables and a read-only weight-health observer.
Episode boundary weights permit independent checks; interior weight arrays are
not all retained. The separate auditor states this limitation explicitly.

Detailed results and reproducible paths live in the active-inference repository's
`simulations/active_inference/experiments/POPULATION_FINDINGS_2026-09-08.md`.
