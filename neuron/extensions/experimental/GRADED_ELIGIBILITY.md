# Graded release with existing bounded adaptation

`GradedEligibilityNeuron` composes `GradedNeuron` and
`EligibilityTraceNeuron` through their existing cooperative inheritance paths.
It adds no update equation. With `graded_gain <= 0`, the existing eligibility
neuron supplies the tick dynamics unchanged. Ordinary spiking neurons can
still use their selected eligibility ports.

With positive graded gain, the existing graded extension suppresses the
somatic spike/reset and releases a rectified, optionally clipped function of
membrane potential. The existing bounded incoming learning and local rate
modulation remain active. The native last-spike timestamp does not advance.
Its native timing direction therefore remains negative, including for incoming
weight adaptation and the errors returned to upstream terminals.

This is not spike-based learning in a non-spiking cell. The constructor rejects
nonempty spike-eligibility ports on a graded cell. A future graded-postsynaptic
eligibility mechanism would need its own definition and tests, rather than
silently treating continuous release as a spike on every tick.

The current audiovisual experiment uses gain 0.25 only on sensory receptors.
Their initial external amplitude is twice the transduced intensity, incoming
weight is two, and one-tick dendritic attenuation is 0.99. The initial emitted
informational amplitude is therefore approximately 0.99 times intensity. This
is a fixed dimensional scaling, not normalization fitted to a recording.
Synaptic adaptation can change the relationship later and must be measured.

The biological motivation for graded sensory release does not make this a
biophysical photoreceptor or cochlea model. It has no ion-channel, vesicle,
horizontal-cell or calcium mechanism. Those must not be inferred from its name.

Tests compare the default path's states, weights, eligibility and events with
`EligibilityTraceNeuron`, check graded release against the inherited membrane
state while adaptation remains positive, and reject undefined graded
spike-eligibility configurations. All 28 focused graded, eligibility, bounded,
rate and port-modulation tests pass on 9 September 2026.
