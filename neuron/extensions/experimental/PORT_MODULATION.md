# Local plasticity modulation by input port

`PortModulationNeuron` extends `EligibilityTraceNeuron`. It lets native-rule
input synapses differ in their sensitivity to the existing cell-local learning
modulator. It adds no external controller and no new dynamic state.

```python
from neuron.extensions.experimental.port_modulation import PortModulationNeuron

metadata = {
    "bounded_plasticity": True,
    "eligibility_ports": [0, 1],
    "plasticity_rate_boost": 499.0,
    "native_port_modulation": [{"port": 2, "sensitivity": 0.25}],
}
```

For a declared port, its effective incoming learning rate is
`eta_basal * (1 + sensitivity * (cell_gain - 1))`. The gain comes from the
existing local neuromodulator concentration, not stimulus identity. Both basal
learning rates must remain positive. Sensitivity zero removes modulation only;
the weight still adapts at the basal rate. Empty declarations and sensitivity
one take the inherited path exactly. An eligibility port cannot also receive a
native-rule override.

Forward propagation uses the current weights as before. After the inherited
tick, the declared active incoming info weights receive the same signed bounded
plasticity flow, evaluated at their own rates. Somatic dynamics, eligibility
updates, event timing and outgoing retrograde adaptation retain their inherited
rules. Intermediate inherited weight logs remain provisional, as in the
existing bounded extension. Metadata must be saved with the configuration;
there are no additional persistent traces beyond the inherited state.

The mechanism is phenomenological. Mitsushima, Sano and Takahashi's 2013 study,
doi:10.1038/ncomms3760, reports different cholinergic receptor contributions to
excitatory and inhibitory plasticity. It does not prescribe this equation or
establish that inhibitory plasticity should always be weaker. Inhibitory
plasticity can itself support memory and must not be treated as an error.

In the 176-neuron synthetic association preparation, sensitivity 0.25 on
inhibitory inputs preserves selective partial/corrupted-cue recall at 16 and
64 presentations per cue across four seeds and both opposite assignments.
Unit sensitivity reproduces the previous acquisition exactly and eventually
silences recall. All synapses continue adapting. These results are finite-horizon
isolated evidence, not an accepted real-media or embodied component. In the
longer exposure test the rejected-output threshold margin narrows substantially.
This extension is not a homeostatic controller or a solution to lifelong
learning. See active-inference's population findings and continual-association
experiment for complete records and controls.
