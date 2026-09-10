# Passive electrical exchange

`ElectricalNeuron` adds passive somatic current exchange to ordinary PAULA.
`ElectricalCurrentNeuron` combines the same mechanism with the opt-in input
current kernel. All chemical ports, queued arrivals, firing/reset equations,
receiving plasticity and return-event adaptation remain native.

The caller declares each undirected contact once. The network wrapper samples
all endpoint voltages before any cell updates. Each cell then computes its own
current from that snapshot. Reversing cell evaluation order does not change the
exchange. No stimulus label, network decoder or corrective controller is used.

```python
from neuron.extensions.experimental.electrical import (
    ElectricalNeuron, ElectricalCoupling,
)

# Construct normal PAULA parameters, ports and chemical connections first.
cell = ElectricalNeuron(neuron_id, params)
cell.configure_electrical_input(leak_conductance=1.0)
# Put configured cells into the native network before constructing the wrapper.
coupling = ElectricalCoupling(network, [(id_a, id_b, 0.03)])
coupling.run_tick()
```

These numbers illustrate the API, not measured conductances. Ordinary network
construction is unchanged. An electrical cell must receive a synchronous
snapshot, so calling the unwrapped network directly fails rather than silently
omitting the junction current.

For an edge with conductance `g`, physical current in shared abstract units is
`J_i = g * (S_j - S_i)`. PAULA's membrane equation already normalizes leak to
one, so the injected model current is `I_i = J_i / g_leak_i`. Effective capacity
in these units is `lambda_i * g_leak_i`. Conservation applies to `J_i`, not
necessarily to the normalized currents of cells with different leaks.

The isolated exchange obeys `J_i + J_j = 0` and dissipates voltage difference,
with `S_i*J_i + S_j*J_j = -g*(S_i-S_j)**2`. The explicit step requires
`dt*(1 + sum(g)/g_leak_i) <= lambda_i` for every endpoint. This conservative
bound keeps the passive update a nonnegative weighted combination with leak.
It does not prove stability of the surrounding active chemical network.

At zero electrical current, the native current's numeric type is retained.
Otherwise the extension adds the current in float64. Reset clears electrical
snapshots and current state without resetting learned weights. The combined
class also clears its synaptic-current tails. Deepcopy retains both states.

## What this model cannot establish

Electrical contacts are not inferred from chemical contact counts. Somatic
`S` is a reset integrate-and-fire state, not an action-potential waveform or a
dendritic junction voltage. The extension has no junction locations, cable
attenuation, rectification, voltage-dependent conductance, ion-channel kinetics
or electrical plasticity. A common voltage scale and relative leak values are
assumptions. Chemical learning remains active, but this does not calibrate its
biological rate.

Tests cover analytic passive transfer, conservative and dissipative exchange,
unequal leaks, evaluation order, exact zero-gap chemical events and receiving
and terminal adaptation, current-kernel composition, deepcopy and reset.
Biological validation requires separate identified-cell experiments.
