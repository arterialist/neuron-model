"""Experimental passive cable inside one PAULA neuron, with local graded release.

This is a numerical compartment hypothesis, not measured APL electrophysiology.
Geometry is in micrometres. Each skeleton edge is a cylinder whose radius is the
mean of its endpoint radii. Half its lateral membrane area belongs to each end.
Sealed ends, uniform membrane properties and no active channels are assumed.

With C = area/sum(area), axial Laplacian L, a = dt/lambda_param:
    (C + a L) v_next = (1-a) C v + a I
This preserves the native explicit membrane leak while treating axial exchange
implicitly. Axial exchange conserves area-weighted potential. Rm/Ra sets L;
lambda_param remains in simulation ticks, not milliseconds.

Input ports retain native potentials, propagation queues, weights and return
events. A port's aggregate current is split across its explicit contact sites.
Each terminal reads the mean rectified release at its explicit output sites.
The native positive-input mask and inherited non-spiking timing-plasticity
limitations still apply. No sensory labels or behavioral policy occur here.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from ...neuron import Neuron
from ..graded import GradedNeuron, _NEVER


class PassiveCable:
    def __init__(self, parents, xyz_um, radius_um, rm_over_ra_um):
        parents = np.asarray(parents)
        self.xyz_um = np.array(xyz_um, dtype=np.float64, copy=True)
        self.radius_um = np.array(radius_um, dtype=np.float64, copy=True)
        n = len(parents)
        if (parents.dtype.kind not in "iu" or parents.shape != (n,)
                or self.xyz_um.shape != (n, 3) or self.radius_um.shape != (n,) or n < 2):
            raise ValueError("Invalid cable geometry shape or parent indices")
        self.parents = parents.astype(np.int64, copy=True)
        if (not np.isfinite(self.xyz_um).all() or not np.isfinite(self.radius_um).all()
                or np.any(self.radius_um < 0) or np.any(parents < -1) or np.any(parents >= n)):
            raise ValueError("Invalid cable coordinates, radii or parents")
        if not np.isfinite(rm_over_ra_um) or rm_over_ra_um < 0:
            raise ValueError("Rm/Ra must be finite and nonnegative")
        self.rm_over_ra_um = float(rm_over_ra_um)
        child = np.flatnonzero(parents >= 0)
        parent = parents[child]
        length = np.linalg.norm(self.xyz_um[child] - self.xyz_um[parent], axis=1)
        radius = (self.radius_um[child] + self.radius_um[parent]) / 2
        if np.any(length <= 0) or np.any(radius <= 0):
            raise ValueError("Zero-length or zero-mean-radius cable edge")
        connectivity = coo_matrix((np.ones(2 * len(child)),
                                   (np.r_[child, parent], np.r_[parent, child])), shape=(n, n)).tocsr()
        if len(child) != n - 1 or connected_components(connectivity, directed=False)[0] != 1:
            raise ValueError("Cable must be one connected acyclic tree")
        edge_area = 2 * np.pi * radius * length
        area = np.bincount(child, weights=edge_area / 2, minlength=n)
        area += np.bincount(parent, weights=edge_area / 2, minlength=n)
        self.area_um2 = area
        self.capacity = area / area.sum()
        conductance = self.rm_over_ra_um * np.pi * radius**2 / length / area.sum()
        diagonal = np.bincount(child, weights=conductance, minlength=n)
        diagonal += np.bincount(parent, weights=conductance, minlength=n)
        self.laplacian = coo_matrix((np.r_[diagonal, -conductance, -conductance],
            (np.r_[np.arange(n), child, parent], np.r_[np.arange(n), parent, child])), shape=(n, n)).tocsc()
        self.voltage = np.zeros(n, dtype=np.float64)
        self.last_current = np.zeros(n, dtype=np.float64)
        self.last_mass_residual = 0.0
        self._factor = None
        self._alpha = None

    def __getstate__(self):
        # SuperLU objects are not pickleable. Rebuild the numerical factor, not
        # the dynamic state, after an exact trusted-local deepcopy/checkpoint.
        state = self.__dict__.copy()
        state["_factor"] = None
        return state

    def step(self, current, alpha):
        current = np.asarray(current, dtype=np.float64)
        if current.shape != self.voltage.shape or not np.isfinite(current).all():
            raise ValueError("Invalid compartment current")
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Explicit membrane leak requires 0 < dt/lambda <= 1")
        if self._factor is None or self._alpha != alpha:
            matrix = diags(self.capacity, format="csc") + alpha * self.laplacian
            self._factor = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0)
            self._alpha = alpha
        old_mass = float(self.capacity @ self.voltage)
        rhs = (1 - alpha) * self.capacity * self.voltage + alpha * current
        next_voltage = self._factor.solve(rhs)
        if not np.isfinite(next_voltage).all():
            raise FloatingPointError("Nonfinite cable state; no reset or silent clipping")
        expected_mass = (1 - alpha) * old_mass + alpha * float(current.sum())
        residual = float(self.capacity @ next_voltage) - expected_mass
        if abs(residual) > 1e-8 * max(1, abs(expected_mass)):
            raise FloatingPointError(f"Cable failed current conservation: {residual}")
        self.voltage[:] = next_voltage
        self.last_current[:] = current
        self.last_mass_residual = residual


def _indices(values, limit):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= limit):
        raise ValueError("Invalid contact indices")
    return values.astype(np.int64, copy=True)


class LocalCableGradedNeuron(GradedNeuron):
    """One neuron with branch state; configure explicitly before the first tick.

    S is the membrane-area-weighted voltage, O the mean local release over all
    declared output contact sites. Neither is an action-potential flag. F_avg
    inherits this O diagnostic. Each terminal has its own actual release value.
    No local calcium or graded-specific plasticity rule is supplied here.
    """

    def configure_cable(self, cable, input_ports, input_nodes, terminal_ports,
                        terminal_nodes, all_output_nodes, *, uniform_input_port):
        if hasattr(self, "cable"):
            raise RuntimeError("Do not replace a running neuron's cable")
        if self._gg <= 0:
            raise ValueError("Local cable requires positive graded_gain")
        if self.propagation_queue or self.S != 0 or self.O != 0:
            raise ValueError("Configure the cable on a fresh neuron, not a live state")
        if set(self.postsynaptic_points) != set(range(self.params.num_inputs)):
            raise ValueError("Cable input ports must match the native input buffer")
        n = len(cable.voltage)
        inputs = _indices(input_ports, self.params.num_inputs)
        nodes = _indices(input_nodes, n)
        if len(inputs) != len(nodes):
            raise ValueError("Input contact mapping length mismatch")
        if type(uniform_input_port) is not int or uniform_input_port not in self.postsynaptic_points:
            raise ValueError("Declare the extra uniform experimental input port")
        counts = np.bincount(inputs, minlength=self.params.num_inputs)
        if counts[uniform_input_port] or np.any(np.delete(counts, uniform_input_port) == 0):
            raise ValueError("Every anatomical input needs contacts; uniform drive must be separate")
        self.cable = cable
        self.uniform_input_port = uniform_input_port
        self.input_projection = coo_matrix((1 / counts[inputs], (nodes, inputs)),
            shape=(n, self.params.num_inputs)).tocsr()
        self.terminal_ids = np.array(sorted(self.presynaptic_points), dtype=np.int64)
        terminal_to_row = {int(t): i for i, t in enumerate(self.terminal_ids)}
        ports = np.asarray(terminal_ports)
        if ports.ndim != 1 or ports.dtype.kind not in "iu" or any(int(t) not in terminal_to_row for t in ports):
            raise ValueError("Unknown terminal contact")
        terminal_rows = np.array([terminal_to_row[int(t)] for t in ports], dtype=np.int64)
        output_nodes = _indices(terminal_nodes, n)
        if len(ports) != len(output_nodes):
            raise ValueError("Terminal contact mapping length mismatch")
        counts = np.bincount(terminal_rows, minlength=len(self.terminal_ids))
        if np.any(counts == 0):
            raise ValueError("Every terminal needs explicit output contacts")
        self.terminal_projection = coo_matrix((1 / counts[terminal_rows], (terminal_rows, output_nodes)),
            shape=(len(self.terminal_ids), n)).tocsr()
        self.all_output_nodes = _indices(all_output_nodes, n)
        if not len(self.all_output_nodes):
            raise ValueError("No sites for the whole-neuron release diagnostic")
        self.arrived_port_current = np.zeros(self.params.num_inputs)
        self.terminal_release = np.zeros(len(self.terminal_ids))
        self.local_release = np.zeros(n)
        self.native_mean_error = 0.0

    def reset_additional_state(self):
        """Network reset hook; geometry and learned coefficients remain intact."""
        if not hasattr(self, "cable"):
            return
        for values in (self.cable.voltage, self.cable.last_current, self.arrived_port_current,
                       self.terminal_release, self.local_release):
            values.fill(0)
        self.cable.last_mass_residual = self.native_mean_error = 0.0

    def tick(self, external_inputs, current_tick, dt=1.0):
        if not hasattr(self, "cable"):
            raise RuntimeError("Local cable was not configured")
        alpha = dt / self.params.lambda_param
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Explicit membrane leak requires 0 < dt/lambda <= 1")
        # Read the same pre-update events the base tick will consume, preserving
        # the native queue and its exact delayed potential rather than recomputing
        # old inputs with newly adapted weights. Include newly arriving zero-delay
        # inputs, which the base schedules and consumes within this tick.
        currents = self.arrived_port_current
        currents.fill(0)
        arrivals = 0
        absolute_current = 0.0
        for arrival, target, potential, sid in self.propagation_queue:
            if target != "hillock":
                raise ValueError("Unrecognized native propagation target")
            if arrival <= current_tick:
                value = potential * self.params.delta_decay**self.distances[sid]
                currents[sid] += value
                absolute_current += abs(float(value))
                arrivals += 1
        for sid in np.flatnonzero(self.input_buffer[:, 0] > 0):
            if self.distances[sid] == 0:
                synapse = self.postsynaptic_points[sid]
                value = self.input_buffer[sid, 0] * (synapse.u_i.info + synapse.u_i.plast)
                currents[sid] += value
                absolute_current += abs(float(value))
                arrivals += 1
        old_mean = float(self.S)
        self.r = self.b = _NEVER
        frozen = "thresholds_frozen" in self._ablation
        self._ablation.add("thresholds_frozen")
        try:
            # Direct base call retains neuromodulation and all native learning,
            # without appending the global graded class's terminal events.
            events = Neuron.tick(self, external_inputs, current_tick, dt)
        finally:
            if not frozen:
                self._ablation.remove("thresholds_frozen")
        if any(isinstance(event, tuple) for event in events):
            raise RuntimeError("Unexpected somatic spike in a non-spiking cable")
        current = self.input_projection @ currents
        current += self.cable.capacity * currents[self.uniform_input_port]
        self.cable.step(current, alpha)
        mean_voltage = float(self.cable.capacity @ self.cable.voltage)
        # Base heap accumulation can stay float32; spatial scatter and the
        # conservative solve are float64. Bound the former's worst-case sum
        # rounding by gamma_n, rather than calling these bit-identical schemes.
        eps = np.finfo(np.float32).eps
        count = arrivals + 8
        gamma = count * eps / (1 - count * eps)
        roundoff = gamma * (absolute_current + abs(old_mean)) + 1e-8 * max(1, abs(mean_voltage))
        self.native_mean_error = mean_voltage - float(self.S)
        if abs(self.native_mean_error) > roundoff:
            raise FloatingPointError("Native soma and conservative cable disagree, or native soma hit its bound")
        self.S = mean_voltage
        np.maximum(self._gg * (self.cable.voltage - self._gs0), 0, out=self.local_release)
        if self._gmax > 0:
            np.minimum(self.local_release, self._gmax, out=self.local_release)
        self.O = float(self.local_release[self.all_output_nodes].mean())
        self.terminal_release[:] = self.terminal_projection @ self.local_release
        for tid, release in zip(self.terminal_ids, self.terminal_release, strict=True):
            if release > 0:
                terminal = self.presynaptic_points[int(tid)]
                events.append((self.id, int(tid), float(terminal.u_o.info) * float(release)))
        return events
