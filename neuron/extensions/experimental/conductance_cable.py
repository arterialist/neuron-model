"""Optional local ligand conductance for the experimental PAULA cable.

This is an intracellular operator hypothesis, not an ATP receptor kinetic model
or a fitted APL neuron. Native explicit membrane leak is retained. Local ligand
current G*(E-v) and axial exchange are evaluated implicitly for stability:
    (C + alpha*(L+G)) v_next = (1-alpha)*C*v + alpha*(I+G*E).
G uses the same normalized conductance units as L. E and v share arbitrary
voltage units until externally calibrated. No calcium or release rule is added.
The inherited step() is unchanged; callers must explicitly use the new method.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .passive_cable import PassiveCable


class ConductanceCable(PassiveCable):
    def __init__(self, parents, xyz_um, radius_um, rm_over_ra_um):
        super().__init__(parents, xyz_um, radius_um, rm_over_ra_um)
        self._ligand_factor = None
        self._ligand_alpha = None
        self._ligand_g = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_ligand_factor"] = None
        return state

    def step_conductance(self, current, alpha, conductance, reversal=1.):
        """Advance local voltage. last_current includes actual ligand current.

        Input current is externally imposed I. After the solve, last_current
        is I + G*(E-v_next), the current used in discrete mass balance. The
        caller must retain commanded I and G separately when recording a run.
        """
        current = np.asarray(current, dtype=np.float64)
        g = np.asarray(conductance, dtype=np.float64)
        if (current.shape != self.voltage.shape or g.shape != self.voltage.shape
                or not np.isfinite(current).all() or not np.isfinite(g).all()
                or np.any(g < 0) or not np.isfinite(reversal)):
            raise ValueError("Invalid current, conductance or reversal")
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Explicit membrane leak requires 0 < dt/lambda <= 1")
        if not np.any(g):
            super().step(current, alpha)
            return
        if (self._ligand_factor is None or self._ligand_alpha != alpha
                or not np.array_equal(self._ligand_g, g)):
            matrix = diags(self.capacity+alpha*g, format="csc") + alpha*self.laplacian
            self._ligand_factor = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0)
            self._ligand_alpha = alpha
            self._ligand_g = g.copy()
        old_mass = float(self.capacity@self.voltage)
        rhs = (1-alpha)*self.capacity*self.voltage + alpha*(current+g*reversal)
        voltage = self._ligand_factor.solve(rhs)
        if not np.isfinite(voltage).all():
            raise FloatingPointError("Nonfinite conductance cable state")
        actual_current = current+g*(reversal-voltage)
        expected_mass = (1-alpha)*old_mass + alpha*float(actual_current.sum())
        residual = float(self.capacity@voltage)-expected_mass
        if abs(residual) > 1e-8*max(1, abs(expected_mass)):
            raise FloatingPointError(f"Conductance cable failed current conservation: {residual}")
        self.voltage[:] = voltage
        self.last_current[:] = actual_current
        self.last_mass_residual = residual
