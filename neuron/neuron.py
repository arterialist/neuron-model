#
# A Proof-of-Concept (PoC) implementation of the Consolidated Formal Neuron Model.
# This code translates the formal model into a runnable Python structure.
#

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from loguru import logger

# Global constants for numerical stability bounds
MAX_MEMBRANE_POTENTIAL = 10.0  # Reasonable biological limit for membrane potential
MIN_MEMBRANE_POTENTIAL = -10.0  # Lower bound for membrane potential
MAX_SYNAPTIC_WEIGHT = 2.0  # Maximum allowed synaptic weight
MIN_SYNAPTIC_WEIGHT = 0.01  # Minimum synaptic weight to prevent zero weights


_loggers_initialized = False


def setup_neuron_logger(level: str = "INFO") -> None:
    """Setup colored logging for neuron model with specified level."""
    global _loggers_initialized

    if not _loggers_initialized:
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            + "<level>{level: <8}</level> | "
            + "<cyan>N:{extra[neuron_int]}({extra[neuron_hex]})</cyan> | "
            + "<level>{message}</level>",
            level=level,
            colorize=True,
        )
        _loggers_initialized = True


@dataclass
class PostsynapticInputVector:
    """Represents the u_i vector for a postsynaptic point."""

    info: float = field(default_factory=lambda: np.random.uniform(0.5, 1.5))
    plast: float = field(default_factory=lambda: np.random.uniform(0.5, 1.5))
    adapt: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class PresynapticOutputVector:
    """Represents the u_o vector for a presynaptic point."""

    info: float = field(default_factory=lambda: np.random.uniform(0.5, 1.5))
    mod: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class PostsynapticPoint:
    """Represents a postsynaptic point (P'_in) on the neuron's graph."""

    u_i: PostsynapticInputVector
    potential: float = 0.0


@dataclass
class PresynapticPoint:
    """Represents a presynaptic point (P'_out) on the neuron's graph."""

    u_o: PresynapticOutputVector
    u_i_retro: float = field(default_factory=lambda: np.random.uniform(0.5, 1.5))


@dataclass
class NeuronParameters:
    """Parameters for the Consolidated Formal Neuron Model."""

    r_base: float = 1.0  # Base firing threshold
    b_base: float = 1.2  # Base post-cooldown threshold
    c: int = 10  # Cooldown period
    lambda_param: float = 20.0  # Membrane time constant (lambda is a keyword)

    num_neuromodulators: int = 2
    num_inputs: int = 10

    # Learning and plasticity
    p: float = 1.0  # Constant magnitude of every output signal
    eta_post: float = 0.01  # Postsynaptic learning rate (plasticity)
    eta_retro: float = 0.01  # Retrograde learning rate (plasticity)

    # Modulation and adaptation
    gamma: np.ndarray = field(
        default_factory=lambda: np.array([0.99, 0.995])
    )  # Decay factor for the neuromodulatory state vector (EMA)
    beta_avg: float = 0.999  # Decay factor for the average firing rate (EMA)

    # Dynamic parameter weights
    w_r: np.ndarray = field(
        default_factory=lambda: np.array([-0.2, 0.05])
    )  # Sensitivity vector for the primary threshold (excitability)
    w_b: np.ndarray = field(
        default_factory=lambda: np.array([-0.2, 0.05])
    )  # Sensitivity vector for the post-cooldown threshold (metaplasticity)
    w_tref: np.ndarray = field(
        default_factory=lambda: np.array([-20.0, 10.0])
    )  # Sensitivity vector for the learning window (metaplasticity)

    # Signal propagation
    delta_decay: float = 0.95  # Per-tick potential decay factor for signals


class Neuron:
    """
    Represents the entire Consolidated Formal Neuron Model, including its state,
    parameters, and graph structure.
    """

    def __init__(
        self, neuron_id: int, params: NeuronParameters, log_level: str = "INFO"
    ):
        # Validate neuron ID range (0 to 2^36 - 1)
        if not (0 <= neuron_id < 2**36):
            raise ValueError(f"Neuron ID {neuron_id} must be in range [0, {2**36-1}]")

        self.id = neuron_id
        setup_neuron_logger(log_level)

        # Create logger context with neuron ID (both int and hex)
        self.logger = logger.bind(neuron_int=neuron_id, neuron_hex=f"{neuron_id:09x}")

        self.logger.info(f"Initializing neuron {neuron_id} (0x{neuron_id:09x})")

        # Store parameters directly
        self.params = params

        # --- Section 1: Graph Architecture ---
        # Using integer IDs for synaptic points (0 to 2^12 - 1)
        self.postsynaptic_points: Dict[int, PostsynapticPoint] = {}
        self.presynaptic_points: Dict[int, PresynapticPoint] = {}
        self.distances: Dict[Tuple[int, str], int] = {}  # (synapse_id, "hillock")

        # --- Section 4: System State Variables and Parameters ---

        # Tick-Dependent State Variables
        self.S: float = 0.0  # Internal potential at the axon hillock H
        self.O: float = 0.0  # Output signal from H (1 on spike, 0 otherwise)
        self.t_last_fire: float = -np.inf
        self.F_avg: float = 0.0  # Long-term average firing rate (EMA)
        self.M_vector: np.ndarray = np.zeros(self.params.num_neuromodulators)
        self.r: float = self.params.r_base
        self.b: float = self.params.b_base

        # Calculate initial t_ref based on bounds
        self.upper_t_ref_bound: float = self.params.c * self.params.num_inputs
        self.lower_t_ref_bound: float = 2 * self.params.c
        self.t_ref: float = self.upper_t_ref_bound

        # Internal signal propagation queue: (arrival_tick, target_node, V_local, source_synapse_id)
        self.propagation_queue: List[Tuple[int, str, float, int]] = []

        self.logger.info(
            f"Neuron {neuron_id} (0x{neuron_id:09x}) initialized with parameters: r_base={self.params.r_base}, "
            f"b_base={self.params.b_base}, lambda={self.params.lambda_param}, "
            f"num_neuromodulators={self.params.num_neuromodulators}"
        )
        self.logger.debug(
            f"Initial state - S={self.S}, r={self.r}, b={self.b}, t_ref={self.t_ref}"
        )

    def add_synapse(self, synapse_id: int, distance_to_hillock: int):
        """Helper to build the neuron's structure."""
        # Validate synapse ID range (0 to 2^12 - 1)
        if not (0 <= synapse_id < 2**12):
            raise ValueError(f"Synapse ID {synapse_id} must be in range [0, {2**12-1}]")

        num_modulators = self.params.num_neuromodulators
        u_i_vector = PostsynapticInputVector(
            adapt=np.random.uniform(0.1, 0.5, num_modulators)
        )
        self.postsynaptic_points[synapse_id] = PostsynapticPoint(u_i=u_i_vector)
        self.distances[(synapse_id, "hillock")] = distance_to_hillock

        self.logger.info(
            f"Added synapse {synapse_id} (0x{synapse_id:03x}) at distance {distance_to_hillock} from hillock"
        )
        self.logger.debug(
            f"Synapse {synapse_id} (0x{synapse_id:03x}) - info={u_i_vector.info:.3f}, "
            f"adapt={u_i_vector.adapt}"
        )

    def add_axon_terminal(self, terminal_id: int, distance_from_hillock: int):
        """Helper to build the neuron's structure."""
        # Validate terminal ID range (0 to 2^12 - 1)
        if not (0 <= terminal_id < 2**12):
            raise ValueError(
                f"Terminal ID {terminal_id} must be in range [0, {2**12-1}]"
            )

        num_modulators = self.params.num_neuromodulators
        u_o_vector = PresynapticOutputVector(
            mod=np.random.uniform(0.1, 0.5, num_modulators)
        )
        self.presynaptic_points[terminal_id] = PresynapticPoint(u_o=u_o_vector)
        self.distances[(terminal_id, "hillock")] = distance_from_hillock

        self.logger.info(
            f"Added axon terminal {terminal_id} (0x{terminal_id:03x}) at distance {distance_from_hillock} from hillock"
        )
        self.logger.debug(
            f"Terminal {terminal_id} (0x{terminal_id:03x}) - info={u_o_vector.info:.3f}, "
            f"mod={u_o_vector.mod}"
        )


# --- 2. State Transition Implementation ---


def neuron_tick(
    neuron: Neuron,
    external_inputs: Dict[int, Dict[str, Any]],
    current_tick: int,
    dt: float,
) -> Neuron:
    """
    Executes a single time step (tick) of the neuron model.
    This function encapsulates all of Section 5: State Transitions.
    """

    # Start timing the tick execution
    tick_start_time = time.perf_counter_ns()

    neuron.logger.debug(f"--- Tick {current_tick} ---")
    neuron.logger.debug(f"Received inputs from {len(external_inputs)} synapses")

    # --- Section 5.A: Neuromodulation and Dynamic Parameter Update ---

    # 5.A.1 & 5.A.2: Aggregate Neuromodulatory Input and Update State Vector
    total_adapt_signal = np.zeros(neuron.params.num_neuromodulators)
    for synapse_id, O_ext in external_inputs.items():
        if synapse_id in neuron.postsynaptic_points and "mod" in O_ext:
            # For PoC, assume u_adapt throughput is proportional to input and receptor efficacy
            u_adapt_throughput = (
                O_ext["mod"] * neuron.postsynaptic_points[synapse_id].u_i.adapt
            )
            total_adapt_signal += u_adapt_throughput
            neuron.logger.debug(
                f"Modulation from {synapse_id} (0x{synapse_id:03x}): {O_ext['mod']} -> throughput: {u_adapt_throughput}"
            )

    # EMA update rule for the neuromodulatory state vector M(t)
    old_M_vector = neuron.M_vector.copy()
    neuron.M_vector = (neuron.params.gamma * neuron.M_vector) + (
        (1 - neuron.params.gamma) * total_adapt_signal
    )

    if not np.array_equal(old_M_vector, neuron.M_vector):
        neuron.logger.debug(
            f"Neuromodulatory state updated: {old_M_vector} -> {neuron.M_vector}"
        )

    # 5.A.3: Update Average Firing Rate (using neuron.O from previous tick)
    old_F_avg = neuron.F_avg
    neuron.F_avg = (neuron.params.beta_avg * neuron.F_avg) + (
        (1 - neuron.params.beta_avg) * neuron.O
    )

    if abs(old_F_avg - neuron.F_avg) > 1e-6:
        neuron.logger.debug(
            f"Average firing rate updated: {old_F_avg:.6f} -> {neuron.F_avg:.6f}"
        )

    # 5.A.4: Calculate Final Dynamic Parameters
    # Excitability (Model H thresholds r and b)
    old_r, old_b = neuron.r, neuron.b
    neuron.r = neuron.params.r_base + np.dot(neuron.params.w_r, neuron.M_vector)
    neuron.b = neuron.params.b_base + np.dot(neuron.params.w_b, neuron.M_vector)

    if abs(old_r - neuron.r) > 1e-6 or abs(old_b - neuron.b) > 1e-6:
        neuron.logger.debug(
            f"Thresholds updated: r={old_r:.4f}->{neuron.r:.4f}, b={old_b:.4f}->{neuron.b:.4f}"
        )

    # Metaplasticity (learning window t_ref)
    old_t_ref = neuron.t_ref
    normalized_F_avg = np.clip(neuron.F_avg * neuron.params.c, 0, 1)
    t_ref_homeostatic = (
        neuron.upper_t_ref_bound
        - (neuron.upper_t_ref_bound - neuron.lower_t_ref_bound) * normalized_F_avg
    )

    neuron.t_ref = t_ref_homeostatic + np.dot(neuron.params.w_tref, neuron.M_vector)
    neuron.t_ref = np.clip(
        neuron.t_ref, neuron.lower_t_ref_bound, neuron.upper_t_ref_bound
    )

    if abs(old_t_ref - neuron.t_ref) > 1e-6:
        neuron.logger.debug(
            f"Learning window updated: t_ref={old_t_ref:.3f}->{neuron.t_ref:.3f}"
        )

    # --- Section 5.B: Input Processing & Propagation ---
    # 5.B.1: Generate local potentials from external inputs
    neuron.logger.debug(f"Processing {len(external_inputs)} external inputs")
    for synapse_id, O_ext in external_inputs.items():
        if synapse_id in neuron.postsynaptic_points:
            synapse = neuron.postsynaptic_points[synapse_id]
            # Dot product is simplified here as O_ext is a dictionary.
            # We use the informational components for this calculation.
            V_local = O_ext.get("info", 0) * synapse.u_i.info
            synapse.potential = V_local

            # 5.B.2: Schedule signal propagation to the hillock
            arrival_tick = current_tick + neuron.distances[(synapse_id, "hillock")]
            neuron.propagation_queue.append(
                (arrival_tick, "hillock", V_local, synapse_id)
            )

            neuron.logger.debug(
                f"Synapse {synapse_id} (0x{synapse_id:03x}): input={O_ext.get('info', 0):.3f}, "
                f"V_local={V_local:.3f}, arrival_tick={arrival_tick}"
            )

    # --- Section 5.C: Signal Integration ---
    # 5.C.1: Integrate signals arriving at the axon hillock at this tick
    I_t = 0.0
    signals_to_process_now = [
        s for s in neuron.propagation_queue if s[0] == current_tick
    ]
    neuron.propagation_queue = [
        s for s in neuron.propagation_queue if s[0] > current_tick
    ]

    neuron.logger.debug(
        f"Processing {len(signals_to_process_now)} signals arriving at hillock"
    )
    for (
        arrival_tick,
        target_node,
        V_initial,
        source_synapse_id,
    ) in signals_to_process_now:
        if target_node == "hillock":
            distance = neuron.distances[(source_synapse_id, "hillock")]
            V_arriving = V_initial * (neuron.params.delta_decay**distance)
            I_t += V_arriving
            neuron.logger.debug(
                f"Signal from synapse {source_synapse_id} (0x{source_synapse_id:03x}): V_initial={V_initial:.3f}, "
                f"distance={distance}, V_arriving={V_arriving:.3f}"
            )

    if I_t > 0:
        neuron.logger.debug(f"Total integrated current: I_t={I_t:.3f}")

    # --- Section 5.D: Somatic Firing (Model H) ---
    # 5.D.1: State Evolution using the discrete update rule
    old_S = neuron.S
    dS = (dt / neuron.params.lambda_param) * (-neuron.S + I_t)
    neuron.S += dS

    # Add numerical stability safeguards
    # Prevent numerical overflow by clamping S to reasonable bounds
    if neuron.S > MAX_MEMBRANE_POTENTIAL:
        neuron.logger.warning(
            f"Membrane potential S={neuron.S:.4f} exceeded maximum {MAX_MEMBRANE_POTENTIAL}, clamping"
        )
        neuron.S = MAX_MEMBRANE_POTENTIAL
    elif neuron.S < MIN_MEMBRANE_POTENTIAL:
        neuron.logger.warning(
            f"Membrane potential S={neuron.S:.4f} below minimum {MIN_MEMBRANE_POTENTIAL}, clamping"
        )
        neuron.S = MIN_MEMBRANE_POTENTIAL

    # Check for numerical instability (NaN or infinity)
    if not np.isfinite(neuron.S):
        neuron.logger.error(
            f"Numerical instability detected: S={neuron.S}, resetting to 0"
        )
        neuron.S = 0.0

    neuron.logger.debug(
        f"Membrane potential: S={old_S:.4f} -> {neuron.S:.4f} (dS={dS:.4f})"
    )

    # Determine the active threshold based on the cooldown period
    cooldown_remaining = neuron.params.c - (current_tick - neuron.t_last_fire)
    active_threshold = (
        neuron.b if (current_tick - neuron.t_last_fire) <= neuron.params.c else neuron.r
    )

    neuron.logger.debug(
        f"Cooldown remaining: {cooldown_remaining}, active_threshold: {active_threshold:.4f}"
    )

    # 5.D.4: Threshold Reset for inactivity
    if neuron.S < 0.01:  # A near-zero resting state
        active_threshold = neuron.r
        neuron.logger.debug("Near-zero state detected, using threshold r")

    # 5.D.2: Firing Condition
    time_since_last_fire = current_tick - neuron.t_last_fire
    will_fire = neuron.S >= active_threshold and time_since_last_fire >= neuron.params.c

    if will_fire:
        # 5.D.3: Firing Dynamics
        neuron.O = 1.0
        neuron.S = 0.0
        neuron.t_last_fire = current_tick

        neuron.logger.success(
            f"🔥 SPIKE at tick {current_tick}! S={old_S:.4f} >= {active_threshold:.4f}"
        )
        neuron.logger.debug(
            f"Post-spike state: S={neuron.S}, O={neuron.O}, t_last_fire={neuron.t_last_fire}"
        )

        # 5.E.1: Presynaptic Release Scheduling is conceptual for a single neuron PoC
    else:
        neuron.O = 0.0
        if neuron.S > 0.1:  # Only log if there's significant activity
            neuron.logger.debug(
                f"No spike: S={neuron.S:.4f} < {active_threshold:.4f} or cooldown active"
            )

    # --- Section 5.E.2: Dendritic Computation & Plasticity ---
    plasticity_updates = []
    for synapse_id, O_ext in external_inputs.items():
        if synapse_id in neuron.postsynaptic_points:
            synapse = neuron.postsynaptic_points[synapse_id]

            # Simplified error calculation for PoC
            E_dir_info = O_ext.get("info", 0) - synapse.u_i.info

            # Temporal Correlation
            delta_t = current_tick - neuron.t_last_fire
            direction = 1.0 if delta_t <= neuron.t_ref else -1.0

            # Postsynaptic Update
            old_u_i_info = synapse.u_i.info
            delta_u_i = (
                neuron.params.eta_post * direction * abs(E_dir_info) * synapse.u_i.info
            )
            synapse.u_i.info += delta_u_i

            # Add bounds to prevent synaptic weights from growing unboundedly
            if synapse.u_i.info > MAX_SYNAPTIC_WEIGHT:
                neuron.logger.debug(
                    f"Synaptic weight {synapse_id} (0x{synapse_id:03x}) exceeded maximum, clamping from {synapse.u_i.info:.4f} to {MAX_SYNAPTIC_WEIGHT}"
                )
                synapse.u_i.info = MAX_SYNAPTIC_WEIGHT
            elif synapse.u_i.info < MIN_SYNAPTIC_WEIGHT:
                neuron.logger.debug(
                    f"Synaptic weight {synapse_id} (0x{synapse_id:03x}) below minimum, clamping from {synapse.u_i.info:.4f} to {MIN_SYNAPTIC_WEIGHT}"
                )
                synapse.u_i.info = MIN_SYNAPTIC_WEIGHT

            if abs(delta_u_i) > 1e-6:  # Only log significant changes
                plasticity_updates.append(
                    f"{synapse_id} (0x{synapse_id:03x}): {old_u_i_info:.4f}->{synapse.u_i.info:.4f} "
                    f"(Δ={delta_u_i:.6f}, dir={direction:.1f}, E_dir={E_dir_info:.4f})"
                )

            # Retrograde signaling is conceptual in this single-neuron PoC

    if plasticity_updates:
        neuron.logger.debug(f"Plasticity updates: {', '.join(plasticity_updates)}")

    # Calculate and log execution time
    tick_end_time = time.perf_counter_ns()
    tick_duration_ms = (tick_end_time - tick_start_time) / 1000000

    neuron.logger.debug(f"Tick {current_tick} complete: S={neuron.S:.4f}, O={neuron.O}")
    neuron.logger.info(f"Tick {current_tick} execution time: {tick_duration_ms:.3f}ms")

    return neuron
