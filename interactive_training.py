import os
import random
import sys
import threading
import time
import webbrowser
import logging
import numpy as np
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Ensure the script can find the neuron and cli modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.nn_core import NNCore
from neuron.network import NeuronNetwork
from neuron.neuron import Neuron, NeuronParameters
from neuron.network_config import NetworkConfig
from cli.web_viz.server import NeuralNetworkWebServer
from torchvision import datasets, transforms


selected_dataset = None
nn_core_instance = NNCore()
input_neuron_ids = []
CURRENT_IMAGE_VECTOR_SIZE = 0
CURRENT_NUM_CLASSES = 10


def select_and_load_dataset():
    """Loads and prepares a selected dataset using torchvision.

    Supported: MNIST, CIFAR10, CIFAR100. Normalizes to [-1, 1].
    Sets globals: selected_dataset, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES.
    """
    global selected_dataset, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES

    logger.info("Select dataset:")
    logger.info("  1) MNIST")
    logger.info("  2) CIFAR10")
    logger.info("  3) CIFAR100")
    logger.info("  4) USPS")
    logger.info("  5) SVHN")
    logger.info("  6) FashionMNIST")
    choice = input("Enter choice [1]: ").strip()
    if choice == "":
        choice = "1"

    root_candidates = [
        "./data",
        "./data/mnist",
        "./data/cifar",
        "./data/usps",
        "./data/svhn",
        "./data/fashionmnist",
    ]

    if choice == "1":
        # MNIST (1x28x28), normalize to [-1, 1]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        last_err = None
        for root in root_candidates:
            try:
                selected_dataset = datasets.MNIST(
                    root=root, train=False, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if selected_dataset is None:
            raise RuntimeError(f"Failed to load MNIST: {last_err}")
        # Determine vector size from a sample
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        logger.info(
            f"Loaded MNIST with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )

    elif choice == "2":
        # CIFAR10 (3x32x32)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        last_err = None
        for root in root_candidates:
            try:
                selected_dataset = datasets.CIFAR10(
                    root=root, train=False, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if selected_dataset is None:
            raise RuntimeError(f"Failed to load CIFAR10: {last_err}")
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        logger.info(
            f"Loaded CIFAR10 with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )

    elif choice == "3":
        # CIFAR100 (3x32x32)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        last_err = None
        for root in root_candidates:
            try:
                selected_dataset = datasets.CIFAR100(
                    root=root, train=False, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if selected_dataset is None:
            raise RuntimeError(f"Failed to load CIFAR100: {last_err}")
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 100
        logger.info(
            f"Loaded CIFAR100 with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )

    elif choice == "4":
        # USPS (1x28x28)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        last_err = None
        for root in root_candidates:
            try:
                selected_dataset = datasets.USPS(
                    root=root, train=False, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if selected_dataset is None:
            raise RuntimeError(f"Failed to load USPS: {last_err}")
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        logger.info(
            f"Loaded USPS with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )

    elif choice == "5":
        # SVHN (3x32x32) — use test split
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        last_err = None
        for root in root_candidates:
            try:
                selected_dataset = datasets.SVHN(
                    root=root, split="test", download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if selected_dataset is None:
            raise RuntimeError(f"Failed to load SVHN: {last_err}")
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        logger.info(
            f"Loaded SVHN with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )
    elif choice == "6":
        # FashionMNIST (1x28x28)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        last_err = None
        for root in root_candidates:
            try:
                selected_dataset = datasets.FashionMNIST(
                    root=root, train=False, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if selected_dataset is None:
            raise RuntimeError(f"Failed to load FashionMNIST: {last_err}")
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        logger.info(
            f"Loaded FashionMNIST with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )
    else:
        raise ValueError("Invalid dataset choice.")


def build_network_interactively():
    """Interactively builds a new neural network with a flexible input layer."""
    global input_neuron_ids
    logger.info("\n--- Interactive Network Builder ---")

    try:
        # User specifies the number of input neurons
        input_size = int(
            input(f"Enter size of input layer (e.g., 20, 100) [100]: ") or 100
        )
        if input_size <= 0:
            logger.error("Input layer size must be positive.")
            return None

        num_hidden_layers = int(input("Enter number of hidden layers [1]: ") or 1)
        hidden_sizes = []
        for i in range(num_hidden_layers):
            size = int(input(f"Enter size of hidden layer {i+1} [128]: ") or 128)
            hidden_sizes.append(size)
        output_size = int(input("Enter size of output layer [10]: ") or 10)
        connectivity = float(
            input("Enter connection density (0.0 to 1.0) [0.5]: ") or 0.5
        )

    except ValueError:
        logger.error(
            "Invalid input. Please enter integers for sizes and a float for density."
        )
        return None

    network_sim = NeuronNetwork(num_neurons=0, synapses_per_neuron=0)
    net_topology = network_sim.network
    all_layers = []

    logger.info("Creating neurons...")

    # --- Create Input Layer ---
    # Each input neuron needs enough synapses to cover its share of the vector
    synapses_per_input_neuron = math.ceil(
        CURRENT_IMAGE_VECTOR_SIZE / max(1, input_size)
    )
    logger.info(
        f"Each of the {input_size} input neurons will have {synapses_per_input_neuron} synapses."
    )

    input_layer_neurons = []
    for i in range(input_size):
        neuron_id = random.randint(0, 2**36 - 1)
        params = NeuronParameters(
            num_inputs=synapses_per_input_neuron,
            num_neuromodulators=2,
            r_base=np.random.uniform(1, 2),
            b_base=np.random.uniform(3, 4),
            c=20,
            lambda_param=20.0,
            p=1.0,
            delta_decay=0.99,
            beta_avg=0.999,
            eta_post=0.01,
            eta_retro=0.01,
            gamma=np.array([0.99, 0.995]),
            w_r=np.array([-0.2, 0.05]),
            w_b=np.array([-0.2, 0.05]),
            w_tref=np.array([-20.0, 10.0]),
        )
        # Create neuron with layer metadata
        neuron = Neuron(
            neuron_id,
            params,
            log_level="WARNING",
            metadata={"layer": 0, "layer_name": "input"},
        )
        for s_id in range(synapses_per_input_neuron):
            neuron.add_synapse(s_id, distance_to_hillock=random.randint(2, 8))
        for t_id in range(10):
            neuron.add_axon_terminal(t_id, distance_from_hillock=random.randint(2, 8))
        net_topology.neurons[neuron_id] = neuron
        input_layer_neurons.append(neuron_id)
    all_layers.append(input_layer_neurons)
    input_neuron_ids = input_layer_neurons
    logger.info(f"Input layer with {input_size} neurons created.")

    # --- Create Hidden and Output Layers ---
    layer_sizes = hidden_sizes + [output_size]
    for layer_index, layer_size in enumerate(layer_sizes):
        layer_neurons = []
        layer_name = "hidden" if layer_index < len(hidden_sizes) else "output"
        for i in range(layer_size):
            num_synapses = 10
            num_terminals = 10
            neuron_id = random.randint(0, 2**36 - 1)
            params = NeuronParameters(
                num_inputs=num_synapses,
                num_neuromodulators=2,
                r_base=np.random.uniform(1.5, 2.0),
                b_base=np.random.uniform(2.0, 2.5),
                c=10,
                lambda_param=10.0,
                p=1.0,
                delta_decay=0.99,
                beta_avg=0.999,
                eta_post=0.01,
                eta_retro=0.01,
                gamma=np.array([0.99, 0.995]),
                w_r=np.array([-0.2, 0.05]),
                w_b=np.array([-0.2, 0.05]),
                w_tref=np.array([-20.0, 10.0]),
            )
            # Create neuron with layer metadata
            neuron = Neuron(
                neuron_id,
                params,
                log_level="WARNING",
                metadata={"layer": layer_index + 1, "layer_name": layer_name},
            )
            for s_id in range(num_synapses):
                neuron.add_synapse(s_id, distance_to_hillock=random.randint(2, 8))
            for t_id in range(num_terminals):
                neuron.add_axon_terminal(
                    t_id, distance_from_hillock=random.randint(2, 8)
                )
            net_topology.neurons[neuron_id] = neuron
            layer_neurons.append(neuron_id)
        all_layers.append(layer_neurons)
        logger.info(f"Layer {layer_index + 1} with {layer_size} neurons created.")

    # --- Connect the Layers ---
    logger.info("Connecting layers...")
    for i in range(len(all_layers) - 1):
        source_layer, target_layer = all_layers[i], all_layers[i + 1]
        for source_neuron_id in source_layer:
            source_neuron = net_topology.neurons[source_neuron_id]
            num_terminals = len(source_neuron.presynaptic_points)
            for target_neuron_id in target_layer:
                if random.random() < connectivity:
                    target_neuron = net_topology.neurons[target_neuron_id]
                    num_synapses = len(target_neuron.postsynaptic_points)
                    source_terminal_id = random.randint(0, num_terminals - 1)
                    target_synapse_id = random.randint(0, num_synapses - 1)
                    connection = (
                        source_neuron_id,
                        source_terminal_id,
                        target_neuron_id,
                        target_synapse_id,
                    )
                    if connection not in net_topology.connections:
                        net_topology.connections.append(connection)

    for neuron_id in input_neuron_ids:
        neuron = net_topology.neurons[neuron_id]
        for s_id in neuron.postsynaptic_points:
            net_topology.external_inputs[(neuron_id, s_id)] = {
                "info": 0.0,
                "mod": np.array([0.0, 0.0]),
            }

    logger.info(f"{len(net_topology.connections)} connections created.")
    return network_sim


def build_network_interactively_v2():
    """Advanced network builder (V2) with per-layer connectivity and optional shortcuts.

    Keeps data structures compatible with existing import/export (neurons, connections, external_inputs).
    """
    global input_neuron_ids
    logger.info("\n--- Advanced Network Builder (V2) ---")

    try:
        # Input layer
        input_size = int(input("Enter size of input layer [100]: ") or 100)
        if input_size <= 0:
            logger.error("Input layer size must be positive.")
            return None

        # Hidden layers
        num_hidden_layers = int(input("Enter number of hidden layers [1]: ") or 1)
        hidden_sizes = []
        for i in range(num_hidden_layers):
            size = int(input(f"Enter size of hidden layer {i+1} [128]: ") or 128)
            hidden_sizes.append(size)

        # Output layer
        output_size = int(input("Enter size of output layer [10]: ") or 10)

        # Connectivity per inter-layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        interlayer_connectivity = []
        logger.info("Enter connectivity (0.0-1.0) for each consecutive layer pair:")
        for li in range(len(layer_sizes) - 1):
            c = float(input(f"Connectivity for layer {li} -> {li+1} [0.5]: ") or 0.5)
            interlayer_connectivity.append(max(0.0, min(1.0, c)))

        # Optional shortcuts (multiple)
        shortcut_specs: list[tuple[int, int, float]] = []
        if (input("Enable shortcut connections? (y/n) [n]: ") or "n").lower() == "y":
            add_more = True
            while add_more:
                src_layer_idx = int(
                    input("Shortcut source layer index (0-based) [0]: ") or 0
                )
                dst_layer_idx = int(
                    input(
                        "Shortcut destination layer index (0-based, >= source+1) [1]: "
                    )
                    or 1
                )
                percent = float(
                    input(
                        "Percent of connections to convert to shortcuts (0.0-1.0) [0.2]: "
                    )
                    or 0.2
                )
                shortcut_specs.append(
                    (src_layer_idx, dst_layer_idx, max(0.0, min(1.0, percent)))
                )
                add_more = (
                    input("Add another shortcut? (y/n) [n]: ") or "n"
                ).lower() == "y"

    except ValueError:
        logger.error(
            "Invalid input. Please enter integers for sizes and a float for density."
        )
        return None

    # Build network
    network_sim = NeuronNetwork(num_neurons=0, synapses_per_neuron=0)
    net_topology = network_sim.network
    all_layers = []

    logger.info("Creating neurons (V2)...")

    # Input layer synapses per neuron based on dataset vector size
    synapses_per_input_neuron = math.ceil(
        CURRENT_IMAGE_VECTOR_SIZE / max(1, input_size)
    )
    logger.info(
        f"Each of the {input_size} input neurons will have {synapses_per_input_neuron} synapses."
    )

    # Helper to create neuron with reasonable params and metadata
    def create_neuron(
        layer_idx: int, layer_name: str, num_synapses: int, num_terminals: int = 10
    ) -> int:
        neuron_id = random.randint(0, 2**36 - 1)
        params = NeuronParameters(
            num_inputs=num_synapses,
            num_neuromodulators=2,
            r_base=np.random.uniform(0.9, 1.3),
            b_base=np.random.uniform(1.1, 1.5),
            c=10,
            lambda_param=20.0,
            p=1.0,
            delta_decay=0.96,
            beta_avg=0.999,
            eta_post=0.005,
            eta_retro=0.002,
            gamma=np.array([0.99, 0.995]),
            w_r=np.array([-0.2, 0.05]),
            w_b=np.array([-0.2, 0.05]),
            w_tref=np.array([-20.0, 10.0]),
        )
        neuron = Neuron(
            neuron_id,
            params,
            log_level="CRITICAL",
            metadata={"layer": layer_idx, "layer_name": layer_name},
        )
        for s_id in range(num_synapses):
            neuron.add_synapse(s_id, distance_to_hillock=random.randint(2, 8))
        for t_id in range(num_terminals):
            neuron.add_axon_terminal(t_id, distance_from_hillock=random.randint(2, 8))
        net_topology.neurons[neuron_id] = neuron
        return neuron_id

    # Input layer
    input_layer_neurons: list[int] = []
    for _ in range(input_size):
        nid = create_neuron(0, "input", synapses_per_input_neuron)
        input_layer_neurons.append(nid)
    all_layers.append(input_layer_neurons)
    input_neuron_ids = input_layer_neurons
    logger.info(f"Input layer with {input_size} neurons created.")

    # Hidden and Output layers
    layer_sizes = hidden_sizes + [output_size]
    for layer_index, layer_size in enumerate(layer_sizes):
        layer_neurons = []
        layer_name = "hidden" if layer_index < len(hidden_sizes) else "output"
        for _ in range(layer_size):
            nid = create_neuron(
                layer_index + 1, layer_name, num_synapses=10, num_terminals=10
            )
            layer_neurons.append(nid)
        all_layers.append(layer_neurons)
        logger.info(f"Layer {layer_index + 1} with {layer_size} neurons created.")

    # Connect consecutive layers with given probabilities
    logger.info("Connecting layers (V2)...")
    for i in range(len(all_layers) - 1):
        src_layer, dst_layer = all_layers[i], all_layers[i + 1]
        p = float(interlayer_connectivity[i])
        for src_neuron_id in src_layer:
            src_neuron = net_topology.neurons[src_neuron_id]
            src_terms = list(src_neuron.presynaptic_points.keys())
            for dst_neuron_id in dst_layer:
                if random.random() <= p:
                    dst_neuron = net_topology.neurons[dst_neuron_id]
                    dst_syns = list(dst_neuron.postsynaptic_points.keys())
                    if not src_terms or not dst_syns:
                        continue
                    connection = (
                        src_neuron_id,
                        random.choice(src_terms),
                        dst_neuron_id,
                        random.choice(dst_syns),
                    )
                    if connection not in net_topology.connections:
                        net_topology.connections.append(connection)

    # Set external inputs on input synapses (compatible with V1)
    for neuron_id in input_neuron_ids:
        neuron = net_topology.neurons[neuron_id]
        for s_id in neuron.postsynaptic_points:
            net_topology.external_inputs[(neuron_id, s_id)] = {
                "info": 0.0,
                "mod": np.array([0.0, 0.0]),
            }

    # Optional shortcuts: remove a percent of edges and reconnect across non-consecutive layers
    if shortcut_specs:
        for src_layer_idx, dst_layer_idx, percent in shortcut_specs:
            if (
                0 <= src_layer_idx < len(all_layers)
                and 0 <= dst_layer_idx < len(all_layers)
                and src_layer_idx + 1 <= dst_layer_idx
            ):
                immediate_dst = src_layer_idx + 1
                out_conns = [
                    c
                    for c in net_topology.connections
                    if c[0] in all_layers[src_layer_idx]
                    and c[2] in all_layers[immediate_dst]
                ]
                in_conns = (
                    [
                        c
                        for c in net_topology.connections
                        if c[0] in all_layers[dst_layer_idx - 1]
                        and c[2] in all_layers[dst_layer_idx]
                    ]
                    if dst_layer_idx > 0
                    else []
                )
                num_remove_out = int(len(out_conns) * percent)
                num_remove_in = int(len(in_conns) * percent)
                num_shortcuts = min(num_remove_out, num_remove_in)
                if num_shortcuts > 0:
                    out_remove = (
                        set(random.sample(out_conns, num_shortcuts))
                        if len(out_conns) >= num_shortcuts
                        else set(out_conns)
                    )
                    in_remove = (
                        set(random.sample(in_conns, num_shortcuts))
                        if len(in_conns) >= num_shortcuts
                        else set(in_conns)
                    )
                    net_topology.connections = [
                        c
                        for c in net_topology.connections
                        if c not in out_remove and c not in in_remove
                    ]
                    added = 0
                    attempts = 0
                    while added < num_shortcuts and attempts < num_shortcuts * 5:
                        attempts += 1
                        src = random.choice(all_layers[src_layer_idx])
                        dst = random.choice(all_layers[dst_layer_idx])
                        src_terms = list(
                            net_topology.neurons[src].presynaptic_points.keys()
                        )
                        dst_syns = list(
                            net_topology.neurons[dst].postsynaptic_points.keys()
                        )
                        if not src_terms or not dst_syns:
                            continue
                        conn = (
                            src,
                            random.choice(src_terms),
                            dst,
                            random.choice(dst_syns),
                        )
                        if conn not in net_topology.connections:
                            net_topology.connections.append(conn)
                            added += 1

    return network_sim


def prompt_network_export(prompt_message="Would you like to save the network?"):
    """Prompts user to export the current network and saves it if requested."""
    logger.info("\n--- Network Export ---")
    export_choice = input(f"{prompt_message} (y/n) [n]: ").lower() or "n"

    if export_choice == "y":
        try:
            default_filename = f"neuron_network_{int(time.time())}.json"
            filename = (
                input(f"Enter filename to save network [default: {default_filename}]: ")
                or default_filename
            )

            # Ensure .json extension
            if not filename.endswith(".json"):
                filename += ".json"

            # Save the network configuration
            NetworkConfig.save_network_config(nn_core_instance.neural_net, filename)  # type: ignore
            logger.info(f"Network successfully saved to '{filename}'")

        except Exception as e:
            logger.error(f"Failed to save network: {e}")
            logger.info("Network will not be saved.")


def present_images_to_network(
    label_to_find,
    num_images,
    ticks_per_image,
    delay_ticks,
    tick_sleep_ms,
    randomize_images,
):
    """Presents images to the network, sending signals for a specified number of consecutive ticks."""
    indices = [
        i for i, (img, label) in enumerate(selected_dataset) if label == label_to_find  # type: ignore
    ]

    if not indices:
        logger.warning(f"No images found with label {label_to_find}")
        return

    tick_sleep_sec = tick_sleep_ms / 1000.0
    num_input_neurons = len(input_neuron_ids)

    for i in range(num_images):
        if randomize_images:
            image_index = random.choice(indices)
        else:
            image_index = indices[i]
        image_tensor, actual_label = selected_dataset[image_index]  # type: ignore
        image_vector = image_tensor.view(-1).numpy()

        logger.info(
            f"\nPresenting image #{i+1}/{num_images} (index: {image_index}, Label: {actual_label}) for {ticks_per_image} ticks..."
        )

        # Prepare the signals list once per image
        signals = []
        for pixel_index, pixel_value in enumerate(image_vector):
            target_neuron_index = pixel_index % num_input_neurons
            target_synapse_index = pixel_index // num_input_neurons

            neuron_id = input_neuron_ids[target_neuron_index]
            # Normalize from [-1,1] to [0,1] and scale
            strength = (float(pixel_value) + 1.0) * 0.5 * 1.5
            signals.append((neuron_id, target_synapse_index, strength))

        # Loop for the specified number of ticks, sending the signals on each tick
        for tick_num in range(ticks_per_image):
            nn_core_instance.send_batch_signals(signals)
            nn_core_instance.do_tick()
            if tick_sleep_sec > 0:
                time.sleep(tick_sleep_sec)

        logger.info(f"Finished presenting image #{image_index}.")

        # Delay between presentations (running ticks without sending image signals)
        if i < num_images - 1 and delay_ticks > 0:
            logger.info(f"Waiting for {delay_ticks} ticks before next presentation...")
            for _ in range(delay_ticks):
                nn_core_instance.do_tick()
                if tick_sleep_sec > 0:
                    time.sleep(tick_sleep_sec)


def pre_run_compatibility_check(network_sim: NeuronNetwork) -> NeuronNetwork:
    """Ensure dataset vector size fits network input capacity. Optionally prompt to change dataset or network.

    Returns the (possibly replaced) network_sim. Updates globals when dataset or network changes.
    """
    global selected_dataset, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES, input_neuron_ids, nn_core_instance

    def compute_input_capacity(sim: NeuronNetwork) -> tuple[int, int, int]:
        # Determine input layer IDs if missing
        if not input_neuron_ids:
            # Try metadata inference
            layer_map = {}
            try:
                for nid, neuron in sim.network.neurons.items():
                    meta = getattr(neuron, "metadata", {}) or {}
                    if "layer" in meta:
                        try:
                            li = int(meta.get("layer", 0))
                        except Exception:
                            li = 0
                        layer_map.setdefault(li, []).append(nid)
            except Exception:
                layer_map = {}
            if layer_map:
                input_ids = layer_map[sorted(layer_map.keys())[0]]
            else:
                # Fallback: take first K neurons
                all_ids = list(sim.network.neurons.keys())
                input_ids = all_ids[: max(1, min(100, len(all_ids)))]
        else:
            input_ids = input_neuron_ids

        # Compute per-neuron synapses; use minimum to be conservative
        syn_counts = [
            len(sim.network.neurons[nid].postsynaptic_points) for nid in input_ids
        ]
        syn_per = min(syn_counts) if syn_counts else 0
        capacity = len(input_ids) * syn_per
        return capacity, len(input_ids), syn_per

    # Ask user if they want to run the check
    run_check = (
        input("Run dataset/network input compatibility check? (y/n) [y]: ")
        .strip()
        .lower()
    )
    if run_check == "n":
        return network_sim

    # Loop until compatible or user decides to proceed
    while True:
        capacity, num_inputs, syn_per = compute_input_capacity(network_sim)
        vec = CURRENT_IMAGE_VECTOR_SIZE
        if capacity >= vec and syn_per > 0 and num_inputs > 0:
            logger.info(
                f"Compatibility OK: capacity={capacity} (inputs={num_inputs} × synapses={syn_per}) >= vector={vec}"
            )
            return network_sim

        logger.warning(
            f"Incompatible: dataset vector={vec} > network capacity={capacity} (inputs={num_inputs} × synapses={syn_per})."
        )
        choice = (
            input(
                "Choose action: [d] change dataset, [n] change network, [i] ignore and proceed [n]: "
            )
            .strip()
            .lower()
            or "n"
        )
        if choice == "i":
            logger.warning(
                "Proceeding with mismatch; many pixels may be dropped or aliased."
            )
            return network_sim
        if choice == "d":
            # Reselect dataset
            try:
                select_and_load_dataset()
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                continue
            # loop to recheck
            continue
        if choice == "n":
            # Change network: load or rebuild
            sub = (
                input("[l] load network file, [b] rebuild network [b]: ")
                .strip()
                .lower()
                or "b"
            )
            if sub == "l":
                path = input("Enter network file path: ").strip()
                try:
                    network_sim = NetworkConfig.load_network_config(path)
                    nn_core_instance.neural_net = network_sim
                    # Try to infer input layer again
                    layer_map = {}
                    try:
                        for nid, neuron in network_sim.network.neurons.items():
                            meta = getattr(neuron, "metadata", {}) or {}
                            if "layer" in meta:
                                try:
                                    li = int(meta.get("layer", 0))
                                except Exception:
                                    li = 0
                                layer_map.setdefault(li, []).append(nid)
                    except Exception:
                        layer_map = {}
                    if layer_map:
                        input_neuron_ids = layer_map[sorted(layer_map.keys())[0]]
                except Exception as e:
                    logger.error(f"Failed to load network: {e}")
                    continue
            else:
                # Rebuild network interactively (V2 by default)
                rebuilt = build_network_interactively_v2()
                if not rebuilt:
                    logger.error("Rebuild failed; try again.")
                    continue
                network_sim = rebuilt
                nn_core_instance.neural_net = network_sim
            # loop to recheck
            continue
        # Unrecognized; default to re-prompt


def main():
    """Main function to run the interactive training application with web visualization."""
    global nn_core_instance, input_neuron_ids

    select_and_load_dataset()

    # --- Network Setup ---
    choice = input("Load an existing network? (y/n) [n]: ").lower() or "n"
    if choice == "y":
        filepath = input("Enter network file path: ")
        try:
            network_sim = NetworkConfig.load_network_config(filepath)
            # Try to infer input layer from metadata
            layer_map = {}
            try:
                for nid, neuron in network_sim.network.neurons.items():
                    meta = getattr(neuron, "metadata", {}) or {}
                    if "layer" in meta:
                        try:
                            li = int(meta.get("layer", 0))
                        except Exception:
                            li = 0
                        layer_map.setdefault(li, []).append(nid)
            except Exception:
                layer_map = {}

            if layer_map:
                first_layer_idx = sorted(layer_map.keys())[0]
                input_neuron_ids = layer_map[first_layer_idx]
                logger.info(
                    f"Inferred input layer from metadata: layer {first_layer_idx} with {len(input_neuron_ids)} neurons."
                )
            else:
                num_inputs_known = int(
                    input("How many input neurons does this network have? ") or "100"
                )
                all_neuron_ids = list(network_sim.network.neurons.keys())
                input_neuron_ids = all_neuron_ids[:num_inputs_known]
                logger.info(
                    f"Assuming the first {num_inputs_known} neurons are the input layer."
                )
        except FileNotFoundError:
            logger.error(f"Error: File not found at '{filepath}'. Exiting.")
            return
        except Exception as e:
            logger.error(f"An error occurred while loading the network: {e}")
            return
    else:
        # Choose builder version
        builder_version = input("Use advanced builder V2? (y/n) [y]: ").lower() or "y"
        if builder_version == "y":
            network_sim = build_network_interactively_v2()
        else:
            network_sim = build_network_interactively()
        if not network_sim:
            logger.error("Network building failed. Exiting.")
            return

    nn_core_instance.neural_net = network_sim

    # --- Pre-run compatibility check ---
    network_sim = pre_run_compatibility_check(network_sim)

    # --- Network Export Prompt (After Creation) ---
    prompt_network_export("Would you like to save the newly created network?")

    # --- Start Web Server ---
    logger.info("\nStarting web visualization server...")
    web_server = NeuralNetworkWebServer(nn_core_instance, host="127.0.0.1", port=5555)
    server_thread = threading.Thread(target=web_server.run, daemon=True)
    server_thread.start()

    time.sleep(2)
    url = "http://127.0.0.1:5555"
    logger.info(f"Web server is running at {url}")
    webbrowser.open(url)

    # --- Interactive Loop ---
    logger.info("\n--- Dataset Interaction Terminal ---")
    logger.info("Control the simulation from here. Watch the results in your browser.")
    while True:
        try:
            tick_time_str = (
                input("\nEnter tick time in milliseconds (e.g., 50) [10]: ") or "10"
            )
            tick_sleep_ms = int(tick_time_str)

            label_str = input(
                f"Enter a label (0-{CURRENT_NUM_CLASSES-1}) to present, or 'q' to quit: "
            )
            if label_str.lower() == "q":
                break

            label = int(label_str)
            if not (0 <= label <= CURRENT_NUM_CLASSES - 1):
                logger.warning(
                    f"Invalid label. Please enter a value from 0 to {CURRENT_NUM_CLASSES-1}."
                )
                continue

            num_images_str = input("Enter the number of images to present [1]: ") or "1"
            num_images = int(num_images_str)

            ticks_str = input("Enter ticks to present each image for [10]: ") or "10"
            ticks_per_image = int(ticks_str)

            delay_str = input("Enter delay ticks between images [5]: ") or "5"
            delay_ticks = int(delay_str)

            randomize_images = input("Randomize images? (y/n) [n]: ").lower() or "n"
            if randomize_images == "y":
                randomize_images = True
            else:
                randomize_images = False

            present_images_to_network(
                label,
                num_images,
                ticks_per_image,
                delay_ticks,
                tick_sleep_ms,
                randomize_images,
            )
            logger.info(f"Presented {num_images} images for label {label}")

        except ValueError:
            logger.error("Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            break

    # --- Final Network Export Prompt ---
    prompt_network_export("Would you like to save the current network before exiting?")

    logger.info("\nExiting application...")


if __name__ == "__main__":
    main()
