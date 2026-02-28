import os
import random
import sys
import threading
import time
import webbrowser
import logging
import numpy as np

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
from neuron.network_config import NetworkConfig
from cli.web_viz.server import NeuralNetworkWebServer
from torchvision import datasets, transforms

from snn_classification_realtime.core.network_utils import infer_layers_from_metadata
from snn_classification_realtime.network_builder import build_network


selected_dataset = None
nn_core_instance = NNCore()
input_neuron_ids = []
CURRENT_IMAGE_VECTOR_SIZE = 0
CURRENT_NUM_CLASSES = 10
# Global toggle: when True signals can be negative (normalized to [-1, 1]);
# when False signals are normalized to [0, 1].
INHIBITION_ENABLED = False
# Global flag: when True, indicates colored CIFAR-10 with 3 synapses per pixel
IS_COLORED_CIFAR10 = False
# Global flag: when True, creates separate neurons for each RGB channel; when False, uses 3 synapses per neuron for RGB
CIFAR10_RGB_SEPARATE_NEURONS = False
# Dataset name for network builder config (mnist, cifar10_grayscale, cifar10_color, cifar100, etc.)
SELECTED_DATASET_NAME = "mnist"


def select_and_load_dataset():
    """Loads and prepares a selected dataset using torchvision.

    Supported: MNIST, CIFAR10, CIFAR100. Normalizes to [-1, 1].
    Sets globals: selected_dataset, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES, SELECTED_DATASET_NAME.
    """
    global selected_dataset, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES, SELECTED_DATASET_NAME

    logger.info("Select dataset:")
    logger.info("  1) MNIST")
    logger.info("  2) CIFAR10")
    logger.info("  3) CIFAR10 (color)")
    logger.info("  4) CIFAR100")
    logger.info("  5) USPS")
    logger.info("  6) SVHN")
    logger.info("  7) FashionMNIST")
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
        IS_COLORED_CIFAR10 = False
        SELECTED_DATASET_NAME = "mnist"
        logger.info(
            f"Loaded MNIST with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )

    elif choice == "2":
        # CIFAR10 (grayscale - 3x32x32 flattened to 1x32x32 equivalent)
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
        IS_COLORED_CIFAR10 = False
        SELECTED_DATASET_NAME = "cifar10_grayscale"
        logger.info(
            f"Loaded CIFAR10 (grayscale) with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )

    elif choice == "3":
        # CIFAR10 (color) - 32x32 pixels with 3 synapses each (RGB)
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
            raise RuntimeError(f"Failed to load CIFAR10 (color): {last_err}")
        img0, _ = selected_dataset[0]
        # For colored CIFAR-10, ask about architecture choice
        global CIFAR10_RGB_SEPARATE_NEURONS
        logger.info("Choose RGB CIFAR-10 architecture:")
        logger.info("  1) One neuron per spatial kernel, 3 synapses per RGB channel")
        logger.info(
            "  2) One neuron per spatial kernel per RGB channel (3x more neurons)"
        )
        arch_choice = input("Enter choice [1]: ").strip() or "1"
        if arch_choice == "1":
            CIFAR10_RGB_SEPARATE_NEURONS = False
            # vector size is pixels * 3 (RGB channels per neuron)
            CURRENT_IMAGE_VECTOR_SIZE = int(img0.shape[1] * img0.shape[2] * 3)
            logger.info("Selected: One neuron per kernel, 3 synapses per RGB channel")
        elif arch_choice == "2":
            CIFAR10_RGB_SEPARATE_NEURONS = True
            # vector size is pixels * 3 (one channel per neuron, but 3x neurons total)
            CURRENT_IMAGE_VECTOR_SIZE = int(img0.shape[1] * img0.shape[2] * 3)
            logger.info("Selected: One neuron per kernel per RGB channel")
        else:
            logger.warning("Invalid choice, defaulting to option 1")
            CIFAR10_RGB_SEPARATE_NEURONS = False
            CURRENT_IMAGE_VECTOR_SIZE = int(img0.shape[1] * img0.shape[2] * 3)

        CURRENT_NUM_CLASSES = 10
        IS_COLORED_CIFAR10 = True
        SELECTED_DATASET_NAME = "cifar10_color"
        logger.info(
            f"Loaded CIFAR10 (color) with {img0.shape[1]}x{img0.shape[2]} pixels × 3 channels = {CURRENT_IMAGE_VECTOR_SIZE} synapses, {CURRENT_NUM_CLASSES} classes."
        )

    elif choice == "4":
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
        IS_COLORED_CIFAR10 = False
        SELECTED_DATASET_NAME = "cifar100"
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
        IS_COLORED_CIFAR10 = False
        SELECTED_DATASET_NAME = "usps"
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
        IS_COLORED_CIFAR10 = False
        SELECTED_DATASET_NAME = "svhn"
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
        IS_COLORED_CIFAR10 = False
        SELECTED_DATASET_NAME = "fashionmnist"
        logger.info(
            f"Loaded FashionMNIST with vector size {CURRENT_IMAGE_VECTOR_SIZE} and {CURRENT_NUM_CLASSES} classes."
        )
    else:
        raise ValueError("Invalid dataset choice.")


def build_network_interactively():
    """Interactively builds a new neural network with a flexible input layer (all dense)."""
    global input_neuron_ids
    logger.info("\n--- Interactive Network Builder ---")

    try:
        input_size = int(
            input(f"Enter size of input layer (e.g., 20, 100) [100]: ") or 100
        )
        if input_size <= 0:
            logger.error("Input layer size must be positive.")
            return None
        num_hidden_layers = int(input("Enter number of hidden layers [1]: ") or 1)
        hidden_sizes = []
        for i in range(num_hidden_layers):
            size = int(input(f"Enter size of hidden layer {i + 1} [128]: ") or 128)
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

    layers = [
        {"type": "dense", "size": s, "connectivity": connectivity}
        for s in hidden_sizes
    ]
    layers.append({"type": "dense", "size": output_size, "connectivity": connectivity})
    config = {
        "dataset": SELECTED_DATASET_NAME,
        "layers": layers,
        "input_size": input_size,
        "inhibitory_signals": INHIBITION_ENABLED,
        "rgb_separate_neurons": CIFAR10_RGB_SEPARATE_NEURONS,
    }
    network_sim = build_network(config, input_shape=None, logger=logger)
    layers_list = infer_layers_from_metadata(network_sim)
    input_neuron_ids = layers_list[0] if layers_list else []
    return network_sim


def build_network_interactively_v2():
    """Advanced network builder (V2) with per-layer connectivity and optional shortcuts.

    Keeps data structures compatible with existing import/export (neurons, connections, external_inputs).
    """
    global input_neuron_ids
    logger.info("\n--- Advanced Network Builder (V2) ---")

    # Per-layer configuration
    layer_configs: list[dict] = []

    try:
        total_layers = int(
            input("Enter total number of layers after input (>=1) [2]: ") or 2
        )
        if total_layers < 1:
            logger.error("Total layers must be at least 1.")
            return None

        logger.info("For each layer, choose type: conv or dense.")
        for li in range(total_layers):
            default_type = "conv" if li == 0 else "dense"
            ltype = (
                input(f"Layer {li} type (conv/dense) [{default_type}]: ")
                or default_type
            ).lower()
            if ltype not in ("conv", "dense"):
                logger.error("Invalid layer type.")
                return None
            config: dict = {"type": ltype}

            if ltype == "conv":
                if selected_dataset is None:
                    logger.error("Load a dataset before building a CNN-style network.")
                    return None
                filters_toggle = (
                    input(f"Enable multiple filters for conv layer {li}? (y/n) [y]: ")
                    or "y"
                ).lower() == "y"
                if filters_toggle:
                    filters = int(
                        input(f"Number of filters for conv layer {li} [16]: ") or 16
                    )
                    filters = max(1, filters)
                else:
                    filters = 1
                k = int(input(f"Kernel size for conv layer {li} [3]: ") or 3)
                s = int(input(f"Stride for conv layer {li} [1]: ") or 1)
                conn = float(
                    input(f"Connectivity for layer {li}->{li + 1} (0-1) [0.8]: ") or 0.8
                )
                config.update(
                    {
                        "filters": filters,
                        "kernel": max(1, k),
                        "stride": max(1, s),
                        "connectivity": max(0.0, min(1.0, conn)),
                    }
                )
            else:
                size = int(
                    input(f"Number of neurons for dense layer {li} [128]: ") or 128
                )
                syn_prompt = input(
                    f"Synapses per neuron for dense layer {li} (blank=auto if first dense) [10]: "
                )
                synapses_per = None if syn_prompt.strip() == "" else int(syn_prompt)
                conn = float(
                    input(f"Connectivity for layer {li}->{li + 1} (0-1) [0.5]: ") or 0.5
                )
                config.update(
                    {
                        "size": max(1, size),
                        "synapses_per": synapses_per,
                        "connectivity": max(0.0, min(1.0, conn)),
                    }
                )

            layer_configs.append(config)

        # Optional shortcuts (multiple) shared by both modes
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

    global INHIBITION_ENABLED
    INHIBITION_ENABLED = (
        input("Enable inhibitory signals (normalize to [-1,1])? (y/n) [n]: ") or "n"
    ).lower() == "y"

    input_size = 100
    if layer_configs and layer_configs[0]["type"] == "dense":
        input_size = int(input("Enter size of input layer [100]: ") or 100)
        if input_size <= 0:
            logger.error("Input layer size must be positive.")
            return None

    config = {
        "dataset": SELECTED_DATASET_NAME,
        "layers": layer_configs,
        "input_size": input_size,
        "inhibitory_signals": INHIBITION_ENABLED,
        "rgb_separate_neurons": CIFAR10_RGB_SEPARATE_NEURONS,
    }
    input_shape = None
    if layer_configs and layer_configs[0]["type"] == "conv":
        if selected_dataset is None:
            logger.error("Load a dataset before building a CNN-style network.")
            return None
        sample_tensor, _ = selected_dataset[0]  # type: ignore
        input_shape = tuple(sample_tensor.shape)
        if len(input_shape) == 2:
            input_shape = (1,) + input_shape

    network_sim = build_network(
        config,
        input_shape=input_shape,
        shortcut_specs=shortcut_specs,
        logger=logger,
    )
    layers_list = infer_layers_from_metadata(network_sim)
    input_neuron_ids = layers_list[0] if layers_list else []
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
    if selected_dataset is None:
        logger.warning("No dataset loaded.")
        return
    indices = [
        i
        for i, (img, label) in enumerate(selected_dataset)  # type: ignore[arg-type]
        if label == label_to_find
    ]

    if not indices:
        logger.warning(f"No images found with label {label_to_find}")
        return

    tick_sleep_sec = tick_sleep_ms / 1000.0
    num_input_neurons = len(input_neuron_ids)

    def build_signals_for_image(image_tensor):
        """Map an image tensor to (neuron_id, synapse_id, strength) signals based on metadata.

        Supports dense input (legacy), colored CIFAR-10, and CNN input (conv layer at layer 0 with kernel metadata).
        Signals are normalized to [0, 1] range.
        """
        # Detect CNN-style input: first input neuron has layer_type == "conv"
        first_neuron = nn_core_instance.neural_net.network.neurons[input_neuron_ids[0]]  # type: ignore
        meta = getattr(first_neuron, "metadata", {}) or {}
        is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

        if not is_cnn_input:
            if IS_COLORED_CIFAR10:
                arr = image_tensor.detach().cpu().numpy().astype(np.float32)
                if arr.ndim == 3 and arr.shape[0] == 3:  # CHW format
                    h, w = arr.shape[1], arr.shape[2]
                    signals = []

                    if CIFAR10_RGB_SEPARATE_NEURONS:
                        # Architecture 2: One neuron per spatial kernel per RGB channel
                        # Each neuron handles one color channel for a subset of pixels
                        total_spatial_positions = h * w
                        neurons_per_color = len(input_neuron_ids) // 3

                        for y in range(h):
                            for x in range(w):
                                pixel_index = y * w + x
                                for c in range(3):  # RGB channels
                                    # Calculate which spatial neuron handles this pixel for this color
                                    spatial_neuron_idx = pixel_index % neurons_per_color
                                    # Calculate global neuron index: color_offset + spatial_idx
                                    global_neuron_idx = (
                                        c * neurons_per_color + spatial_neuron_idx
                                    )

                                    if global_neuron_idx >= len(input_neuron_ids):
                                        continue  # Skip if neuron index exceeds available neurons

                                    # Each neuron handles one color channel for multiple pixels
                                    pixels_per_neuron = (
                                        total_spatial_positions // neurons_per_color
                                    )
                                    synapse_index = pixel_index // neurons_per_color

                                    neuron_id = input_neuron_ids[global_neuron_idx]
                                    # Normalize based on inhibition toggle
                                    pixel_value = arr[c, y, x]
                                    if INHIBITION_ENABLED:
                                        strength = float(
                                            pixel_value
                                        )  # already in [-1, 1]
                                    else:
                                        strength = (
                                            float(pixel_value) + 1.0
                                        ) * 0.5  # to [0, 1]
                                    signals.append((neuron_id, synapse_index, strength))
                    else:
                        # Architecture 1: One neuron per spatial kernel, 3 synapses per RGB channel
                        for y in range(h):
                            for x in range(w):
                                for c in range(3):  # RGB channels
                                    # Calculate which input neuron handles this pixel
                                    pixel_index = y * w + x
                                    target_neuron_index = (
                                        pixel_index % num_input_neurons
                                    )
                                    # Each neuron handles multiple pixels, each pixel has 3 synapses
                                    pixels_per_neuron = (h * w) // num_input_neurons
                                    if (
                                        pixel_index // num_input_neurons
                                        >= pixels_per_neuron
                                    ):
                                        continue  # This pixel doesn't fit in the network
                                    base_synapse_index = (
                                        pixel_index // num_input_neurons
                                    ) * 3
                                    synapse_index = base_synapse_index + c

                                    neuron_id = input_neuron_ids[target_neuron_index]
                                    # Normalize based on inhibition toggle
                                    pixel_value = arr[c, y, x]
                                    if INHIBITION_ENABLED:
                                        strength = float(
                                            pixel_value
                                        )  # already in [-1, 1]
                                    else:
                                        strength = (
                                            float(pixel_value) + 1.0
                                        ) * 0.5  # to [0, 1]
                                    signals.append((neuron_id, synapse_index, strength))
                    return signals
            else:
                # Legacy dense mapping
                image_vector = image_tensor.view(-1).numpy()
                signals = []
                for pixel_index, pixel_value in enumerate(image_vector):
                    target_neuron_index = pixel_index % num_input_neurons
                    target_synapse_index = pixel_index // num_input_neurons

                    neuron_id = input_neuron_ids[target_neuron_index]
                    # Normalize based on inhibition toggle
                    if INHIBITION_ENABLED:
                        strength = float(pixel_value)  # already in [-1, 1]
                    else:
                        strength = (float(pixel_value) + 1.0) * 0.5  # to [0, 1]
                    signals.append((neuron_id, target_synapse_index, strength))
                return signals

        # CNN mapping: each input neuron is a kernel (receptive field)
        arr = image_tensor.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(f"Unsupported image shape for CNN input: {arr.shape}")

        signals = []
        for neuron_id in input_neuron_ids:
            neuron = nn_core_instance.neural_net.network.neurons[neuron_id]  # type: ignore
            m = getattr(neuron, "metadata", {}) or {}
            k = int(m.get("kernel_size", 1))
            s = int(m.get("stride", 1))
            in_c = int(m.get("in_channels", arr.shape[0]))
            y_out = int(m.get("y", 0))
            x_out = int(m.get("x", 0))
            for c in range(in_c):
                for ky in range(k):
                    for kx in range(k):
                        in_y = y_out * s + ky
                        in_x = x_out * s + kx
                        if in_y >= arr.shape[1] or in_x >= arr.shape[2]:
                            continue
                        syn_id = (c * k + ky) * k + kx
                        if INHIBITION_ENABLED:
                            strength = float(arr[c, in_y, in_x])  # [-1, 1]
                        else:
                            strength = (float(arr[c, in_y, in_x]) + 1.0) * 0.5  # [0, 1]
                        signals.append((neuron_id, syn_id, strength))
        return signals

    for i in range(num_images):
        if randomize_images:
            image_index = random.choice(indices)
        else:
            image_index = indices[i]
        image_tensor, actual_label = selected_dataset[image_index]  # type: ignore

        logger.info(
            f"\nPresenting image #{i + 1}/{num_images} (index: {image_index}, Label: {actual_label}) for {ticks_per_image} ticks..."
        )

        signals = build_signals_for_image(image_tensor)

        for tick_num in range(ticks_per_image):
            nn_core_instance.send_batch_signals(signals)
            nn_core_instance.do_tick()
            if tick_sleep_sec > 0:
                time.sleep(tick_sleep_sec)

        logger.info(f"Finished presenting image #{image_index}.")

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
    global \
        selected_dataset, \
        CURRENT_IMAGE_VECTOR_SIZE, \
        CURRENT_NUM_CLASSES, \
        input_neuron_ids, \
        nn_core_instance

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
                f"Enter a label (0-{CURRENT_NUM_CLASSES - 1}) to present, or 'q' to quit: "
            )
            if label_str.lower() == "q":
                break

            label = int(label_str)
            if not (0 <= label <= CURRENT_NUM_CLASSES - 1):
                logger.warning(
                    f"Invalid label. Please enter a value from 0 to {CURRENT_NUM_CLASSES - 1}."
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
