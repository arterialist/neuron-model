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

mnist_test_dataset = None
nn_core_instance = NNCore()
input_neuron_ids = []
MNIST_IMAGE_SIZE = 784  # 28x28 pixels


def load_mnist_data():
    """Loads and prepares the MNIST dataset using PyTorch."""
    global mnist_test_dataset
    logger.info("Loading MNIST dataset using PyTorch...")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalizes to [-1, 1]
        ]
    )

    mnist_test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    logger.info(f"MNIST dataset loaded with {len(mnist_test_dataset)} test images.")


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
    # Each input neuron needs enough synapses to cover its share of the pixels
    synapses_per_input_neuron = math.ceil(MNIST_IMAGE_SIZE / input_size)
    logger.info(
        f"Each of the {input_size} input neurons will have {synapses_per_input_neuron} synapses."
    )

    input_layer_neurons = []
    for i in range(input_size):
        neuron_id = random.randint(0, 2**36 - 1)
        params = NeuronParameters(
            num_inputs=synapses_per_input_neuron, r_base=np.random.uniform(0.8, 1.2)
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
                num_inputs=num_synapses, r_base=np.random.uniform(0.8, 1.2)
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
        i for i, (img, label) in enumerate(mnist_test_dataset) if label == label_to_find  # type: ignore
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
        image_tensor, actual_label = mnist_test_dataset[image_index]  # type: ignore
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
            strength = (float(pixel_value) + 1) * 0.75
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


def main():
    """Main function to run the interactive MNIST application with web visualization."""
    global nn_core_instance, input_neuron_ids

    load_mnist_data()

    # --- Network Setup ---
    choice = input("Load an existing network? (y/n) [n]: ").lower() or "n"
    if choice == "y":
        filepath = input("Enter network file path: ")
        try:
            network_sim = NetworkConfig.load_network_config(filepath)
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
        network_sim = build_network_interactively()
        if not network_sim:
            logger.error("Network building failed. Exiting.")
            return

    nn_core_instance.neural_net = network_sim

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
    logger.info("\n--- MNIST Interaction Terminal ---")
    logger.info("Control the simulation from here. Watch the results in your browser.")
    while True:
        try:
            tick_time_str = (
                input("\nEnter tick time in milliseconds (e.g., 50) [10]: ") or "10"
            )
            tick_sleep_ms = int(tick_time_str)

            label_str = input("Enter a digit (0-9) to present, or 'q' to quit: ")
            if label_str.lower() == "q":
                break

            label = int(label_str)
            if not (0 <= label <= 9):
                logger.warning("Invalid label. Please enter a digit from 0 to 9.")
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
