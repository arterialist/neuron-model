import os
import sys
import argparse
import logging
import random
import numpy as np
import math
import torch
from torchvision import datasets, transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add repo root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron.network import NeuronNetwork
from neuron.neuron import Neuron, NeuronParameters
from neuron.network_config import NetworkConfig
from pipeline.config import NetworkConfig as PydanticNetworkConfig

def build_network(config: PydanticNetworkConfig, output_dir: str):
    """
    Builds the network based on the provided configuration and saves it.
    """
    logger.info("Building network from configuration...")

    # We need to assume some input size. For MNIST/CIFAR standard:
    # We will use hardcoded defaults or heuristics for now since the config
    # doesn't strictly specify input dimensions unless we look at the layers.
    # Assuming MNIST-like input for the example workflow (1x28x28)
    # The original script loaded the dataset to find this out.
    # Here we might need to be explicit or generic.
    # Let's assume standard MNIST 28x28 = 784 inputs for dense, or (1,28,28) for conv.

    # Actually, the user's example JSON shows "in_height": 28, "in_width": 28.
    # We will infer dimensions from the first layer config if it is conv.

    input_height = 28
    input_width = 28
    input_channels = 1

    # Initialize an empty network
    network_sim = NeuronNetwork(num_neurons=0, synapses_per_neuron=0)
    net_topology = network_sim.network
    all_layers = [] # List of lists of neuron IDs

    # Helper to create a neuron
    def create_neuron_instance(layer_idx, layer_name, num_synapses, metadata_extra=None, synapse_dist_fn=None):
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
        metadata = {"layer": layer_idx, "layer_name": layer_name}
        if metadata_extra:
            metadata.update(metadata_extra)

        neuron = Neuron(neuron_id, params, log_level="CRITICAL", metadata=metadata)

        for s_id in range(num_synapses):
            dist = synapse_dist_fn(s_id) if synapse_dist_fn else random.randint(2, 8)
            neuron.add_synapse(s_id, distance_to_hillock=dist)

        for t_id in range(10):
            neuron.add_axon_terminal(t_id, distance_from_hillock=random.randint(2, 8))

        net_topology.neurons[neuron_id] = neuron
        return neuron_id

    # --- Build Input Layer ---
    # We create specific input neurons. For a CNN, we usually model input as a layer 0
    # that matches the image structure.

    # Check first layer config to see if we are doing Conv
    if config.layers and config.layers[0].layer_type == "conv":
         # Calculate receptive field synapses
        k = config.layers[0].kernel_size
        s = config.layers[0].stride
        # For the first layer, input neurons map to image pixels/patches.
        # But wait, in the reference code, "input_layer_neurons" are created SEPARATELY
        # from the "configured layers".
        # If the first configured layer is CONV, the "input layer" is actually the *image source*
        # and the "first configured layer" is the first set of processing neurons.

        # In the provided `interactive_training.py` logic:
        # 1. Create Input Layer (Layer 0) - raw pixels placeholders or receptive fields mapping to image.
        # 2. Create Hidden Layers (Layer 1+)

        # Let's create Layer 0 (Input)
        # Based on the user's example: 2 layers configured.
        # Layer 1: conv. Layer 2: conv.
        # So we need a Layer 0 which is the input.

        # We assume standard MNIST input size
        logger.info(f"Creating Input Layer (Layer 0) for {input_channels}x{input_height}x{input_width} input.")

        # To calculate how many neurons/synapses for layer 0:
        # The reference code uses `synapses_per_input_neuron` and `actual_input_neurons`.
        # For CNNs in `build_network_interactively_v2`, it calls `create_neuron` for input layer.
        # But wait, `build_network_interactively_v2` creates "input_layer_neurons" first.

        # Let's stick to the logic:
        # If it's a CNN, we treat the input image as the source.
        # The "input neurons" in the JSON (layer 0) seem to act as the first processing layer
        # connected to the external input (the image).
        # Looking at the JSON:
        # Neuron metadata: "layer": 0, "layer_type": "conv" ... "in_height": 28 ...
        # This implies Layer 0 IS the first conv layer, receiving inputs from the "image".
        # So we do NOT create a separate dummy input layer of neurons.
        # The neurons in "layer": 0 ARE the input neurons in the sense that `external_inputs` attach to them.

        pass
    else:
        # Dense logic
        pass

    # Actually, looking at `interactive_training.py` V2 builder:
    # It creates `input_layer_neurons` (Layer 0) separately.
    # AND THEN it iterates `layer_configs`.
    # BUT, if `layer_configs[0]` is conv, it says "Conv layers must precede dense".
    # And it uses `prev_coord_to_id` etc.

    # In the JSON provided by user:
    # The neurons have "layer": 0, "layer_name": "conv".
    # This matches the result of `build_network_interactively_v2` where `li=0` in the loop corresponds to the first configured layer.
    # The `input_layer_neurons` created before the loop seem to be ignored or treated as Layer -1?
    # Wait, in V2 builder:
    # `all_layers.append(input_layer_neurons)` -> This is index 0 in `all_layers`.
    # Then loop `li` from 0 to N.
    # Inside loop: `all_layers.append(layer_neurons)`.
    # So `all_layers[0]` is "input", `all_layers[1]` is "conv layer 0".

    # However, in the User's JSON:
    # The neurons start at "layer": 0.
    # This suggests that `input_layer_neurons` from the script might correspond to Layer 0 in JSON?
    # Let's verify `interactive_training.py`:
    # `metadata={"layer": 0, "layer_name": "input"}` for input layer.
    # Then inside the loop `li`, it uses `metadata={"layer_type": "conv", "layer": li, ...}`?
    # No, the loop variable `li` starts at 0.
    # If the metadata in JSON says "layer": 0 is "conv", then the script probably skipped the "Input Layer" creation
    # or the "Input Layer" is implicitly the "layer 0 conv".

    # Let's look closer at `build_network_interactively_v2`:
    # It creates `input_layer_neurons` with `layer=0`.
    # Then in the loop:
    # `nid = create_neuron(li, ...)` -> `li` starts at 0.
    # So we have separate neurons with `layer=0` (input) and `layer=0` (first conv)?
    # That would be confusing.

    # Actually, in the user's JSON, the neurons at layer 0 have `num_inputs: 16`.
    # A 4x4 kernel (kernel_size=4) on 1 channel = 16 inputs.
    # This strongly suggests these ARE the conv neurons processing the image.
    # They are NOT "input placeholder" neurons (which usually have 1 input or represent a pixel).
    # So the "Input Layer" in the script (which connects to the image) is actually what processes the external signals.

    # In `interactive_training.py` V2:
    # `input_layer_neurons` are created. `all_layers.append`.
    # Then loop `layer_configs`.
    # Inside loop (conv):
    # `next_coord_to_id[(...)] = nid`
    # `layer_neurons.append(nid)`
    # ...
    # `all_layers.append(layer_neurons)`

    # If `prev_coord_to_id` is None (first conv layer), it connects to... nothing in the topology?
    # `if prev_coord_to_id is not None: ... else: ...`
    # Wait, the code says:
    # `if prev_coord_to_id is None and all_layers:` -> This check prevents Conv after Dense.
    # But for the FIRST layer (li=0), `prev_coord_to_id` is None.
    # So it creates neurons but does NOT create connections `if prev_coord_to_id is not None`.
    # So the first conv layer has NO incoming connections from other neurons.
    # This confirms: Layer 0 Conv neurons are "Input Neurons" in the sense that they receive External Inputs directly.

    # The `input_layer_neurons` created *before* the loop in `interactive_training.py` seem to be DENSE input neurons.
    # `if layer_configs and layer_configs[0]["type"] == "conv": ... logger.info("CNN first layer detected ...")`
    # In this block, `input_layer_neurons` is NOT populated!
    # Instead, `prev_coord_to_id` is initialized to None.
    # And `prev_channels/h/w` are set from the sample image.

    # So: If first layer is Conv, we skip creating a distinct "input layer" of neurons.
    # The first Conv layer acts as the entry point.

    prev_channels = input_channels
    prev_h = input_height
    prev_w = input_width
    prev_coord_to_id = None

    # We track layers to handle connections
    layers_neurons = []

    for i, layer_cfg in enumerate(config.layers):
        if layer_cfg.layer_type == "conv":
            filters = layer_cfg.filters
            if filters == 0: filters = 1 # Handle 0 as 1
            k = layer_cfg.kernel_size
            s = layer_cfg.stride
            p = layer_cfg.connectivity

            # Compute output dims
            out_h = int(math.floor((prev_h - k) / s) + 1)
            out_w = int(math.floor((prev_w - k) / s) + 1)

            current_layer_neurons = []
            next_coord_to_id = {}

            # Center for distance calc
            center = (k - 1) / 2.0
            def conv_dist(s_id):
                 # calculate distance within kernel
                 # s_id goes from 0 to (k*k*prev_channels)-1
                 # effective kernel pos
                 per_channel = k*k
                 rem = s_id % per_channel
                 ky = rem // k
                 kx = rem % k
                 dx = (kx - center) * 5.0
                 dy = (ky - center) * 5.0
                 base = 5.0
                 return float(math.sqrt(base*base + dx*dx + dy*dy))

            for f in range(filters):
                for y in range(out_h):
                    for x in range(out_w):
                        num_synapses = k * k * prev_channels
                        meta = {
                            "layer": i,
                            "layer_name": "conv",
                            "layer_type": "conv",
                            "filter": f,
                            "y": y,
                            "x": x,
                            "kernel_size": k,
                            "stride": s,
                            "in_channels": prev_channels,
                            "in_height": prev_h,
                            "in_width": prev_w,
                            "out_height": out_h,
                            "out_width": out_w
                        }

                        nid = create_neuron_instance(i, "conv", num_synapses, meta, conv_dist)
                        current_layer_neurons.append(nid)
                        next_coord_to_id[(f, y, x)] = nid

                        # Connect to previous layer if it exists
                        if prev_coord_to_id:
                            # Standard CNN connection logic
                            # Iterate over kernel footprint in previous layer
                            dst_neuron = net_topology.neurons[nid]

                            for c_in in range(prev_channels):
                                for ky in range(k):
                                    for kx in range(k):
                                        in_y = y * s + ky
                                        in_x = x * s + kx

                                        if (c_in, in_y, in_x) in prev_coord_to_id:
                                            if random.random() <= p:
                                                src_id = prev_coord_to_id[(c_in, in_y, in_x)]
                                                src_neuron = net_topology.neurons[src_id]

                                                src_terminals = list(src_neuron.presynaptic_points.keys())
                                                if src_terminals:
                                                    # Map to specific synapse index
                                                    syn_idx = (c_in * k * k) + (ky * k) + kx
                                                    conn = (src_id, random.choice(src_terminals), nid, syn_idx)
                                                    if conn not in net_topology.connections:
                                                        net_topology.connections.append(conn)

            layers_neurons.append(current_layer_neurons)

            # Update state for next layer
            prev_h, prev_w = out_h, out_w
            prev_channels = filters
            prev_coord_to_id = next_coord_to_id

        else:
            # Handle other layer types if added later
            pass

    # --- Setup External Inputs ---
    # In this V2 logic, if the first layer is Conv, it has NO incoming neuron connections.
    # So we attach external inputs to all its synapses.
    if len(layers_neurons) > 0:
        first_layer = layers_neurons[0]
        for nid in first_layer:
            neuron = net_topology.neurons[nid]
            for s_id in neuron.postsynaptic_points:
                 net_topology.external_inputs[(nid, s_id)] = {
                    "info": 0.0,
                    "mod": np.array([0.0, 0.0]),
                }

    # Save network
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "network.json")
    NetworkConfig.save_network_config(network_sim, output_path)
    logger.info(f"Network saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    import yaml
    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        # Extract just the network part
        net_cfg = PydanticNetworkConfig(**cfg_dict['network'])

    build_network(net_cfg, args.output_dir)
