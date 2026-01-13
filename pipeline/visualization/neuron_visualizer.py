#!/usr/bin/env python3
"""
Real-time Neuron Network Visualizer - Web Interface
Displays per-neuron membrane potential and refractory time graphs with interactive input controls.
Two web pages: one for graphs, one for input controls.
"""

import os
import sys
import time
import threading
import io
import base64
from flask import Flask, render_template_string, request, jsonify
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for web
import matplotlib.pyplot as plt
from collections import deque
import asyncio
import websockets
import json

# Ensure local imports resolve
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.network_config import NetworkConfig

# Global visualizer instance
visualizer = None

app = Flask(__name__)


class NeuronVisualizer:
    """Real-time visualizer for neuron network simulation with web interface."""

    def __init__(self):
        self.network_sim = None
        self.layers = []
        self.neurons_data = {}  # {neuron_id: {'S': deque, 't_ref': deque, 'spikes': deque}}
        self.max_history = 200
        self.tick_time_ms = 100  # Default tick time in milliseconds
        self.is_running = False
        self.simulation_thread = None
        self.signal_strength = 1.0  # Default signal strength
        self.max_t_ref_value = 20  # Default max t_ref value

        # Input controls
        self.active_inputs = (
            set()
        )  # Set of (neuron_id, synapse_id) tuples for active inputs

        # Neuromediator injections (cleared after 1 tick)
        self.pending_injections = {}  # {(neuron_id, synapse_id): mod_array}

        # WebSocket server
        self.websocket_server = None
        self.websocket_clients = set()
        self.websocket_thread = None
        self.websocket_loop = None

        # Throttling for broadcasts (send updates max once per 50ms)
        self.last_broadcast_time = 0
        self.broadcast_throttle_ms = 50
        self.pending_broadcast = False

        # Start WebSocket server
        self.start_websocket_server()

    def load_network(self, file_path):
        """Load a network from JSON file."""
        if not file_path or not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        try:
            # Load network
            network_sim = NetworkConfig.load_network_config(file_path)
            self.network_sim = network_sim
            self.layers = self.infer_layers_from_metadata(network_sim)

            # Initialize neuron data tracking
            self.initialize_neuron_data()

            return (
                True,
                f"Network loaded successfully! Neurons: {len(network_sim.network.neurons)}, Layers: {len(self.layers)}",
            )

        except Exception as e:
            return False, f"Failed to load network: {e}"

    def infer_layers_from_metadata(self, network_sim):
        """Groups neurons by their 'layer' metadata."""
        if network_sim is None:
            return []
        net = network_sim.network
        layer_to_neurons = {}
        for nid, neuron in net.neurons.items():
            layer_idx = int(neuron.metadata.get("layer", 0))
            layer_to_neurons.setdefault(layer_idx, []).append(nid)
        return [layer_to_neurons[k] for k in sorted(layer_to_neurons.keys())]

    def initialize_neuron_data(self):
        """Initialize data structures for tracking neuron states."""
        if self.network_sim is None:
            return

        self.neurons_data = {}
        max_t_ref = 0

        for layer in self.layers:
            for neuron_id in layer:
                neuron = self.network_sim.network.neurons[neuron_id]
                self.neurons_data[neuron_id] = {
                    "S": deque(maxlen=self.max_history),
                    "t_ref": deque(maxlen=self.max_history),
                    "spikes": deque(maxlen=self.max_history),  # Track spike times
                    "r": deque(maxlen=self.max_history),  # Track firing threshold
                    "b": deque(
                        maxlen=self.max_history
                    ),  # Track post-cooldown threshold
                    "layer": neuron.metadata.get("layer", 0),
                }
                max_t_ref = max(max_t_ref, neuron.upper_t_ref_bound)

        self.max_t_ref_value = max_t_ref if max_t_ref > 0 else 20  # Fallback

    def get_input_info(self):
        """Get information about available inputs."""
        if not self.layers or self.network_sim is None:
            return {}

        inputs_info = {}
        for layer_idx, layer in enumerate(self.layers):
            layer_inputs = {}
            for neuron_id in layer:
                neuron = self.network_sim.network.neurons[neuron_id]
                num_synapses = len(neuron.postsynaptic_points)
                if num_synapses > 0:
                    layer_inputs[str(neuron_id)] = list(range(num_synapses))
            if layer_inputs:
                inputs_info[str(layer_idx)] = layer_inputs

        return inputs_info

    def start_simulation(self):
        """Start the simulation."""
        if self.network_sim is None:
            return False, "Please load a network first"

        if self.is_running:
            return False, "Simulation already running"

        self.is_running = True

        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

        return True, f"Simulation started (tick time: {self.tick_time_ms}ms)"

    def stop_simulation(self):
        """Stop the simulation."""
        self.is_running = False
        return "Simulation stopped"

    def run_simulation(self):
        """Run the simulation loop."""
        while self.is_running and self.network_sim is not None:
            # Send active inputs
            for neuron_id, synapse_id in self.active_inputs:
                self.network_sim.set_external_input(
                    neuron_id, synapse_id, self.signal_strength
                )

            # Apply pending neuromediator injections
            # Check for enqueued signals and existing buffer values, set appropriate values
            for (neuron_id, synapse_id), mod in self.pending_injections.items():
                neuron = self.network_sim.network.neurons[neuron_id]

                # Check if this synapse has signals enqueued in propagation queue
                has_enqueued_signal = False
                for (
                    arrival_tick,
                    target_node,
                    V_initial,
                    source_syn_id,
                ) in neuron.propagation_queue:
                    if source_syn_id == synapse_id:
                        has_enqueued_signal = True
                        break

                # Check existing buffer values (may have mod values from propagating signals)
                existing_info = neuron.input_buffer[synapse_id, 0]
                existing_mod = neuron.input_buffer[synapse_id, 2:].copy()

                # Combine injected mod values with existing mod values from enqueued signals
                combined_mod = mod + existing_mod

                # Set info to existing value if present, or small non-zero if enqueued signals exist
                # This ensures the synapse is considered "active" for learning updates
                if existing_info > 0:
                    info_value = existing_info
                elif has_enqueued_signal:
                    info_value = 0.001  # Small value to make synapse active
                else:
                    info_value = 0.0

                self.network_sim.set_external_input(
                    neuron_id, synapse_id, info_value, combined_mod
                )

            # Run simulation tick
            self.network_sim.run_tick()

            # Clear pending injections after 1 tick
            self.pending_injections.clear()

            # Update neuron data
            self.update_neuron_data()

            # If we have pending broadcast and enough time has passed, send it
            if self.pending_broadcast:
                import time as time_module

                current_time = time_module.time() * 1000
                if (
                    current_time - self.last_broadcast_time
                    >= self.broadcast_throttle_ms
                ):
                    self.broadcast_neuron_data()

            # Sleep for tick time
            time.sleep(self.tick_time_ms / 1000.0)

    def update_neuron_data(self):
        """Update neuron data from current simulation state."""
        if self.network_sim is None:
            return
        for neuron_id, data in self.neurons_data.items():
            neuron = self.network_sim.network.neurons[neuron_id]
            data["S"].append(float(neuron.S))
            data["t_ref"].append(float(neuron.t_ref))
            # Track spikes (1 if neuron fired, 0 otherwise)
            data["spikes"].append(1.0 if neuron.O > 0 else 0.0)
            # Track firing thresholds
            data["r"].append(float(neuron.r))
            data["b"].append(float(neuron.b))

            # Keep only the most recent data points
            if len(data["S"]) > self.max_history:
                data["S"].popleft()
                data["t_ref"].popleft()
                data["spikes"].popleft()
                data["r"].popleft()
                data["b"].popleft()

        # Broadcast updated data to WebSocket clients
        self.broadcast_neuron_data()

    def start_websocket_server(self):
        """Start the WebSocket server for real-time data updates."""

        def websocket_server():
            async def ws_handler(websocket):
                """Handle WebSocket connections."""
                client_id = id(websocket)
                self.websocket_clients.add(websocket)
                print(f"WebSocket client connected: {client_id}")

                try:
                    # Send initial data - convert deques to lists and neuron IDs to strings
                    serializable_data = {}
                    for neuron_id, data in self.neurons_data.items():
                        serializable_data[str(neuron_id)] = {
                            "S": list(data["S"]),
                            "t_ref": list(data["t_ref"]),
                            "spikes": list(data["spikes"]),
                            "r": list(data["r"]),
                            "b": list(data["b"]),
                            "layer": data["layer"],
                        }
                    await websocket.send(
                        json.dumps(
                            {"type": "neuron_data", "neurons": serializable_data}
                        )
                    )

                    # Keep connection alive
                    async for message in websocket:
                        # Handle any client messages if needed
                        pass

                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.discard(websocket)
                    print(f"WebSocket client disconnected: {client_id}")

            # Start WebSocket server
            async def start_server():
                server = await websockets.serve(ws_handler, "localhost", 5556)
                await server.wait_closed()

            # Run in separate thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.websocket_loop = loop
            loop.run_until_complete(start_server())

        self.websocket_thread = threading.Thread(target=websocket_server, daemon=True)
        self.websocket_thread.start()
        print("WebSocket server started on ws://localhost:5556")

    def broadcast_neuron_data(self):
        """Broadcast current neuron data to all connected WebSocket clients (throttled)."""
        if not self.websocket_clients or not self.websocket_loop:
            return

        # Throttle broadcasts to avoid overwhelming the client
        import time as time_module

        current_time = time_module.time() * 1000  # Convert to milliseconds
        time_since_last = current_time - self.last_broadcast_time

        if time_since_last < self.broadcast_throttle_ms:
            # Mark that we have pending data but don't send yet
            self.pending_broadcast = True
            return

        self.last_broadcast_time = current_time
        self.pending_broadcast = False

        # Convert deques to lists and ensure neuron IDs are strings for JSON compatibility
        # Only send the last N points to reduce data size
        max_points_to_send = 100  # Reduce from 200 to 100 for performance
        serializable_data = {}
        for neuron_id, data in self.neurons_data.items():
            # Get only the most recent data points
            s_list = list(data["S"])
            t_ref_list = list(data["t_ref"])
            spikes_list = list(data["spikes"])
            r_list = list(data["r"])
            b_list = list(data["b"])

            # Take only last max_points_to_send
            if len(s_list) > max_points_to_send:
                s_list = s_list[-max_points_to_send:]
                t_ref_list = t_ref_list[-max_points_to_send:]
                spikes_list = spikes_list[-max_points_to_send:]
                r_list = r_list[-max_points_to_send:]
                b_list = b_list[-max_points_to_send:]

            serializable_data[str(neuron_id)] = {
                "S": s_list,
                "t_ref": t_ref_list,
                "spikes": spikes_list,
                "r": r_list,
                "b": b_list,
                "layer": data["layer"],
            }

        message = json.dumps({"type": "neuron_data", "neurons": serializable_data})

        # Send message to all connected clients using the websocket event loop
        async def send_to_clients():
            disconnected_clients = set()
            for client in list(self.websocket_clients):
                try:
                    await client.send(message)
                except Exception as e:
                    print(f"Error sending to client: {e}")
                    disconnected_clients.add(client)

            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients

        # Schedule the coroutine in the websocket event loop
        try:
            asyncio.run_coroutine_threadsafe(send_to_clients(), self.websocket_loop)
        except Exception as e:
            print(f"Error broadcasting data: {e}")

    def generate_plot(self):
        """Generate matplotlib plot and return as base64 encoded image."""
        if not self.neurons_data:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(
                0.5, 0.5, "No network loaded", ha="center", va="center", fontsize=16
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # Calculate layout: layers horizontally, neurons vertically within each layer
            num_layers = len(self.layers)
            max_neurons_per_layer = (
                max(len(layer) for layer in self.layers) if self.layers else 1
            )

            fig, axes = plt.subplots(
                max_neurons_per_layer,
                num_layers,
                figsize=(4 * num_layers, 3 * max_neurons_per_layer),
                squeeze=False,
            )

            fig.suptitle("Neuron Network Visualization", fontsize=16)

            for layer_idx in range(num_layers):
                layer = self.layers[layer_idx]

                for neuron_pos in range(max_neurons_per_layer):
                    ax = axes[neuron_pos, layer_idx]

                    if neuron_pos < len(layer):
                        neuron_id = layer[neuron_pos]
                        data = self.neurons_data[neuron_id]

                        # Plot data if available
                        if data["S"]:
                            time_axis = list(range(len(data["S"])))

                            # Red line for membrane potential (S) - scaled 0-20
                            s_values = [min(20, max(0, s)) for s in data["S"]]
                            ax.plot(
                                time_axis,
                                s_values,
                                "r-",
                                linewidth=2,
                                label="Membrane Potential (S)",
                            )

                            # Blue line for t_ref - scaled 0 to max_t_ref_value
                            t_ref_values = [
                                min(self.max_t_ref_value, max(0, t))
                                for t in data["t_ref"]
                            ]
                            ax.plot(
                                time_axis,
                                t_ref_values,
                                "b-",
                                linewidth=2,
                                label="Refractory Time (t_ref)",
                            )

                            ax.set_title(f"Neuron {neuron_id}")
                            ax.set_xlabel("Time (ticks)")
                            ax.set_ylabel("Value")
                            ax.legend(loc="upper right")
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim(0, max(20, self.max_t_ref_value))
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                f"Neuron {neuron_id}\n(No data yet)",
                                ha="center",
                                va="center",
                                fontsize=12,
                            )
                            ax.set_xlim(0, self.max_history)
                            ax.set_ylim(0, max(20, self.max_t_ref_value))
                    else:
                        # Empty subplot
                        ax.axis("off")

            plt.tight_layout()

        # Convert plot to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        return image_base64

    def toggle_input(self, neuron_id, synapse_id):
        """Toggle input signal for a neuron synapse."""
        key = (int(neuron_id), int(synapse_id))
        if key in self.active_inputs:
            self.active_inputs.remove(key)
            return "stopped"
        else:
            self.active_inputs.add(key)
            return "started"

    def set_signal_strength(self, strength):
        """Set signal strength."""
        try:
            self.signal_strength = max(0.0, min(1.0, float(strength)))
            return f"Signal strength set to {self.signal_strength:.1f}"
        except ValueError:
            return "Invalid strength value"

    def set_tick_time(self, tick_time):
        """Set tick time."""
        try:
            ms = int(tick_time)
            if 1 <= ms <= 10000:
                self.tick_time_ms = ms
                return f"Tick time set to {self.tick_time_ms}ms"
            else:
                return "Tick time must be between 1 and 10000 milliseconds"
        except ValueError:
            return "Invalid tick time value"

    def inject_neuromediator(self, layer_idx, mod_0, mod_1):
        """Inject neuromediators to all neurons in a layer for 1 tick."""
        if self.network_sim is None:
            return False, "No network loaded"

        if layer_idx < 0 or layer_idx >= len(self.layers):
            return False, f"Invalid layer index: {layer_idx}"

        import numpy as np

        mod = np.array([float(mod_0), float(mod_1)])
        layer = self.layers[layer_idx]

        # Add to pending injections (will be applied on next tick and cleared after)
        injected_count = 0
        for neuron_id in layer:
            neuron = self.network_sim.network.neurons[neuron_id]
            # Inject to all synapses of this neuron
            for synapse_id in neuron.postsynaptic_points.keys():
                self.pending_injections[(neuron_id, synapse_id)] = mod
                injected_count += 1

        return (
            True,
            f"Queued neuromediator injection [mod_0={mod_0:.2f}, mod_1={mod_1:.2f}] to {injected_count} synapses in layer {layer_idx} (will apply for 1 tick)",
        )


# HTML templates
GRAPHS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Neuron Network Graphs</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .controls { margin-bottom: 20px; }
        button { margin: 5px; padding: 10px 15px; }
        .status { margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }
        .layers-container { display: flex; flex-direction: row; gap: 20px; overflow-x: auto; }
        .layer-column { display: flex; flex-direction: column; min-width: 350px; flex: 0 0 auto; }
        .layer-title { font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #333; text-align: center; padding: 5px; background: #f5f5f5; border-radius: 4px; }
        .neuron-charts { display: flex; flex-direction: column; gap: 10px; }
        .neuron-chart { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px; min-width: 300px; flex: 1; height: 250px; overflow: hidden; }
        .neuron-chart h4 { margin-top: 0; margin-bottom: 8px; color: #333; font-size: 14px; }
        .neuron-chart canvas { width: 100% !important; height: 200px !important; max-width: 100%; max-height: 200px; }
    </style>
</head>
<body>
    <h1>Neuron Network Graphs</h1>

    <div class="controls">
        <button onclick="loadNetwork()">Load Network</button>
        <button onclick="startSimulation()">Start Simulation</button>
        <button onclick="stopSimulation()">Stop Simulation</button>
        <label>Tick Time (ms): <input type="number" id="tickTime" value="100" min="1" max="10000"></label>
        <button onclick="setTickTime()">Set Tick Time</button>
        <button onclick="refreshPlot()">Refresh Plot</button>
    </div>

    <div id="status" class="status">Ready</div>

    <div id="plot-container">
        <div id="neuron-charts" class="neuron-grid">
            <!-- Charts will be dynamically added here -->
        </div>
    </div>

    <script>
        let charts = {};
        let ws = null;
        const maxDataPoints = 100;
        let updateQueue = new Map(); // Queue for batched chart updates
        let updateScheduled = false;

        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }

        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:5556');

            ws.onopen = function() {
                console.log('WebSocket connected');
                updateStatus('Connected to server');
            };

            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'neuron_data') {
                        // Queue updates instead of processing immediately
                        queueChartUpdates(data.neurons);
                    }
                } catch (e) {
                    console.error('Error parsing WebSocket data:', e);
                }
            };

            ws.onclose = function() {
                console.log('WebSocket disconnected');
                updateStatus('Disconnected from server');
                // Attempt to reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateStatus('WebSocket error');
            };
        }

        function loadNetwork() {
            const filePath = prompt("Enter path to network JSON file:");
            if (filePath) {
                fetch('/api/load_network', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_path: filePath })
                })
                .then(response => response.json())
                .then(data => {
                    updateStatus(data.message);
                    if (data.success) {
                        // Request network structure instead of reloading page
                        fetch('/api/network_structure')
                        .then(response => response.json())
                        .then(structure => {
                            console.log('Network structure received:', structure);
                            if (structure.layers && structure.layers.length > 0) {
                                createCharts(structure.layers);
                                updateStatus('Network loaded, charts ready');
                            } else {
                                updateStatus('Network loaded but no layers found');
                            }
                        })
                        .catch(error => {
                            console.error('Error loading network structure:', error);
                            updateStatus('Error loading network structure');
                        });
                    }
                })
                .catch(error => updateStatus('Error: ' + error.message));
            }
        }

        function createCharts(layers) {
            const container = document.getElementById('neuron-charts');
            if (!container) {
                console.error('Container element not found');
                return;
            }
            container.innerHTML = ''; // Clear existing charts
            charts = {}; // Reset charts object

            console.log('Creating charts for', layers.length, 'layers');

            // Create a container for all layers (horizontal layout)
            const layersContainer = document.createElement('div');
            layersContainer.className = 'layers-container';

            layers.forEach((layerNeurons, layerIdx) => {
                // Create layer column container
                const layerColumn = document.createElement('div');
                layerColumn.className = 'layer-column';

                // Create compact layer title
                const layerTitle = document.createElement('div');
                layerTitle.className = 'layer-title';
                layerTitle.textContent = `Layer ${layerIdx}`;
                layerColumn.appendChild(layerTitle);

                const chartsContainer = document.createElement('div');
                chartsContainer.className = 'neuron-charts';

                layerNeurons.forEach((neuronId, neuronPos) => {
                    // Ensure neuronId is a string for consistent key matching
                    const neuronIdStr = String(neuronId);
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'neuron-chart';

                    // Create title
                    const title = document.createElement('h4');
                    title.textContent = `Neuron ${neuronIdStr}`;
                    chartDiv.appendChild(title);

                    // Create canvas element directly with explicit dimensions
                    const canvas = document.createElement('canvas');
                    canvas.id = `chart-${neuronIdStr}`;
                    canvas.width = 400;  // Explicit width
                    canvas.height = 200; // Explicit height
                    canvas.style.width = '100%';
                    canvas.style.height = '200px';
                    canvas.style.maxWidth = '100%';
                    canvas.style.maxHeight = '200px';
                    chartDiv.appendChild(canvas);

                    chartsContainer.appendChild(chartDiv);

                    // Get context from the canvas element we just created
                    const ctx = canvas.getContext('2d');
                    if (!ctx) {
                        console.error(`Failed to get 2d context for neuron ${neuronIdStr}`);
                        return;
                    }
                    charts[neuronIdStr] = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Membrane Potential (S)',
                                data: [],
                                borderColor: 'rgb(255, 99, 132)',
                                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                borderWidth: 2,
                                fill: false,
                                tension: 0.1,
                                pointRadius: 0, // Hide points for better performance
                                pointHoverRadius: 3
                            }, {
                                label: 'Refractory Time (t_ref)',
                                data: [],
                                borderColor: 'rgb(54, 162, 235)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                borderWidth: 2,
                                fill: false,
                                tension: 0.1,
                                yAxisID: 'y1',
                                pointRadius: 0, // Hide points for better performance
                                pointHoverRadius: 3
                            }, {
                                label: 'Spikes',
                                data: [],
                                type: 'bar',
                                backgroundColor: 'rgba(255, 0, 0, 0.6)',
                                borderColor: 'rgba(255, 0, 0, 1)',
                                borderWidth: 1,
                                yAxisID: 'y',
                                barThickness: 2,
                                maxBarThickness: 2
                            }, {
                                label: 'Firing Threshold (r)',
                                data: [],
                                borderColor: 'rgb(0, 200, 0)',
                                backgroundColor: 'rgba(0, 200, 0, 0.1)',
                                borderWidth: 2,
                                fill: false,
                                tension: 0,
                                pointRadius: 0,
                                pointHoverRadius: 3,
                                yAxisID: 'y'
                            }, {
                                label: 'Post-Cooldown Threshold (b)',
                                data: [],
                                borderColor: 'rgb(128, 0, 128)',
                                backgroundColor: 'rgba(128, 0, 128, 0.1)',
                                borderWidth: 2,
                                fill: false,
                                tension: 0,
                                pointRadius: 0,
                                pointHoverRadius: 3,
                                yAxisID: 'y'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            layout: {
                                padding: {
                                    top: 5,
                                    bottom: 5,
                                    left: 5,
                                    right: 5
                                }
                            },
                            animation: {
                                duration: 0 // Disable animations for real-time updates
                            },
                            interaction: {
                                mode: 'index',
                                intersect: false
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Time (ticks)'
                                    },
                                    max: 200 // Limit to 200 ticks
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Membrane Potential (S)'
                                    },
                                    beginAtZero: true,
                                    max: 1, // Initial value, will be updated dynamically
                                    min: 0,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.1)'
                                    },
                                    ticks: {
                                        precision: 1
                                    }
                                },
                                y1: {
                                    position: 'right',
                                    title: {
                                        display: true,
                                        text: 'Refractory Time (t_ref)'
                                    },
                                    beginAtZero: true,
                                    max: 20, // More conservative maximum
                                    min: 0,
                                    grid: {
                                        drawOnChartArea: false
                                    }
                                }
                            },
                            plugins: {
                                legend: {
                                    display: true,
                                    position: 'top'
                                },
                                tooltip: {
                                    mode: 'index',
                                    intersect: false
                                }
                            }
                        }
                    });
                });

                layerColumn.appendChild(chartsContainer);
                layersContainer.appendChild(layerColumn);
            });

            container.appendChild(layersContainer);
            console.log('Created', Object.keys(charts).length, 'charts');
        }

        function queueChartUpdates(neuronsData) {
            // Merge new data into update queue
            Object.entries(neuronsData).forEach(([neuronId, data]) => {
                updateQueue.set(neuronId, data);
            });

            // Schedule batched update if not already scheduled
            if (!updateScheduled) {
                updateScheduled = true;
                requestAnimationFrame(processChartUpdates);
            }
        }

        function processChartUpdates() {
            updateScheduled = false;

            // Process all queued updates in a single batch
            if (updateQueue.size === 0) return;

            const neuronsData = Object.fromEntries(updateQueue);
            updateQueue.clear();

            updateCharts(neuronsData);
        }

        function updateCharts(neuronsData) {
            // Use requestAnimationFrame for smooth updates
            const neuronIds = Object.keys(neuronsData);
            let index = 0;
            const batchSize = 10; // Update 10 charts per frame

            function updateBatch() {
                const end = Math.min(index + batchSize, neuronIds.length);
                for (let i = index; i < end; i++) {
                    const neuronId = neuronIds[i];
                    const data = neuronsData[neuronId];
                    updateSingleChart(neuronId, data);
                }
                index = end;

                if (index < neuronIds.length) {
                    requestAnimationFrame(updateBatch);
                }
            }

            updateBatch();
        }

        function updateSingleChart(neuronId, data) {
            // Ensure neuronId is a string for consistent key matching
            const neuronIdStr = String(neuronId);
            const chart = charts[neuronIdStr];
            if (!chart) {
                return; // Silently skip missing charts
            }

            // Data is already limited on server side, but ensure we don't exceed max
            const maxDataPoints = 100;
            const dataLength = Math.min(data.S.length, data.t_ref.length);
            const startIdx = Math.max(0, dataLength - maxDataPoints);

            // Ensure we have matching data lengths
            const sData = data.S.slice(startIdx);
            const tRefData = data.t_ref.slice(startIdx);
            const spikesData = (data.spikes || []).slice(startIdx);
            const rData = (data.r || []).slice(startIdx);
            const bData = (data.b || []).slice(startIdx);

            // Double-check that arrays are the same length
            const minLength = Math.min(sData.length, tRefData.length, spikesData.length || sData.length, rData.length || sData.length, bData.length || sData.length);
            if (minLength === 0) return; // Skip if no data

            let finalSData = sData.slice(-minLength);
            let finalTRefData = tRefData.slice(-minLength);
            let finalSpikesData = (spikesData.length > 0 ? spikesData : Array(minLength).fill(0)).slice(-minLength);
            let finalRData = (rData.length > 0 ? rData : Array(minLength).fill(0)).slice(-minLength);
            let finalBData = (bData.length > 0 ? bData : Array(minLength).fill(0)).slice(-minLength);

            // Clamp data values to prevent chart scaling issues (optimized)
            const clamp = (val) => Math.max(0, Math.min(20, val));
            finalSData = finalSData.map(clamp);
            finalTRefData = finalTRefData.map(clamp);
            finalRData = finalRData.map(clamp);
            finalBData = finalBData.map(clamp);

            // Scale spikes to be visible
            const maxSValue = finalSData.length > 0 ? Math.max(...finalSData) : 1;
            const spikeHeight = Math.max(1, Math.min(20, maxSValue || 1));
            finalSpikesData = finalSpikesData.map(val => val > 0 ? spikeHeight : null);

            const timeLabels = Array.from({length: finalSData.length}, (_, i) => i);

            // Calculate autoscaled Y axis limits for membrane potential
            const maxS = finalSData.length > 0 ? Math.max(...finalSData) : 1;
            const yMax = Math.max(1, Math.min(20, maxS));
            const yMin = 0;

            // Update chart data (batch DOM updates)
            chart.data.labels = timeLabels;
            chart.data.datasets[0].data = finalSData;
            chart.data.datasets[1].data = finalTRefData;
            chart.data.datasets[2].data = finalSpikesData;
            chart.data.datasets[3].data = finalRData;
            chart.data.datasets[4].data = finalBData;

            // Update Y axis scale with autoscaling
            chart.options.scales.y.max = yMax;
            chart.options.scales.y.min = yMin;
            chart.options.scales.y1.max = 20;
            chart.options.scales.y1.min = 0;

            chart.update('none'); // Update without animation for real-time performance
        }

        function startSimulation() {
            fetch('/api/start_simulation', { method: 'POST' })
            .then(response => response.json())
            .then(data => updateStatus(data.message));
        }

        function stopSimulation() {
            fetch('/api/stop_simulation', { method: 'POST' })
            .then(response => response.json())
            .then(data => updateStatus(data.message));
        }

        function setTickTime() {
            const tickTime = document.getElementById('tickTime').value;
            fetch('/api/set_tick_time', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tick_time: tickTime })
            })
            .then(response => response.json())
            .then(data => updateStatus(data.message));
        }

        // Initialize WebSocket connection
        connectWebSocket();
    </script>
</body>
</html>
"""

CONTROLS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Neuron Input Controls</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .layer { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .neuron { margin: 10px 0; }
        .synapse-btn { margin: 2px; padding: 8px 12px; border: none; border-radius: 3px;
                      cursor: pointer; transition: background-color 0.2s; }
        .synapse-btn.active { background-color: #4CAF50; color: white; }
        .synapse-btn.inactive { background-color: #f44336; color: white; }
        .controls { margin-bottom: 20px; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        input[type="range"] { width: 200px; }
        .injection-controls { margin: 20px 0; padding: 15px; background: #e8f5e9; border: 1px solid #4CAF50; border-radius: 5px; }
        .injection-controls h3 { margin-top: 0; color: #2e7d32; }
        .injection-inputs { display: flex; gap: 15px; align-items: center; margin: 10px 0; }
        .injection-inputs label { display: flex; flex-direction: column; gap: 5px; }
        .injection-inputs input[type="number"] { width: 100px; padding: 5px; }
        .inject-btn { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .inject-btn:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <h1>Neuron Input Controls</h1>

    <div class="controls">
        <label>Signal Strength: <input type="range" id="signalStrength" min="0" max="1" step="0.1" value="1.0">
        <span id="strengthValue">1.0</span></label>
        <button onclick="setSignalStrength()">Set Strength</button>
        <button onclick="loadInputs()">Refresh Inputs</button>
    </div>

    <div id="inputs-container">
        <p>Loading input controls...</p>
    </div>

    <div id="injection-container">
        <h2>Neuromediator Injection Controls</h2>
        <p>Loading injection controls...</p>
    </div>

    <script>
        document.getElementById('signalStrength').addEventListener('input', function() {
            document.getElementById('strengthValue').textContent = this.value;
        });

        function setSignalStrength() {
            const strength = document.getElementById('signalStrength').value;
            fetch('/api/set_signal_strength', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strength: strength })
            });
        }

        function toggleInput(neuronId, synapseId, button) {
            fetch('/api/toggle_input', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ neuron_id: neuronId, synapse_id: synapseId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.action === 'started') {
                    button.className = 'synapse-btn active';
                    button.textContent = `${synapseId} (ON)`;
                } else {
                    button.className = 'synapse-btn inactive';
                    button.textContent = `${synapseId} (OFF)`;
                }
            });
        }

        function loadInputs() {
            fetch('/api/input_info')
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('inputs-container');
                container.innerHTML = '';

                for (const [layerId, layerData] of Object.entries(data)) {
                    const layerDiv = document.createElement('div');
                    layerDiv.className = 'layer';
                    layerDiv.innerHTML = `<h3>Layer ${layerId}</h3>`;

                    for (const [neuronId, synapses] of Object.entries(layerData)) {
                        const neuronDiv = document.createElement('div');
                        neuronDiv.className = 'neuron';
                        neuronDiv.innerHTML = `<h4>Neuron ${neuronId}</h4>`;

                        synapses.forEach(synapseId => {
                            const button = document.createElement('button');
                            button.className = 'synapse-btn inactive';
                            button.textContent = `${synapseId} (OFF)`;
                            button.onclick = () => toggleInput(neuronId, synapseId, button);
                            neuronDiv.appendChild(button);
                        });

                        layerDiv.appendChild(neuronDiv);
                    }

                    container.appendChild(layerDiv);
                }
            })
            .catch(error => {
                document.getElementById('inputs-container').innerHTML =
                    '<p>Error loading input controls. Make sure a network is loaded.</p>';
            });
        }

        function injectNeuromediator(layerIdx) {
            const mod0Input = document.getElementById(`mod0-layer-${layerIdx}`);
            const mod1Input = document.getElementById(`mod1-layer-${layerIdx}`);
            const mod0 = parseFloat(mod0Input.value) || 0.0;
            const mod1 = parseFloat(mod1Input.value) || 0.0;

            fetch('/api/inject_neuromediator', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layer_idx: layerIdx, mod_0: mod0, mod_1: mod1 })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                alert('Error injecting neuromediator: ' + error.message);
            });
        }

        function loadInjectionControls() {
            fetch('/api/network_structure')
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('injection-container');
                container.innerHTML = '<h2>Neuromediator Injection Controls</h2>';

                if (data.layers && data.layers.length > 0) {
                    data.layers.forEach((layerNeurons, layerIdx) => {
                        const injectionDiv = document.createElement('div');
                        injectionDiv.className = 'injection-controls';
                        injectionDiv.innerHTML = `
                            <h3>Layer ${layerIdx}</h3>
                            <div class="injection-inputs">
                                <label>
                                    <span>Mod 0:</span>
                                    <input type="number" id="mod0-layer-${layerIdx}" value="0.0" step="0.1" min="-10" max="10">
                                </label>
                                <label>
                                    <span>Mod 1:</span>
                                    <input type="number" id="mod1-layer-${layerIdx}" value="0.0" step="0.1" min="-10" max="10">
                                </label>
                                <button class="inject-btn" onclick="injectNeuromediator(${layerIdx})">Inject to Layer ${layerIdx}</button>
                            </div>
                        `;
                        container.appendChild(injectionDiv);
                    });
                } else {
                    container.innerHTML += '<p>No network loaded. Load a network first.</p>';
                }
            })
            .catch(error => {
                document.getElementById('injection-container').innerHTML =
                    '<h2>Neuromediator Injection Controls</h2><p>Error loading injection controls. Make sure a network is loaded.</p>';
            });
        }

        // Load inputs and injection controls on page load
        loadInputs();
        loadInjectionControls();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Main graphs page."""
    return render_template_string(GRAPHS_HTML)


@app.route("/controls")
def controls():
    """Input controls page."""
    return render_template_string(CONTROLS_HTML)


@app.route("/api/network_structure")
def get_network_structure():
    """Get the network structure (layers and neurons)."""
    if visualizer and visualizer.layers:
        # Convert neuron IDs to strings for consistency with JSON serialization
        layers_str = [
            [str(neuron_id) for neuron_id in layer] for layer in visualizer.layers
        ]
        return jsonify({"layers": layers_str})
    else:
        return jsonify({"layers": []})


@app.route("/api/neuron_data")
def get_neuron_data():
    """Get current neuron data for real-time updates."""
    if visualizer:
        # Convert deques to lists and ensure neuron IDs are strings for JSON compatibility
        serializable_data = {}
        for neuron_id, data in visualizer.neurons_data.items():
            serializable_data[str(neuron_id)] = {
                "S": list(data["S"]),
                "t_ref": list(data["t_ref"]),
                "spikes": list(data["spikes"]),
                "r": list(data["r"]),
                "b": list(data["b"]),
                "layer": data["layer"],
            }
        return jsonify({"neurons": serializable_data})
    else:
        return jsonify({"neurons": {}})


@app.route("/api/load_network", methods=["POST"])
def load_network():
    """Load a network from file."""
    data = request.get_json()
    file_path = data.get("file_path", "")

    if visualizer:
        success, message = visualizer.load_network(file_path)
        return jsonify({"success": success, "message": message})
    else:
        return jsonify({"success": False, "message": "Visualizer not initialized"})


@app.route("/api/start_simulation", methods=["POST"])
def start_simulation():
    """Start the simulation."""
    if visualizer:
        success, message = visualizer.start_simulation()
        return jsonify({"success": success, "message": message})
    else:
        return jsonify({"success": False, "message": "Visualizer not initialized"})


@app.route("/api/stop_simulation", methods=["POST"])
def stop_simulation():
    """Stop the simulation."""
    if visualizer:
        message = visualizer.stop_simulation()
        return jsonify({"message": message})
    else:
        return jsonify({"message": "Visualizer not initialized"})


@app.route("/api/input_info")
def get_input_info():
    """Get information about available inputs."""
    if visualizer:
        return jsonify(visualizer.get_input_info())
    else:
        return jsonify({})


@app.route("/api/toggle_input", methods=["POST"])
def toggle_input():
    """Toggle input signal for a neuron synapse."""
    data = request.get_json()
    neuron_id = data.get("neuron_id")
    synapse_id = data.get("synapse_id")

    if visualizer and neuron_id is not None and synapse_id is not None:
        action = visualizer.toggle_input(neuron_id, synapse_id)
        return jsonify({"action": action})
    else:
        return jsonify({"error": "Invalid parameters"})


@app.route("/api/set_signal_strength", methods=["POST"])
def set_signal_strength():
    """Set signal strength."""
    data = request.get_json()
    strength = data.get("strength")

    if visualizer:
        message = visualizer.set_signal_strength(strength)
        return jsonify({"message": message})
    else:
        return jsonify({"message": "Visualizer not initialized"})


@app.route("/api/set_tick_time", methods=["POST"])
def set_tick_time():
    """Set tick time."""
    data = request.get_json()
    tick_time = data.get("tick_time")

    if visualizer:
        message = visualizer.set_tick_time(tick_time)
        return jsonify({"message": message})
    else:
        return jsonify({"message": "Visualizer not initialized"})


@app.route("/api/inject_neuromediator", methods=["POST"])
def inject_neuromediator():
    """Inject neuromediators to a layer."""
    data = request.get_json()
    layer_idx = data.get("layer_idx")
    mod_0 = data.get("mod_0", 0.0)
    mod_1 = data.get("mod_1", 0.0)

    if visualizer:
        success, message = visualizer.inject_neuromediator(layer_idx, mod_0, mod_1)
        return jsonify({"success": success, "message": message})
    else:
        return jsonify({"success": False, "message": "Visualizer not initialized"})


def main():
    """Main entry point."""
    global visualizer
    visualizer = NeuronVisualizer()

    print("Neuron Network Visualizer")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser to:")
    print("  Graphs: http://localhost:8001/")
    print("  Controls: http://localhost:8001/controls")
    print()
    print("Press Ctrl+C to stop the server")

    try:
        app.run(host="0.0.0.0", port=8001, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if visualizer:
            visualizer.stop_simulation()


if __name__ == "__main__":
    main()
