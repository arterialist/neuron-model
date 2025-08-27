#!/usr/bin/env python3
"""
Web server for neural network visualization using Flask and WebSocket.
Provides REST API endpoints and real-time updates via WebSocket.
"""

import json
import threading
import time
from typing import Dict, Any, Optional, Set
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import logging
import asyncio
import websockets
import threading
import json
import traceback

from .data_transformer import NetworkDataTransformer


class NeuralNetworkWebServer:
    """Web server for neural network visualization."""

    def __init__(
        self, nn_core, host: str = "127.0.0.1", port: int = 5555, debug: bool = False
    ):
        self.nn_core = nn_core
        self.host = host
        self.port = port
        self.debug = debug

        # Initialize Flask app
        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        self.app.config["SECRET_KEY"] = "neural_network_viz_secret_key"

        # Enable CORS for all domains
        CORS(self.app, origins="*")

        # WebSocket server
        self.websocket_server = None
        self.websocket_clients = set()
        self.websocket_thread = None
        self.websocket_loop = None

        # Data transformer
        self.transformer = NetworkDataTransformer()

        # Update thread control
        self.update_thread = None
        self.update_thread_running = False

        # Load configuration
        from .config import WebVizConfig

        config = WebVizConfig()
        self.update_interval = config.update_interval  # Use config value
        self.min_update_interval = config.min_update_interval  # Use config value

        # Setup routes and start WebSocket server
        self._setup_routes()
        self._start_websocket_server()

        # Configure logging
        if not debug:
            logging.getLogger("werkzeug").setLevel(logging.WARNING)
            logging.getLogger("socketio").setLevel(logging.WARNING)
            logging.getLogger("engineio").setLevel(logging.WARNING)

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve the main visualization page."""
            return render_template("index.html")

        @self.app.route("/api/test")
        def test_endpoint():
            """Simple test endpoint to verify Flask is working."""
            return jsonify({"status": "ok", "message": "Flask server is working"})

        @self.app.route("/api/network/state")
        def get_network_state():
            """Get current network state in Cytoscape.js format."""
            try:
                # Check if neural network core is available
                if not hasattr(self, 'nn_core') or self.nn_core is None:
                    return jsonify({"error": "Neural network not initialized"}), 503
                
                raw_state = self.nn_core.get_network_state()
                if not raw_state:
                    return jsonify({"error": "Failed to get network state"}), 500
                
                transformed_state = self.transformer.transform_network_state(raw_state)
                return jsonify(transformed_state)
            except Exception as e:
                print(f"Error in get_network_state: {e}")
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/style")
        def get_network_style():
            """Get Cytoscape.js style definition."""
            try:
                style = self.transformer.get_cytoscape_style()
                return jsonify(style)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/edge-colors")
        def get_edge_colors():
            """Get updated edge colors based on current firing states."""
            try:
                # Check if neural network core is available
                if not hasattr(self, 'nn_core') or self.nn_core is None:
                    return jsonify({"error": "Neural network not initialized"}), 503
                
                raw_state = self.nn_core.get_network_state()
                if not raw_state:
                    return jsonify({"error": "Failed to get network state"}), 500
                
                neurons = raw_state.get("network", {}).get("neurons", {})
                if not neurons:
                    return jsonify({"error": "No neurons found in network state"}), 500
                
                # Get current edges from the transformed state
                transformed_state = self.transformer.transform_network_state(raw_state)
                edges = transformed_state.get("elements", {}).get("edges", [])
                
                if not edges:
                    return jsonify({"error": "No edges found in transformed state"}), 500
                
                # Update edge colors based on current firing states
                updated_edges = self.transformer.update_edge_colors(edges, neurons)
                
                if not updated_edges:
                    return jsonify({"error": "Failed to update edge colors"}), 500
                
                return jsonify({"edges": updated_edges})
                
            except Exception as e:
                print(f"Error in get_edge_colors: {e}")
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/layout/<layout_name>")
        def get_layout_config(layout_name: str):
            """Get layout configuration for Cytoscape.js."""
            try:
                layout_config = self.transformer.get_layout_config(layout_name)
                return jsonify(layout_config)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/signal", methods=["POST"])
        def send_signal():
            """Send a signal to the network."""
            try:
                data = request.get_json()
                neuron_id = data.get("neuron_id")
                synapse_id = data.get("synapse_id")
                strength = data.get("strength", 1.0)

                if neuron_id is None or synapse_id is None:
                    return (
                        jsonify({"error": "neuron_id and synapse_id are required"}),
                        400,
                    )

                success = self.nn_core.send_signal(neuron_id, synapse_id, strength)
                return jsonify({"success": success})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/tick", methods=["POST"])
        def execute_tick():
            """Execute a single network tick."""
            try:
                result = self.nn_core.do_tick()
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/ticks", methods=["POST"])
        def execute_n_ticks():
            """Execute multiple network ticks."""
            try:
                data = request.get_json()
                n_ticks = data.get("n_ticks", 1)

                if n_ticks <= 0:
                    return jsonify({"error": "n_ticks must be positive"}), 400

                results = self.nn_core.do_n_ticks(n_ticks)
                return jsonify({"results": results, "count": len(results)})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/start", methods=["POST"])
        def start_time_flow():
            """Start autonomous time flow."""
            try:
                data = request.get_json()
                tick_rate = data.get("tick_rate", 1.0)

                success = self.nn_core.start_time_flow(tick_rate)
                return jsonify({"success": success, "tick_rate": tick_rate})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/stop", methods=["POST"])
        def stop_time_flow():
            """Stop autonomous time flow."""
            try:
                success = self.nn_core.stop_time_flow()
                return jsonify({"success": success})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/neuron/<int:neuron_id>")
        def get_neuron_details(neuron_id: int):
            """Get detailed information about a specific neuron."""
            try:
                neuron_data = self.nn_core.get_neuron(neuron_id)
                if not neuron_data:
                    return jsonify({"error": "Neuron not found"}), 404
                return jsonify(neuron_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({"error": "Endpoint not found"}), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({"error": "Internal server error"}), 500

    def _start_websocket_server(self):
        """Start the WebSocket server in a separate thread."""
        def websocket_server():
            async def ws_handler(websocket):
                """Handle WebSocket connections."""
                client_id = id(websocket)
                self.websocket_clients.add(websocket)
                print(f"WebSocket client connected: {client_id}")
                
                try:
                    # Send connection confirmation
                    await websocket.send(json.dumps({
                        "type": "connected",
                        "status": "connected",
                        "client_id": client_id
                    }))
                    
                    # Send initial network state
                    try:
                        raw_state = self.nn_core.get_network_state()
                        transformed_state = self.transformer.transform_network_state(raw_state)
                        await websocket.send(json.dumps({
                            "type": "network_state",
                            "data": transformed_state
                        }))
                    except Exception as e:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))
                    
                    # Start update thread if this is the first client
                    if len(self.websocket_clients) == 1:
                        self._start_update_thread()

                    
                    # Handle incoming messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(websocket, data)
                        except json.JSONDecodeError:
                            print(f"Invalid JSON from client {client_id}")
                        except Exception as e:
                            print(f"Error handling message from client {client_id}: {e}")
                            
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.discard(websocket)
                    print(f"WebSocket client disconnected: {client_id}")
                    
                    # Stop update thread if no clients are connected
                    if len(self.websocket_clients) == 0:
                        self._stop_update_thread()
            
            # Start WebSocket server
            async def start_websocket():
                server = await websockets.serve(ws_handler, "127.0.0.1", 5556)
                await server.wait_closed()
            
            # Run the WebSocket server
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Store the loop reference for the update thread to use
            self.websocket_loop = loop
            loop.run_until_complete(start_websocket())
        
        # Start WebSocket server in a separate thread
        self.websocket_thread = threading.Thread(target=websocket_server, daemon=True)
        self.websocket_thread.start()
        print("WebSocket server started on ws://127.0.0.1:5556")

    async def _handle_websocket_message(self, websocket, data):
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        
        if message_type == "get_network_state":
            try:
                raw_state = self.nn_core.get_network_state()
                transformed_state = self.transformer.transform_network_state(raw_state)
                await websocket.send(json.dumps({
                    "type": "network_state",
                    "data": transformed_state
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
        
        elif message_type == "send_signal":
            try:
                neuron_id = data.get("neuron_id")
                synapse_id = data.get("synapse_id")
                strength = data.get("strength", 1.0)

                if neuron_id is None or synapse_id is None:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "neuron_id and synapse_id are required"
                    }))
                    return

                success = self.nn_core.send_signal(neuron_id, synapse_id, strength)
                await websocket.send(json.dumps({
                    "type": "signal_result",
                    "success": success,
                    "strength": strength
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
        
        elif message_type == "execute_tick":
            try:
                result = self.nn_core.do_tick()
                await websocket.send(json.dumps({
                    "type": "tick_result",
                    "result": result
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))

    def _start_update_thread(self):
        """Start the real-time update thread."""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            print("Started real-time update thread")

    def _stop_update_thread(self):
        """Stop the real-time update thread."""
        self.update_thread_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        print("Stopped real-time update thread")

    def _update_loop(self):
        """Main update loop for real-time data broadcasting."""
        last_state = None

        while self.update_thread_running and self.websocket_clients:
            try:
                # Get current network state
                raw_state = self.nn_core.get_network_state()
                current_tick = raw_state.get("core_state", {}).get("current_tick", 0)

                # Only send updates if state has changed
                if (
                    last_state is None
                    or last_state.get("core_state", {}).get("current_tick", -1)
                    != current_tick
                ):
                    transformed_state = self.transformer.transform_network_state(
                        raw_state
                    )

                    # Broadcast to all connected WebSocket clients
                    if self.websocket_clients:
                        message = json.dumps({
                            "type": "network_update",
                            "data": transformed_state
                        })
                        # Send to all connected clients
                        disconnected_clients = set()
                        for client in list(self.websocket_clients):  # Copy to avoid modification during iteration
                            try:
                                # Get the event loop from the WebSocket server thread
                                if hasattr(self, 'websocket_loop') and self.websocket_loop:
                                    future = asyncio.run_coroutine_threadsafe(
                                        client.send(message), self.websocket_loop
                                    )
                                    future.result(timeout=1.0)  # Wait up to 1 second
                                else:
                                    # Fallback: try to send directly (may not work)
                                    print("Warning: No WebSocket event loop available")
                            except websockets.exceptions.ConnectionClosed:
                                disconnected_clients.add(client)
                            except Exception as e:
                                print(f"Error sending to client: {e}")
                                disconnected_clients.add(client)
                        
                        # Remove disconnected clients
                        self.websocket_clients -= disconnected_clients
                        
                        # Stop update thread if no clients are connected
                        if not self.websocket_clients:
                            self._stop_update_thread()
                    
                    last_state = raw_state

                # Sleep for the configured interval
                time.sleep(self.update_interval)

            except Exception as e:
                print(f"Error in update loop: {e}")
                time.sleep(self.update_interval)

    def run(self, threaded: bool = True):
        """Run the web server."""
        print(f"Starting Neural Network Web Visualization Server...")
        print(f"Server will be available at: http://{self.host}:{self.port}")
        print(f"Press Ctrl+C to stop the server")

        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,  # Disable reloader to prevent issues
                threaded=True,
            )
        except KeyboardInterrupt:
            print("\nShutting down server...")
            self._stop_update_thread()
        except Exception as e:
            print(f"Server error: {e}")
            self._stop_update_thread()

    def stop(self):
        """Stop the web server."""
        self._stop_update_thread()
        # Note: Flask-SocketIO doesn't have a direct stop method
        # The server will stop when the main thread exits

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}",
            "connected_clients": len(self.websocket_clients),
            "update_thread_running": self.update_thread_running,
            "debug": self.debug,
        }
