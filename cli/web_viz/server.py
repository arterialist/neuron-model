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
from flask_socketio import SocketIO, emit, disconnect
import logging

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

        # Initialize SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode="threading",
            logger=False,
            engineio_logger=False,
        )

        # Data transformer
        self.transformer = NetworkDataTransformer()

        # Connected clients tracking
        self.connected_clients: Set[str] = set()

        # Update thread control
        self.update_thread = None
        self.update_thread_running = False
        self.update_interval = 0.1  # 100ms updates

        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()

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

        @self.app.route("/api/network/state")
        def get_network_state():
            """Get current network state in Cytoscape.js format."""
            try:
                raw_state = self.nn_core.get_network_state()
                transformed_state = self.transformer.transform_network_state(raw_state)
                return jsonify(transformed_state)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/style")
        def get_network_style():
            """Get Cytoscape.js style definition."""
            try:
                style = self.transformer.get_cytoscape_style()
                return jsonify(style)
            except Exception as e:
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

    def _setup_socket_handlers(self):
        """Setup SocketIO event handlers."""

        @self.socketio.on("connect")
        def handle_connect():
            """Handle client connection."""
            from flask import request as flask_request

            client_id = flask_request.sid  # type: ignore
            self.connected_clients.add(client_id)
            print(
                f"Client {client_id} connected. Total clients: {len(self.connected_clients)}"
            )

            # Send initial network state
            try:
                raw_state = self.nn_core.get_network_state()
                transformed_state = self.transformer.transform_network_state(raw_state)
                emit("network_state", transformed_state)
            except Exception as e:
                emit("error", {"message": str(e)})

            # Start update thread if this is the first client
            if len(self.connected_clients) == 1:
                self._start_update_thread()

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            from flask import request as flask_request

            client_id = flask_request.sid  # type: ignore
            self.connected_clients.discard(client_id)
            print(
                f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}"
            )

            # Stop update thread if no clients are connected
            if len(self.connected_clients) == 0:
                self._stop_update_thread()

        @self.socketio.on("get_network_state")
        def handle_get_network_state():
            """Handle request for current network state."""
            try:
                raw_state = self.nn_core.get_network_state()
                transformed_state = self.transformer.transform_network_state(raw_state)
                emit("network_state", transformed_state)
            except Exception as e:
                emit("error", {"message": str(e)})

        @self.socketio.on("send_signal")
        def handle_send_signal(data):
            """Handle signal sending request via WebSocket."""
            try:
                neuron_id = data.get("neuron_id")
                synapse_id = data.get("synapse_id")
                strength = data.get("strength", 1.0)

                if neuron_id is None or synapse_id is None:
                    emit("error", {"message": "neuron_id and synapse_id are required"})
                    return

                success = self.nn_core.send_signal(neuron_id, synapse_id, strength)
                emit(
                    "signal_result",
                    {
                        "success": success,
                        "neuron_id": neuron_id,
                        "synapse_id": synapse_id,
                        "strength": strength,
                    },
                )
            except Exception as e:
                emit("error", {"message": str(e)})

        @self.socketio.on("execute_tick")
        def handle_execute_tick():
            """Handle tick execution request via WebSocket."""
            try:
                result = self.nn_core.do_tick()
                emit("tick_result", result)
            except Exception as e:
                emit("error", {"message": str(e)})

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

        while self.update_thread_running and self.connected_clients:
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

                    # Broadcast to all connected clients
                    self.socketio.emit("network_update", transformed_state)
                    last_state = raw_state

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
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,  # Disable reloader to prevent issues
                allow_unsafe_werkzeug=True,
            )  # Allow for development
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
            "connected_clients": len(self.connected_clients),
            "update_thread_running": self.update_thread_running,
            "debug": self.debug,
        }
