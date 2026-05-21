#!/usr/bin/env python3
"""FastAPI server for the high-performance network web viewer."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time
import traceback
from typing import Any, Dict, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn

from .data_transformer import NetworkDataTransformer


class SignalBody(BaseModel):
    neuron_id: int
    synapse_id: int
    strength: float = 1.0


class TicksBody(BaseModel):
    n_ticks: int = 1


class StartBody(BaseModel):
    tick_rate: float = 1.0


class StimulusMetadataBody(BaseModel):
    label: Any | None = None
    class_name: str | None = None
    presentation_id: str | int | None = None
    sample_index: int | None = None
    dataset_name: str | None = None
    epoch: int | float | None = None
    source: str | None = None
    predicted_label: Any | None = None
    confidence: float | None = None
    second_predicted_label: Any | None = None
    second_confidence: float | None = None
    third_predicted_label: Any | None = None
    third_confidence: float | None = None
    tags: list[str] | None = None
    extra: Dict[str, Any] | None = None

    class Config:
        extra = "allow"


class NeuralNetworkWebServer:
    """Web server for neural network visualization."""

    def __init__(
        self, nn_core, host: str = "127.0.0.1", port: int = 5555, debug: bool = False
    ):
        self.nn_core = nn_core
        self.host = host
        self.port = port
        self.debug = debug
        self.root_dir = Path(__file__).resolve().parent
        self.dist_dir = self.root_dir / "dist"

        from .config import WebVizConfig

        config = WebVizConfig()
        self.update_interval = config.update_interval
        self.min_update_interval = config.min_update_interval
        self.websocket_path = config.websocket_path

        self.transformer = NetworkDataTransformer()
        self.websocket_clients: Set[WebSocket] = set()
        self.uvicorn_server: uvicorn.Server | None = None
        self.stimulus_metadata: Dict[str, Any] = {}
        self.stimulus_sequence = 0

        self.app = FastAPI(title="Neuron Model Web Visualization")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=False,
        )
        self._setup_routes()

        if not debug:
            logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    def _setup_routes(self) -> None:
        @self.app.get("/api/test")
        async def test_endpoint() -> Dict[str, Any]:
            return {
                "status": "ok",
                "message": "FastAPI server is working",
                "renderer": "canvas",
            }

        @self.app.get("/api/config")
        async def get_client_config(request: Request) -> Dict[str, Any]:
            return {
                "renderer": "canvas",
                "websocket_url": self._websocket_url(request),
                "update_interval": self.update_interval,
            }

        @self.app.get("/api/network/state")
        async def get_network_state() -> Dict[str, Any]:
            try:
                return self._transformed_network_state()
            except Exception as e:
                logging.error("Error in get_network_state: %s", e)
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/api/network/style")
        async def get_network_style() -> Dict[str, Any]:
            return self.transformer.get_canvas_style()

        @self.app.get("/api/network/edge-colors")
        async def get_edge_colors() -> Dict[str, Any]:
            try:
                raw_state = self._raw_network_state()
                neurons = raw_state.get("network", {}).get("neurons", {})
                transformed_state = self.transformer.transform_network_state(raw_state)
                edges = transformed_state.get("elements", {}).get("edges", [])
                return {"edges": self.transformer.update_edge_colors(edges, neurons)}
            except Exception as e:
                logging.error("Error in get_edge_colors: %s", e)
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/api/network/layout/{layout_name}")
        async def get_layout_config(layout_name: str) -> Dict[str, Any]:
            return self.transformer.get_layout_config(layout_name)

        @self.app.post("/api/network/signal")
        async def send_signal(body: SignalBody) -> Dict[str, Any]:
            try:
                success = self.nn_core.send_signal(
                    body.neuron_id, body.synapse_id, body.strength
                )
                return {
                    "success": success,
                    "neuron_id": body.neuron_id,
                    "synapse_id": body.synapse_id,
                    "strength": body.strength,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.post("/api/network/tick")
        async def execute_tick() -> Dict[str, Any]:
            try:
                return self.nn_core.do_tick(force=True)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.post("/api/network/ticks")
        async def execute_n_ticks(body: TicksBody) -> Dict[str, Any]:
            try:
                if body.n_ticks <= 0:
                    raise HTTPException(
                        status_code=400, detail="n_ticks must be positive"
                    )
                results = self.nn_core.do_n_ticks(body.n_ticks, force=True)
                return {"results": results, "count": len(results)}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.post("/api/network/start")
        async def start_time_flow(body: StartBody) -> Dict[str, Any]:
            try:
                success = self.nn_core.start_time_flow(body.tick_rate)
                return {
                    "success": success,
                    "tick_rate": body.tick_rate,
                    "is_paused": self.nn_core.state.is_paused,
                    "external_driver_active": self.nn_core.state.external_driver_active,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.post("/api/network/stop")
        async def stop_time_flow() -> Dict[str, Any]:
            try:
                success = self.nn_core.stop_time_flow()
                return {
                    "success": success,
                    "is_paused": self.nn_core.state.is_paused,
                    "external_driver_active": self.nn_core.state.external_driver_active,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/api/stimulus/metadata")
        async def get_stimulus_metadata() -> Dict[str, Any]:
            return self._stimulus_snapshot()

        @self.app.post("/api/stimulus/metadata")
        async def set_stimulus_metadata(
            body: StimulusMetadataBody,
        ) -> Dict[str, Any]:
            payload = self._metadata_body_to_dict(body)
            self.stimulus_sequence += 1
            self.stimulus_metadata = payload
            return self._stimulus_snapshot()

        @self.app.delete("/api/stimulus/metadata")
        async def clear_stimulus_metadata() -> Dict[str, Any]:
            self.stimulus_sequence += 1
            self.stimulus_metadata = {}
            return self._stimulus_snapshot()

        @self.app.get("/api/neuron/{neuron_id}")
        async def get_neuron_details(neuron_id: int) -> Dict[str, Any]:
            try:
                neuron_data = self.nn_core.get_neuron(neuron_id)
                if not neuron_data:
                    raise HTTPException(status_code=404, detail="Neuron not found")
                return self.transformer._convert_numpy_types(neuron_data)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.websocket(self.websocket_path)
        async def network_websocket(ws: WebSocket) -> None:
            await self._handle_websocket(ws)

        @self.app.get("/")
        async def index():
            return self._serve_frontend("index.html")

        @self.app.head("/")
        async def head_index():
            return self._serve_frontend("index.html")

        @self.app.get("/{path:path}")
        async def frontend_asset_or_spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Endpoint not found")
            candidate = self.dist_dir / path
            if candidate.is_file():
                return FileResponse(candidate)
            return self._serve_frontend("index.html")

        @self.app.head("/{path:path}")
        async def head_frontend_asset_or_spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Endpoint not found")
            candidate = self.dist_dir / path
            if candidate.is_file():
                return FileResponse(candidate)
            return self._serve_frontend("index.html")

    def _serve_frontend(self, filename: str):
        index = self.dist_dir / "index.html"
        if not index.is_file():
            return PlainTextResponse(
                "web_viz frontend is not built. Run `npm install` and "
                "`npm run build` from cli/web_viz/webapp.",
                status_code=503,
            )
        return FileResponse(self.dist_dir / filename)

    def _websocket_url(self, request: Request) -> str:
        scheme = "wss" if request.url.scheme == "https" else "ws"
        forwarded_proto = request.headers.get("x-forwarded-proto")
        if forwarded_proto:
            scheme = "wss" if forwarded_proto == "https" else "ws"
        host = request.headers.get("host") or f"{self.host}:{self.port}"
        return f"{scheme}://{host}{self.websocket_path}"

    async def _handle_websocket(self, ws: WebSocket) -> None:
        await ws.accept()
        self.websocket_clients.add(ws)
        last_tick = None
        try:
            await ws.send_json(
                {"type": "connected", "status": "connected", "client_id": id(ws)}
            )
            initial_state = self._transformed_network_state()
            await ws.send_json({"type": "network_state", "data": initial_state})
            last_tick = initial_state.get("current_tick", 0)

            while True:
                try:
                    message = await asyncio.wait_for(
                        ws.receive_json(), timeout=self.update_interval
                    )
                    await self._handle_websocket_message(ws, message)
                    state = self._transformed_network_state()
                    await ws.send_json({"type": "network_update", "data": state})
                    last_tick = state.get("current_tick", last_tick)
                except asyncio.TimeoutError:
                    state = self._transformed_network_state()
                    current_tick = state.get("current_tick", 0)
                    if last_tick != current_tick:
                        await ws.send_json({"type": "network_update", "data": state})
                        last_tick = current_tick
        except WebSocketDisconnect:
            return
        except Exception:
            logging.exception("WebSocket handler crashed")
        finally:
            self.websocket_clients.discard(ws)

    async def _handle_websocket_message(
        self, websocket: WebSocket, data: Dict[str, Any]
    ) -> None:
        message_type = data.get("type")

        if message_type == "get_network_state":
            await websocket.send_json(
                {"type": "network_state", "data": self._transformed_network_state()}
            )
            return

        if message_type == "send_signal":
            neuron_id = data.get("neuron_id")
            synapse_id = data.get("synapse_id")
            strength = data.get("strength", 1.0)
            if neuron_id is None or synapse_id is None:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "neuron_id and synapse_id are required",
                    }
                )
                return
            success = self.nn_core.send_signal(neuron_id, synapse_id, strength)
            await websocket.send_json(
                {
                    "type": "signal_result",
                    "success": success,
                    "neuron_id": neuron_id,
                    "synapse_id": synapse_id,
                    "strength": strength,
                }
            )
            return

        if message_type == "execute_tick":
            await websocket.send_json(
                {"type": "tick_result", "result": self.nn_core.do_tick(force=True)}
            )
            return

        if message_type == "set_stimulus_metadata":
            payload = data.get("metadata")
            if isinstance(payload, dict):
                self.stimulus_sequence += 1
                self.stimulus_metadata = self._clean_metadata(payload)
                await websocket.send_json(
                    {"type": "stimulus_metadata", "data": self._stimulus_snapshot()}
                )
            else:
                await websocket.send_json(
                    {"type": "error", "message": "metadata object is required"}
                )
            return

        if message_type == "clear_stimulus_metadata":
            self.stimulus_sequence += 1
            self.stimulus_metadata = {}
            await websocket.send_json(
                {"type": "stimulus_metadata", "data": self._stimulus_snapshot()}
            )
            return

    def _raw_network_state(self) -> Dict[str, Any]:
        if not hasattr(self, "nn_core") or self.nn_core is None:
            raise RuntimeError("Neural network not initialized")
        raw_state = self.nn_core.get_network_state()
        if not raw_state:
            raise RuntimeError("Failed to get network state")
        return raw_state

    def _transformed_network_state(self) -> Dict[str, Any]:
        state = self.transformer.transform_network_state(self._raw_network_state())
        state["stimulus"] = self._stimulus_snapshot(state.get("current_tick"))
        return state

    def _metadata_body_to_dict(self, body: StimulusMetadataBody) -> Dict[str, Any]:
        if hasattr(body, "model_dump"):
            payload = body.model_dump(exclude_none=True)
        else:
            payload = body.dict(exclude_none=True)
        return self._clean_metadata(payload)

    def _clean_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in payload.items():
            if value is None:
                continue
            if key == "extra" and isinstance(value, dict):
                cleaned[key] = {
                    str(extra_key): extra_value
                    for extra_key, extra_value in value.items()
                    if extra_value is not None
                }
            else:
                cleaned[str(key)] = value
        return self.transformer._convert_numpy_types(cleaned)

    def _stimulus_snapshot(self, tick: Any | None = None) -> Dict[str, Any]:
        payload = dict(self.stimulus_metadata)
        predictions = []
        for rank, label_key, confidence_key in [
            (1, "predicted_label", "confidence"),
            (2, "second_predicted_label", "second_confidence"),
            (3, "third_predicted_label", "third_confidence"),
        ]:
            if label_key in payload:
                predictions.append(
                    {
                        "rank": rank,
                        "label": payload.get(label_key),
                        "confidence": payload.get(confidence_key),
                    }
                )
        payload["active"] = bool(self.stimulus_metadata)
        payload["sequence"] = self.stimulus_sequence
        payload["updated_at_tick"] = tick
        payload["server_time_ms"] = int(time.time() * 1000)
        if predictions:
            payload["predictions"] = predictions
        return payload

    def run(self, threaded: bool = True) -> None:
        """Run the web server."""
        logging.info("Starting Neural Network Web Visualization Server...")
        logging.info("Server will be available at: http://%s:%s", self.host, self.port)
        logging.info("WebSocket updates will be available at: %s", self.websocket_path)
        logging.info("Press Ctrl+C to stop the server")

        log_level = "debug" if self.debug else "warning"
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=log_level,
            lifespan="off",
        )
        self.uvicorn_server = uvicorn.Server(config)
        self.uvicorn_server.run()

    def stop(self) -> None:
        """Stop the web server."""
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        connected_clients = len(self.websocket_clients)
        return {
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}",
            "websocket_url": f"ws://{self.host}:{self.port}{self.websocket_path}",
            "connected_clients": connected_clients,
            "update_thread_running": connected_clients > 0,
            "debug": self.debug,
        }
