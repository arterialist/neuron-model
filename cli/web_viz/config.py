#!/usr/bin/env python3
"""
Configuration settings for the web visualization server.
"""

import os
from typing import Dict, Any


class WebVizConfig:
    """Configuration class for web visualization settings."""

    def __init__(self):
        # Server settings
        self.host = os.getenv("WEBVIZ_HOST", "127.0.0.1")
        self.port = int(os.getenv("WEBVIZ_PORT", "5555"))
        self.debug = os.getenv("WEBVIZ_DEBUG", "false").lower() == "true"

        # Update settings
        self.update_interval = float(
            os.getenv("WEBVIZ_UPDATE_INTERVAL", "0.1")
        )  # 100ms
        self.max_update_rate = float(
            os.getenv("WEBVIZ_MAX_UPDATE_RATE", "30.0")
        )  # 30 FPS

        # WebSocket settings
        self.websocket_timeout = int(os.getenv("WEBVIZ_WS_TIMEOUT", "60"))  # seconds
        self.websocket_ping_interval = int(
            os.getenv("WEBVIZ_WS_PING_INTERVAL", "25")
        )  # seconds

        # Visualization settings
        self.default_layout = os.getenv("WEBVIZ_DEFAULT_LAYOUT", "cose")
        self.max_nodes_for_animation = int(
            os.getenv("WEBVIZ_MAX_NODES_ANIMATION", "100")
        )
        self.signal_animation_duration = int(
            os.getenv("WEBVIZ_SIGNAL_DURATION", "2000")
        )  # ms

        # Performance settings
        self.enable_compression = (
            os.getenv("WEBVIZ_COMPRESSION", "true").lower() == "true"
        )
        self.max_traveling_signals = int(os.getenv("WEBVIZ_MAX_SIGNALS", "50"))

        # Security settings
        self.cors_origins = os.getenv("WEBVIZ_CORS_ORIGINS", "*").split(",")
        self.secret_key = os.getenv(
            "WEBVIZ_SECRET_KEY", "neural_network_viz_secret_key"
        )

    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "cors_origins": self.cors_origins,
            "secret_key": self.secret_key,
        }

    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration dictionary."""
        return {
            "timeout": self.websocket_timeout,
            "ping_interval": self.websocket_ping_interval,
            "cors_allowed_origins": self.cors_origins,
        }

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration dictionary."""
        return {
            "default_layout": self.default_layout,
            "max_nodes_for_animation": self.max_nodes_for_animation,
            "signal_animation_duration": self.signal_animation_duration,
            "max_traveling_signals": self.max_traveling_signals,
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration dictionary."""
        return {
            "update_interval": self.update_interval,
            "max_update_rate": self.max_update_rate,
            "enable_compression": self.enable_compression,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert all configuration to dictionary."""
        return {
            "server": self.get_server_config(),
            "websocket": self.get_websocket_config(),
            "visualization": self.get_visualization_config(),
            "performance": self.get_performance_config(),
        }

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if "server" in config_dict:
            server_config = config_dict["server"]
            self.host = server_config.get("host", self.host)
            self.port = server_config.get("port", self.port)
            self.debug = server_config.get("debug", self.debug)

        if "websocket" in config_dict:
            ws_config = config_dict["websocket"]
            self.websocket_timeout = ws_config.get("timeout", self.websocket_timeout)
            self.websocket_ping_interval = ws_config.get(
                "ping_interval", self.websocket_ping_interval
            )

        if "visualization" in config_dict:
            viz_config = config_dict["visualization"]
            self.default_layout = viz_config.get("default_layout", self.default_layout)
            self.max_nodes_for_animation = viz_config.get(
                "max_nodes_for_animation", self.max_nodes_for_animation
            )
            self.signal_animation_duration = viz_config.get(
                "signal_animation_duration", self.signal_animation_duration
            )

        if "performance" in config_dict:
            perf_config = config_dict["performance"]
            self.update_interval = perf_config.get(
                "update_interval", self.update_interval
            )
            self.max_update_rate = perf_config.get(
                "max_update_rate", self.max_update_rate
            )


# Global configuration instance
default_config = WebVizConfig()
