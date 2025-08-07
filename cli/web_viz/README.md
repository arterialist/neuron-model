# Neural Network Web Visualization

A modern, interactive web-based visualization system for neural networks using Cytoscape.js.

## Features

- **Interactive Graph Visualization**: Powered by Cytoscape.js with support for various layouts
- **Real-time Updates**: WebSocket-based live updates of network state
- **Node Activity Visualization**: Color-coded neurons based on activity levels
- **Traveling Signal Animation**: Animated signals moving between neurons
- **Interactive Controls**: Send signals, execute ticks, and control simulation
- **Detailed Node Information**: Click on neurons to see detailed properties
- **Multiple Layout Options**: Force-directed, grid, circle, and concentric layouts
- **Responsive Design**: Works on desktop and tablet devices

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI System    │    │   Web Server     │    │   Frontend      │
│                 │    │                  │    │                 │
│  ┌───────────┐  │    │  ┌─────────────┐ │    │  ┌────────────┐ │
│  │  NNCore   │◄─┼────┼─►│ REST API    │ │    │  │ Cytoscape  │ │
│  └───────────┘  │    │  └─────────────┘ │    │  │    .js     │ │
│                 │    │                  │    │  └────────────┘ │
│  ┌───────────┐  │    │  ┌─────────────┐ │    │  ┌────────────┐ │
│  │ Commands  │  │    │  │ WebSocket   │◄┼────┼─►│ Real-time  │ │
│  └───────────┘  │    │  │   Server    │ │    │  │  Updates   │ │
└─────────────────┘    │  └─────────────┘ │    │  └────────────┘ │
                       │                  │    │                 │
                       │  ┌─────────────┐ │    │  ┌────────────┐ │
                       │  │ Data        │ │    │  │ Animation  │ │
                       │  │ Transformer │ │    │  │   System   │ │
                       │  └─────────────┘ │    │  └────────────┘ │
                       └──────────────────┘    └─────────────────┘
```

## Components

### Backend Components

1. **NeuralNetworkWebServer** (`server.py`)
   - Flask-based web server with REST API endpoints
   - WebSocket support for real-time updates
   - Handles client connections and data broadcasting

2. **NetworkDataTransformer** (`data_transformer.py`)
   - Converts neural network state to Cytoscape.js format
   - Handles node styling based on activity levels
   - Processes traveling signals for animation

3. **WebVizConfig** (`config.py`)
   - Configuration management for server settings
   - Environment variable support
   - Performance and visualization options

### Frontend Components

1. **Main Application** (`main.js`)
   - Application initialization and coordination
   - Event handling and error management
   - Loading states and user feedback

2. **Network Visualization** (`network_viz.js`)
   - Cytoscape.js integration and management
   - Layout application and node/edge rendering
   - Export and view manipulation functions

3. **Data Manager** (`data_manager.js`)
   - API communication and state management
   - Network state caching and updates
   - Error handling and retry logic

4. **Animation System** (`animations.js`)
   - Traveling signal animations
   - Node flashing and edge highlighting
   - Smooth transitions and effects

5. **Interaction Handler** (`interactions.js`)
   - User input processing
   - Control panel management
   - Node selection and details display

6. **WebSocket Client** (`websocket_client.js`)
   - Real-time communication with server
   - Connection management and reconnection
   - Event broadcasting to application

## Usage

### Starting the Web Visualization

From the CLI:
```bash
> web_viz
```

This will:
1. Start the web server on `http://localhost:5000`
2. Open your default browser to the visualization
3. Begin real-time updates of the network state

### Available Commands

- `web_viz` - Launch web visualization server
- `web_status` - Show server status and connected clients
- `web_stop` - Stop the web visualization server

### Web Interface Features

#### Control Panel
- **Simulation Controls**: Execute ticks, start/stop auto mode
- **Signal Controls**: Send signals to specific neurons
- **Layout Controls**: Change graph layout and fit view
- **Statistics Display**: Real-time network metrics

#### Visualization Area
- **Interactive Graph**: Zoom, pan, and select nodes
- **Node Colors**:
  - 🔴 Red: Firing neurons (output > 0)
  - 🟠 Orange: High potential (> 0.5)
  - 🟡 Yellow: Some potential (> 0)
  - 🔵 Blue: Inactive neurons
  - 🟢 Green squares: External inputs
- **Traveling Signals**:
  - 🟠 Orange dots: Presynaptic events
  - 🟣 Purple dots: Retrograde events

#### Keyboard Shortcuts
- `Space` - Execute single tick
- `Enter` - Execute multiple ticks
- `S` - Start auto mode
- `X` - Stop auto mode
- `F` - Fit view to network
- `Escape` - Close details panel

## Configuration

### Environment Variables

```bash
# Server settings
export WEBVIZ_HOST=127.0.0.1
export WEBVIZ_PORT=5000
export WEBVIZ_DEBUG=false

# Performance settings
export WEBVIZ_UPDATE_INTERVAL=0.1
export WEBVIZ_MAX_UPDATE_RATE=30.0

# Visualization settings
export WEBVIZ_DEFAULT_LAYOUT=cose
export WEBVIZ_MAX_NODES_ANIMATION=100
export WEBVIZ_SIGNAL_DURATION=2000

# Security settings
export WEBVIZ_CORS_ORIGINS=*
export WEBVIZ_SECRET_KEY=your_secret_key
```

### Layout Options

1. **COSE (Force-directed)** - Default, good for most networks
2. **Grid** - Organized grid layout
3. **Circle** - Circular arrangement
4. **Concentric** - Concentric circles based on node degree

## API Endpoints

### REST API

- `GET /api/network/state` - Get current network state
- `GET /api/network/style` - Get Cytoscape.js style definition
- `GET /api/network/layout/<name>` - Get layout configuration
- `POST /api/network/signal` - Send signal to network
- `POST /api/network/tick` - Execute single tick
- `POST /api/network/ticks` - Execute multiple ticks
- `POST /api/network/start` - Start autonomous time flow
- `POST /api/network/stop` - Stop autonomous time flow
- `GET /api/neuron/<id>` - Get detailed neuron information

### WebSocket Events

**Client → Server:**
- `get_network_state` - Request current state
- `send_signal` - Send signal to neuron
- `execute_tick` - Execute single tick

**Server → Client:**
- `network_state` - Initial network state
- `network_update` - Real-time state updates
- `signal_result` - Signal sending result
- `tick_result` - Tick execution result
- `error` - Error messages

## Performance Considerations

- **Large Networks**: Automatic level-of-detail rendering for networks with many nodes
- **Update Rate**: Configurable update intervals to balance responsiveness and performance
- **Animation Limits**: Maximum number of traveling signals to prevent UI overload
- **WebGL Rendering**: Hardware acceleration for smooth interactions

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Troubleshooting

### Common Issues

1. **Server won't start**: Check if port is already in use
2. **WebSocket connection fails**: Verify firewall settings
3. **Slow performance**: Reduce update rate or disable animations for large networks
4. **Browser compatibility**: Use a modern browser with WebGL support

### Debug Mode

Enable debug mode for detailed logging:
```bash
export WEBVIZ_DEBUG=true
```

### Testing

Run the test suite:
```bash
python cli/test_web_viz.py
```

## Development

### Adding New Features

1. **Backend**: Add endpoints to `server.py` and update `data_transformer.py`
2. **Frontend**: Add JavaScript modules and update `main.js`
3. **Styling**: Modify `styles.css` and Cytoscape.js styles
4. **Configuration**: Update `config.py` for new settings

### File Structure

```
cli/web_viz/
├── __init__.py              # Package initialization
├── server.py                # Flask web server
├── data_transformer.py      # Data transformation layer
├── config.py               # Configuration management
├── static/
│   ├── css/
│   │   └── styles.css      # Main stylesheet
│   └── js/
│       ├── main.js         # Application entry point
│       ├── network_viz.js  # Cytoscape.js integration
│       ├── data_manager.js # API communication
│       ├── animations.js   # Animation system
│       ├── interactions.js # User interactions
│       └── websocket_client.js # WebSocket client
├── templates/
│   └── index.html          # Main HTML template
└── README.md               # This file
```