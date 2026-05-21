# Neural Network Web Visualization

Interactive browser visualization for `NNCore`, rebuilt around the same
React/Vite + canvas rendering pattern used by the newer C. elegans lab UI.

## What It Provides

- Canvas-rendered network map with layers, grid, circle, and concentric layouts
- Real-time updates over WebSocket with REST fallbacks for commands
- Tick, multi-tick, auto start/stop, and external signal controls
- Node selection, activity-aware coloring, incident-edge highlighting, and details
- Live attractor landscape and dynamical-regime view for all, selected, or
  layer-scoped neuron populations
- Dynamic K-means clustering for all neurons or a selected layer, with 2D/3D
  parameter selectors, unique-axis enforcement, and automatic cluster count
  selection
- Labelled stimulus metadata and analysis views for network activity,
  label-conditioned responses, assemblies, synapses, spike rasters, and phase
  portraits
- Global and per-layer free-energy graphs
- Runtime statistics for tick, activity, potentials, density, FPS, nodes, and edges

## Runtime Architecture

```
NNCore
  └─ NeuralNetworkWebServer
      ├─ FastAPI REST: /api/network/*, /api/neuron/*, /api/config
      ├─ FastAPI WebSocket: /ws
      └─ Static app: dist/ built from webapp/

webapp/
  └─ React + TypeScript + Vite
      └─ CanvasNetwork.tsx: requestAnimationFrame renderer
```

The old browser stack has been removed. There are no legacy DOM modules or
third-party graph-renderer bundles in this package.

## Build

From this directory:

```bash
cd cli/web_viz/webapp
npm install
npm run build
```

The production bundle is written to `cli/web_viz/dist/` and served by FastAPI.

For frontend-only development, run:

```bash
cd cli/web_viz/webapp
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:5555`.

## Usage

From the interactive CLI:

```text
> web_viz
```

The command starts one uvicorn process on `WEBVIZ_HOST` / `WEBVIZ_PORT`; REST,
static files, and WebSocket streaming share that port.

The historical launch paths (`interactive_training.py`, `interactive_mnist.py`,
`snn_classification_realtime/realtime_classifier/run.py`, and
`snn_classification_realtime/activity_dataset_builder/build.py`) all construct
the same `NeuralNetworkWebServer` and honor the same `WEBVIZ_*` environment
variables. If another local app already owns port 5555, start with:

```bash
WEBVIZ_PORT=5665 python cli/neuron_cli.py
```

## API Surface

- `GET /api/config`
- `GET /api/network/state`
- `GET /api/network/style`
- `GET /api/network/layout/<name>`
- `GET /api/network/edge-colors`
- `POST /api/network/signal`
- `POST /api/network/tick`
- `POST /api/network/ticks`
- `POST /api/network/start`
- `POST /api/network/stop`
- `GET /api/stimulus/metadata`
- `POST /api/stimulus/metadata`
- `DELETE /api/stimulus/metadata`
- `GET /api/neuron/<id>`

## Configuration

```bash
export WEBVIZ_HOST=127.0.0.1
export WEBVIZ_PORT=5555
export WEBVIZ_WS_PATH=/ws
export WEBVIZ_DEBUG=false
export WEBVIZ_UPDATE_INTERVAL=0.1
export WEBVIZ_CORS_ORIGINS=*
export WEBVIZ_SECRET_KEY=your_secret_key
```

## Keyboard Shortcuts

- `Space`: single tick
- `Enter`: run the configured tick count
- `S`: start auto mode
- `X`: stop auto mode
- `F`: fit view

## Dynamics View

The dynamics panel keeps a bounded browser-side history from the WebSocket
stream and projects each population into user-selected 2D or 3D metric axes.
Available metrics include activity, firing rate, membrane potential, free
energy, reference time (`t_ref`), prediction, precision, and other neuron
parameters exposed by the runtime. The heat field is dwell time in the projected
state space; the trajectory and regime label update for all neurons, the
selected neuron, or any metadata layer.

## Clustering View

The clustering tab runs browser-side K-means continuously over the rolling
network state. Scope can be all neurons or one metadata layer. Axes can be 2D or
3D, parameter selections are kept unique, and auto mode estimates the cluster
count from the current population before updating quality metrics.

## Live Analysis View

The analysis tab uses the same rolling WebSocket history for live network
activity, labelled stimulus activity, neuron assemblies, and synaptic
connectivity views. Label-aware plots are enabled by setting optional stimulus
metadata with `POST /api/stimulus/metadata`; unlabelled networks still render
the generic activity, synchrony, and connectivity views.

## Stop Semantics

`POST /api/network/stop` pauses automatic ticks and also marks externally driven
simulation loops as paused. Manual `tick` and `ticks` commands still force
explicit steps so a paused app remains inspectable.
