import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import clsx from "clsx";
import {
  Activity,
  CircleStop,
  Maximize2,
  PanelRightClose,
  PanelRightOpen,
  Pause,
  Play,
  RotateCcw,
  Send,
  StepForward,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";
import { DynamicsPanel } from "./AttractorLandscape";
import { LiveAnalysisPanel } from "./LiveAnalysisPanel";
import { NeuronClusteringPanel } from "./NeuronClusteringPanel";
import { NeuronBottomSheet } from "./NeuronBottomSheet";
import {
  clearStimulusMetadata,
  executeTick,
  executeTicks,
  getClientConfig,
  getNetworkState,
  sendSignal,
  setStimulusMetadata,
  startNetwork,
  stopNetwork,
} from "./api";
import {
  CanvasNetwork,
  type CanvasNetworkHandle,
} from "./CanvasNetwork";
import type {
  FrameStats,
  GraphNode,
  LayoutName,
  NetworkState,
} from "./types";

const HISTORY_LIMIT = 420;

type ConnectionStatus = "connecting" | "connected" | "disconnected";
type RightPaneTab = "dynamics" | "clustering" | "analysis";

const LAYOUTS: { id: LayoutName; label: string }[] = [
  { id: "layers", label: "Layers" },
  { id: "grid", label: "Grid" },
  { id: "circle", label: "Circle" },
  { id: "concentric", label: "Concentric" },
];

export function App() {
  const [state, setState] = useState<NetworkState | null>(null);
  const [history, setHistory] = useState<NetworkState[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [layout, setLayout] = useState<LayoutName>("layers");
  const [showEdges, setShowEdges] = useState(true);
  const [edgeOpacity, setEdgeOpacity] = useState(0.16);
  const [nodeScale, setNodeScale] = useState(1);
  const [dynamicsCollapsed, setDynamicsCollapsed] = useState(false);
  const [rightPaneTab, setRightPaneTab] = useState<RightPaneTab>("dynamics");
  const [tickCount, setTickCount] = useState(10);
  const [tickRate, setTickRate] = useState(1);
  const [signalNeuron, setSignalNeuron] = useState("");
  const [signalSynapse, setSignalSynapse] = useState(0);
  const [signalStrength, setSignalStrength] = useState(1.5);
  const [stimulusLabel, setStimulusLabel] = useState("");
  const [stimulusClass, setStimulusClass] = useState("");
  const [stimulusPresentation, setStimulusPresentation] = useState("");
  const [stimulusDataset, setStimulusDataset] = useState("");
  const [stimulusEpoch, setStimulusEpoch] = useState(0);
  const [stimulusPrediction, setStimulusPrediction] = useState("");
  const [stimulusConfidence, setStimulusConfidence] = useState(0);
  const [status, setStatus] = useState("Loading network state");
  const [connection, setConnection] = useState<ConnectionStatus>("connecting");
  const [frameStats, setFrameStats] = useState<FrameStats>({
    fps: 0,
    nodes: 0,
    edges: 0,
  });
  const [throughput, setThroughput] = useState({ tps: 0, msPerTick: 0 });
  const [busy, setBusy] = useState(false);
  const canvasRef = useRef<CanvasNetworkHandle | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const lastHistoryKeyRef = useRef("");
  const pendingStateRef = useRef<NetworkState | null>(null);
  const stateFlushRafRef = useRef<number | null>(null);
  const tickTimingRef = useRef<{ tick: number; time: number; tps: number } | null>(
    null,
  );

  const selectedNode = useMemo(() => {
    if (!selectedId || !state) return null;
    return state.elements.nodes.find((node) => node.data.id === selectedId) ?? null;
  }, [selectedId, state]);

  const handleSelectNode = useCallback(
    (id: string | null) => {
      setSelectedId(id);
      if (!id) return;
      const node = state?.elements.nodes.find((candidate) => candidate.data.id === id);
      if (node?.data.neuron_id != null) {
        setSignalNeuron(String(node.data.neuron_id));
      }
    },
    [state],
  );

  const updateThroughputStats = useCallback((nextState: NetworkState) => {
    const tick = Number(nextState.current_tick);
    if (!Number.isFinite(tick)) return;

    const now = performance.now();
    const previous = tickTimingRef.current;
    if (!previous || tick <= previous.tick || now <= previous.time) {
      tickTimingRef.current = { tick, time: now, tps: previous?.tps ?? 0 };
      return;
    }

    const instantTps = ((tick - previous.tick) * 1000) / (now - previous.time);
    const tps = previous.tps > 0 ? previous.tps * 0.72 + instantTps * 0.28 : instantTps;
    tickTimingRef.current = { tick, time: now, tps };
    setThroughput({ tps, msPerTick: tps > 0 ? 1000 / tps : 0 });
  }, []);

  const resetThroughputStats = useCallback(() => {
    tickTimingRef.current = null;
    setThroughput({ tps: 0, msPerTick: 0 });
  }, []);

  const commitState = useCallback((nextState: NetworkState) => {
    setState(nextState);
    updateThroughputStats(nextState);
    const key = historyKey(nextState);
    if (key !== lastHistoryKeyRef.current) {
      lastHistoryKeyRef.current = key;
      setHistory((previous) => {
        const next = [...previous, nextState];
        return next.length > HISTORY_LIMIT ? next.slice(-HISTORY_LIMIT) : next;
      });
    }
  }, [updateThroughputStats]);

  const enqueueRealtimeState = useCallback(
    (nextState: NetworkState) => {
      pendingStateRef.current = nextState;
      if (stateFlushRafRef.current !== null) return;
      stateFlushRafRef.current = window.requestAnimationFrame(() => {
        stateFlushRafRef.current = null;
        const pending = pendingStateRef.current;
        pendingStateRef.current = null;
        if (pending) commitState(pending);
      });
    },
    [commitState],
  );

  const refreshState = useCallback(async () => {
    const nextState = await getNetworkState();
    commitState(nextState);
    setStatus(`Tick ${nextState.current_tick}`);
  }, [commitState]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [config, initialState] = await Promise.all([
          getClientConfig(),
          getNetworkState(),
        ]);
        if (cancelled) return;
        commitState(initialState);
        setStatus(`Tick ${initialState.current_tick}`);

        const connect = () => {
          if (cancelled) return;
          setConnection("connecting");
          const ws = new WebSocket(config.websocket_url);
          wsRef.current = ws;
          ws.onopen = () => {
            setConnection("connected");
            setStatus("Streaming updates");
          };
          ws.onmessage = (event) => {
            const message = JSON.parse(event.data) as {
              type: string;
              data?: NetworkState;
              message?: string;
              result?: unknown;
              success?: boolean;
            };
            if (
              (message.type === "network_state" || message.type === "network_update") &&
              message.data
            ) {
              enqueueRealtimeState(message.data);
            } else if (message.type === "error") {
              setStatus(message.message ?? "Server error");
            } else if (message.type === "signal_result") {
              setStatus(message.success ? "Signal sent" : "Signal failed");
            } else if (message.type === "tick_result") {
              setStatus("Tick executed");
            }
          };
          ws.onclose = () => {
            setConnection("disconnected");
            if (!cancelled) {
              reconnectTimerRef.current = window.setTimeout(connect, 1200);
            }
          };
          ws.onerror = () => {
            setConnection("disconnected");
          };
        };

        connect();
      } catch (error) {
        if (!cancelled) {
          setConnection("disconnected");
          setStatus(error instanceof Error ? error.message : String(error));
        }
      }
    })();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (stateFlushRafRef.current !== null) {
        window.cancelAnimationFrame(stateFlushRafRef.current);
        stateFlushRafRef.current = null;
      }
      pendingStateRef.current = null;
      wsRef.current?.close();
    };
  }, [commitState, enqueueRealtimeState]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        void runCommand("tick", async () => {
          await executeTick();
          await refreshState();
        });
      } else if (event.key === "Enter") {
        event.preventDefault();
        void runCommand("ticks", async () => {
          await executeTicks(tickCount);
          await refreshState();
        });
      } else if (event.key.toLowerCase() === "s") {
        void runCommand("start", async () => {
          await startNetwork(tickRate);
          await refreshState();
        });
      } else if (event.key.toLowerCase() === "x") {
        void runCommand("stop", async () => {
          await stopNetwork();
          await refreshState();
        });
      } else if (event.key.toLowerCase() === "f") {
        canvasRef.current?.fit();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [refreshState, tickCount, tickRate]);

  const runCommand = useCallback(
    async (label: string, command: () => Promise<void>) => {
      setBusy(true);
      setStatus(`${label}...`);
      try {
        await command();
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const stats = state?.statistics;
  const activeRatio = stats?.num_neurons
    ? stats.active_neurons / stats.num_neurons
    : 0;

  return (
    <main className="appShell">
      <header className="topBar">
        <div>
          <h1>Neural Network Visualization</h1>
          <div className="topMeta">
            <StatusPill connection={connection} />
            <span>{status}</span>
          </div>
        </div>
        <div className="topStats">
          <Metric label="tick" value={String(state?.current_tick ?? 0)} />
          <Metric label="fps" value={frameStats.fps.toFixed(0)} />
          <Metric label="tps" value={formatTps(throughput.tps)} />
          <Metric label="ms/t" value={formatMsPerTick(throughput.msPerTick)} />
          <Metric label="nodes" value={String(frameStats.nodes)} />
          <Metric label="edges" value={String(frameStats.edges)} />
        </div>
      </header>

      <section className={clsx("workspace", dynamicsCollapsed && "dynamicsCollapsed")}>
        <aside className="sidePanel">
          <section className="panelSection">
            <h2>Simulation</h2>
            <div className="buttonRow">
              <IconButton
                icon={<StepForward size={16} />}
                label="Tick"
                disabled={busy}
                onClick={() =>
                  void runCommand("tick", async () => {
                    await executeTick();
                    await refreshState();
                  })
                }
              />
              <IconButton
                icon={<Play size={16} />}
                label="Start"
                disabled={busy}
                onClick={() =>
                  void runCommand("start", async () => {
                    await startNetwork(tickRate);
                    await refreshState();
                  })
                }
              />
              <IconButton
                icon={<Pause size={16} />}
                label="Stop"
                disabled={busy}
                onClick={() =>
                  void runCommand("stop", async () => {
                    await stopNetwork();
                    await refreshState();
                    resetThroughputStats();
                    setStatus("Stopped");
                  })
                }
              />
            </div>
            <div className="fieldGrid">
              <NumberField
                label="ticks"
                value={tickCount}
                min={1}
                max={1000}
                step={1}
                onChange={setTickCount}
              />
              <NumberField
                label="TPS"
                value={tickRate}
                min={0.1}
                max={100}
                step={0.1}
                onChange={setTickRate}
              />
            </div>
            <IconButton
              icon={<Zap size={16} />}
              label={`Run ${tickCount}`}
              disabled={busy}
              full
              onClick={() =>
                void runCommand("ticks", async () => {
                  await executeTicks(tickCount);
                  await refreshState();
                })
              }
            />
          </section>

          <section className="panelSection">
            <h2>Signal</h2>
            <div className="fieldGrid">
              <TextField
                label="neuron"
                value={signalNeuron}
                onChange={setSignalNeuron}
              />
              <NumberField
                label="synapse"
                value={signalSynapse}
                min={0}
                max={9999}
                step={1}
                onChange={setSignalSynapse}
              />
              <NumberField
                label="strength"
                value={signalStrength}
                min={0}
                max={10}
                step={0.1}
                onChange={setSignalStrength}
              />
            </div>
            <IconButton
              icon={<Send size={16} />}
              label="Send Signal"
              disabled={busy || Number.isNaN(Number(signalNeuron))}
              full
              onClick={() =>
                void runCommand("signal", async () => {
                  await sendSignal({
                    neuronId: Number(signalNeuron),
                    synapseId: signalSynapse,
                    strength: signalStrength,
                  });
                  await refreshState();
                })
              }
            />
          </section>

          <section className="panelSection">
            <h2>Stimulus</h2>
            <div className="fieldGrid">
              <TextField
                label="label"
                value={stimulusLabel}
                onChange={setStimulusLabel}
              />
              <TextField
                label="class"
                value={stimulusClass}
                onChange={setStimulusClass}
              />
              <TextField
                label="presentation"
                value={stimulusPresentation}
                onChange={setStimulusPresentation}
              />
              <TextField
                label="dataset"
                value={stimulusDataset}
                onChange={setStimulusDataset}
              />
              <NumberField
                label="epoch"
                value={stimulusEpoch}
                min={0}
                max={1_000_000}
                step={1}
                onChange={setStimulusEpoch}
              />
              <NumberField
                label="confidence"
                value={stimulusConfidence}
                min={0}
                max={1}
                step={0.01}
                onChange={setStimulusConfidence}
              />
              <TextField
                label="prediction"
                value={stimulusPrediction}
                onChange={setStimulusPrediction}
              />
            </div>
            <div className="buttonRow stimulusButtons">
              <IconButton
                icon={<Activity size={16} />}
                label="Set"
                disabled={busy}
                onClick={() =>
                  void runCommand("stimulus", async () => {
                    await setStimulusMetadata({
                      label: valueOrUndefined(stimulusLabel),
                      class_name: valueOrUndefined(stimulusClass),
                      presentation_id: valueOrUndefined(stimulusPresentation),
                      dataset_name: valueOrUndefined(stimulusDataset),
                      epoch: stimulusEpoch,
                      predicted_label: valueOrUndefined(stimulusPrediction),
                      confidence: stimulusConfidence > 0 ? stimulusConfidence : undefined,
                      source: "web_viz",
                    });
                    await refreshState();
                  })
                }
              />
              <IconButton
                icon={<CircleStop size={16} />}
                label="Clear"
                disabled={busy}
                onClick={() =>
                  void runCommand("clear stimulus", async () => {
                    await clearStimulusMetadata();
                    await refreshState();
                  })
                }
              />
            </div>
          </section>

          <section className="panelSection">
            <h2>Renderer</h2>
            <div className="segmented">
              {LAYOUTS.map((item) => (
                <button
                  key={item.id}
                  className={clsx(layout === item.id && "active")}
                  onClick={() => setLayout(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <label className="checkRow">
              <input
                type="checkbox"
                checked={showEdges}
                onChange={(event) => setShowEdges(event.target.checked)}
              />
              <span>edges</span>
            </label>
            <RangeField
              label="edge alpha"
              value={edgeOpacity}
              min={0}
              max={0.55}
              step={0.01}
              onChange={setEdgeOpacity}
            />
            <RangeField
              label="node size"
              value={nodeScale}
              min={0.7}
              max={2}
              step={0.05}
              onChange={setNodeScale}
            />
            <div className="buttonRow">
              <IconButton
                icon={<Maximize2 size={16} />}
                label="Fit"
                onClick={() => canvasRef.current?.fit()}
              />
              <IconButton
                icon={<RotateCcw size={16} />}
                label="Reload"
                disabled={busy}
                onClick={() => void runCommand("reload", refreshState)}
              />
            </div>
          </section>

          <section className="panelSection">
            <h2>Network</h2>
            <div className="statsGrid">
              <Metric label="active" value={String(stats?.active_neurons ?? 0)} />
              <Metric
                label="avg S"
                value={formatNumber(stats?.avg_potential ?? 0)}
              />
              <Metric
                label="max S"
                value={formatNumber(stats?.max_potential ?? 0)}
              />
              <Metric
                label="density"
                value={formatPercent(stats?.graph_density ?? 0)}
              />
            </div>
            <div className="activityBar" aria-label="active neuron ratio">
              <div style={{ width: `${Math.min(100, activeRatio * 100)}%` }} />
            </div>
          </section>
        </aside>

        <section className="canvasPanel">
          <CanvasNetwork
            ref={canvasRef}
            state={state}
            selectedId={selectedId}
            layout={layout}
            showEdges={showEdges}
            edgeOpacity={edgeOpacity}
            nodeScale={nodeScale}
            onSelect={handleSelectNode}
            onFrameStats={setFrameStats}
          />
        </section>

        <aside className={clsx("detailPanel", "dynamicsRail", dynamicsCollapsed && "collapsed")}>
          <button
            className="dynamicsCollapseButton"
            type="button"
            aria-expanded={!dynamicsCollapsed}
            aria-label={dynamicsCollapsed ? "Expand dynamics" : "Collapse dynamics"}
            title={dynamicsCollapsed ? "Expand dynamics" : "Collapse dynamics"}
            onClick={() => setDynamicsCollapsed((collapsed) => !collapsed)}
          >
            {dynamicsCollapsed ? (
              <PanelRightOpen size={16} />
            ) : (
              <PanelRightClose size={16} />
            )}
          </button>
          {dynamicsCollapsed ? (
            <div className="dynamicsCollapsedLabel">
              {rightPaneTab === "dynamics"
                ? "Dynamics"
                : rightPaneTab === "clustering"
                  ? "Clusters"
                  : "Analysis"}
            </div>
          ) : (
            <>
              <div className="rightPaneTabs" role="tablist" aria-label="Right pane">
                <button
                  type="button"
                  role="tab"
                  aria-selected={rightPaneTab === "dynamics"}
                  className={clsx(rightPaneTab === "dynamics" && "active")}
                  onClick={() => setRightPaneTab("dynamics")}
                >
                  Dynamics
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={rightPaneTab === "clustering"}
                  className={clsx(rightPaneTab === "clustering" && "active")}
                  onClick={() => setRightPaneTab("clustering")}
                >
                  Clustering
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={rightPaneTab === "analysis"}
                  className={clsx(rightPaneTab === "analysis" && "active")}
                  onClick={() => setRightPaneTab("analysis")}
                >
                  Analysis
                </button>
              </div>
              {rightPaneTab === "dynamics" ? (
                <DynamicsPanel
                  state={state}
                  history={history}
                  selectedId={selectedId}
                  onResetHistory={() => {
                    lastHistoryKeyRef.current = state ? historyKey(state) : "";
                    setHistory(state ? [state] : []);
                  }}
                />
              ) : rightPaneTab === "clustering" ? (
                <NeuronClusteringPanel
                  state={state}
                  selectedId={selectedId}
                  onSelect={handleSelectNode}
                />
              ) : (
                <LiveAnalysisPanel
                  state={state}
                  history={history}
                  selectedId={selectedId}
                />
              )}
            </>
          )}
        </aside>
      </section>
      <NeuronBottomSheet
        node={selectedNode}
        state={state}
        history={history}
        onClose={() => setSelectedId(null)}
      />
    </main>
  );
}

function StatusPill({ connection }: { connection: ConnectionStatus }) {
  const connected = connection === "connected";
  return (
    <span className={clsx("statusPill", connected ? "good" : "bad")}>
      {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
      {connection}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function IconButton({
  icon,
  label,
  onClick,
  disabled,
  full,
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  full?: boolean;
}) {
  return (
    <button
      className={clsx("iconButton", full && "full")}
      onClick={onClick}
      disabled={disabled}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        data-testid={`number-${testIdFor(label)}`}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        data-testid={`text-${testIdFor(label)}`}
        type="text"
        inputMode="numeric"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function RangeField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="rangeField">
      <span>
        {label}
        <strong>{value.toFixed(step < 0.05 ? 2 : 1)}</strong>
      </span>
      <input
        data-testid={`range-${testIdFor(label)}`}
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function NeuronDetails({
  node,
  state,
}: {
  node: GraphNode | null;
  state: NetworkState | null;
}) {
  const incident = useMemo(() => {
    if (!node || !state) return [];
    return state.elements.edges
      .filter((edge) => edge.data.source === node.data.id || edge.data.target === node.data.id)
      .slice(0, 32);
  }, [node, state]);

  if (!node) {
    return (
      <div className="emptyDetails">
        <Activity size={22} />
        <span>No neuron selected</span>
      </div>
    );
  }

  const d = node.data;
  return (
    <div className="detailsContent">
      <div className="detailsHeader">
        <h2>{d.label ?? d.id}</h2>
        <span>{d.type}</span>
      </div>
      <dl className="detailList">
        <Detail label="id" value={String(d.neuron_id ?? d.id)} />
        <Detail label="S" value={formatNumber(d.membrane_potential ?? 0)} />
        <Detail label="F_avg" value={formatNumber(d.firing_rate ?? 0)} />
        <Detail label="t_ref" value={formatNumber(d.t_ref ?? 0)} />
        <Detail label="O" value={formatNumber(d.output ?? 0)} />
        <Detail label="layer" value={d.layer_name ?? String(d.layer ?? "-")} />
        <Detail label="synapses" value={String(d.synapses?.length ?? 0)} />
        <Detail label="terminals" value={String(d.terminals?.length ?? 0)} />
      </dl>
      <section className="edgeList">
        <h3>Connections</h3>
        {incident.length === 0 ? (
          <p>None</p>
        ) : (
          incident.map((edge) => (
            <div key={edge.data.id} className="edgeRow">
              <CircleStop size={10} />
              <span>{edge.data.source}</span>
              <span>{edge.data.target}</span>
            </div>
          ))
        )}
      </section>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}

function formatNumber(value: number) {
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(3);
}

function formatTps(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "0.0";
  if (value >= 1000) return value.toFixed(0);
  if (value >= 100) return value.toFixed(1);
  return value.toFixed(2);
}

function formatMsPerTick(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "0.0";
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function valueOrUndefined(value: string) {
  const trimmed = value.trim();
  return trimmed === "" ? undefined : trimmed;
}

function historyKey(state: NetworkState) {
  const stats = state.statistics;
  return [
    state.current_tick,
    stats?.active_neurons ?? 0,
    rounded(stats?.avg_potential ?? 0),
    rounded(stats?.avg_firing_rate ?? 0),
    rounded(stats?.avg_t_ref ?? 0),
    rounded(stats?.max_potential ?? 0),
    rounded(stats?.free_energy ?? stats?.state_energy ?? 0),
    stats?.num_traveling_signals ?? 0,
  ].join("|");
}

function rounded(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function testIdFor(label: string) {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}
