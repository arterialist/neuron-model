import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import clsx from "clsx";
import type {
  GraphEdge,
  GraphNode,
  GraphNodeData,
  NetworkState,
  StimulusMetadata,
} from "./types";

type AnalysisView = "network" | "labels" | "assemblies" | "connectivity";
type ScalarMetric = "meanS" | "meanF" | "activeRatio" | "meanTRef" | "varianceS";

interface LiveAnalysisPanelProps {
  state: NetworkState | null;
  history: NetworkState[];
  selectedId: string | null;
}

interface LayerOption {
  key: string;
  label: string;
  count: number;
}

interface LayerStats extends LayerOption {
  meanS: number;
  meanF: number;
  meanTRef: number;
  activeRatio: number;
  varianceS: number;
  fired: number;
  tRefMin: number;
  tRefMax: number;
}

interface Point2D {
  id?: string;
  x: number;
  y: number;
  color?: string;
  r?: number;
}

interface HeatCell {
  x: string;
  y: string;
  value: number;
}

interface CorrelationGraphData {
  nodes: Array<{ id: string }>;
  edges: Array<{ source: string; target: string; weight: number }>;
}

const ANALYSIS_VIEWS: Array<{ id: AnalysisView; label: string }> = [
  { id: "network", label: "Network" },
  { id: "labels", label: "Labels" },
  { id: "assemblies", label: "Assemblies" },
  { id: "connectivity", label: "Synapses" },
];

const METRICS: Array<{ id: ScalarMetric; label: string }> = [
  { id: "meanS", label: "S" },
  { id: "meanF", label: "F_avg" },
  { id: "activeRatio", label: "active" },
  { id: "meanTRef", label: "t_ref" },
  { id: "varianceS", label: "var S" },
];

const PALETTE = [
  "#74d4ff",
  "#f87171",
  "#5ee090",
  "#fb923c",
  "#c084fc",
  "#facc15",
  "#38bdf8",
  "#f472b6",
  "#a3e635",
  "#60a5fa",
  "#f59e0b",
  "#2dd4bf",
];

export function LiveAnalysisPanel({
  state,
  history,
  selectedId,
}: LiveAnalysisPanelProps) {
  const [view, setView] = useState<AnalysisView>("network");
  const labelledCount = useMemo(
    () => history.filter((sample) => labelKey(sample.stimulus) !== null).length,
    [history],
  );

  return (
    <section className="panelSection analysisPanel" data-testid="analysis-panel">
      <h2>Live Analysis</h2>
      <StimulusReadout stimulus={state?.stimulus} labelledCount={labelledCount} />
      <div className="segmented analysisTabs" role="tablist" aria-label="Analysis views">
        {ANALYSIS_VIEWS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={view === item.id}
            className={clsx(view === item.id && "active")}
            onClick={() => setView(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {view === "network" && (
        <NetworkActivityView state={state} history={history} selectedId={selectedId} />
      )}
      {view === "labels" && <LabelActivityView state={state} history={history} />}
      {view === "assemblies" && (
        <AssembliesView state={state} history={history} selectedId={selectedId} />
      )}
      {view === "connectivity" && <ConnectivityView state={state} history={history} />}
    </section>
  );
}

function StimulusReadout({
  stimulus,
  labelledCount,
}: {
  stimulus?: StimulusMetadata;
  labelledCount: number;
}) {
  const label = labelKey(stimulus) ?? "-";
  const prediction = predictionLabel(stimulus);
  return (
    <div className="analysisStimulus" data-testid="stimulus-readout">
      <div>
        <span>label</span>
        <strong>{label}</strong>
      </div>
      <div>
        <span>presentation</span>
        <strong>{formatText(stimulus?.presentation_id ?? "-")}</strong>
      </div>
      <div>
        <span>dataset</span>
        <strong>{formatText(stimulus?.dataset_name ?? "-")}</strong>
      </div>
      <div>
        <span>prediction</span>
        <strong>{prediction}</strong>
      </div>
      <div>
        <span>labelled</span>
        <strong>{String(labelledCount)}</strong>
      </div>
      <div>
        <span>seq</span>
        <strong>{String(stimulus?.sequence ?? 0)}</strong>
      </div>
    </div>
  );
}

function NetworkActivityView({
  state,
  history,
  selectedId,
}: LiveAnalysisPanelProps) {
  const nodes = useMemo(() => neuronNodes(state), [state]);
  const layers = useMemo(() => layerStats(state), [state]);
  const series = useMemo(() => globalSeries(history), [history]);
  const raster = useMemo(() => spikeRaster(history, selectedId), [history, selectedId]);
  const correlations = useMemo(() => temporalCorrelations(history), [history]);
  const homeostat = useMemo(() => homeostaticCurve(nodes), [nodes]);
  const scatterPoints = useMemo(
    () =>
      nodes.map((node) => ({
        id: node.data.id,
        x: numberValue(node.data.firing_rate),
        y: numberValue(node.data.t_ref),
        color: colorForKey(getLayerKey(node.data)),
        r: node.data.id === selectedId ? 4.8 : 3.1,
      })),
    [nodes, selectedId],
  );

  return (
    <div className="analysisStack">
      <MiniCard title="F_avg × t_ref" value={`${nodes.length}n`}>
        <ScatterPlot points={scatterPoints} xLabel="F_avg" yLabel="t_ref" />
      </MiniCard>

      <MiniCard title="Layer firing rate" value={`${layers.length}L`}>
        <BarChart
          items={layers.map((layer) => ({
            key: layer.key,
            label: compactLabel(layer.label),
            value: layer.meanF,
            color: colorForKey(layer.key),
          }))}
        />
      </MiniCard>

      <MiniCard title="Layer state" value={formatScalar(series.at(-1)?.meanS ?? 0)}>
        <LayerTimeline history={history} metric="meanS" />
      </MiniCard>

      <MiniCard title="Spike raster" value={`${raster.neurons}n`}>
        <RasterPlot raster={raster} />
      </MiniCard>

      <MiniCard title="Phase portrait" value={`${series.length}t`}>
        <ScatterPlot
          points={series.map((sample) => ({
            x: sample.meanS,
            y: sample.meanF,
            color: "#74d4ff",
            r: 2.8,
          }))}
          xLabel="S"
          yLabel="F_avg"
          polyline
        />
      </MiniCard>

      <MiniCard title="S variance decay" value={formatScalar(series.at(-1)?.varianceS ?? 0)}>
        <Sparkline
          values={series.map((sample) => sample.varianceS)}
          color="#c084fc"
          height={72}
        />
      </MiniCard>

      <MiniCard title="Homeostatic response" value={`${homeostat.length} bins`}>
        <ScatterPlot
          points={homeostat.map((point) => ({
            x: point.x,
            y: point.y,
            color: "#fb923c",
            r: 3.4,
          }))}
          xLabel="F_avg"
          yLabel="t_ref"
          polyline
        />
      </MiniCard>

      <MiniCard title="Temporal correlation" value={`${correlations.edges.length}e`}>
        <CorrelationGraph graph={correlations} />
      </MiniCard>
    </div>
  );
}

function LabelActivityView({
  state,
  history,
}: {
  state: NetworkState | null;
  history: NetworkState[];
}) {
  const [metric, setMetric] = useState<ScalarMetric>("meanF");
  const labelled = useMemo(
    () => history.filter((sample) => labelKey(sample.stimulus) !== null),
    [history],
  );
  const labelMatrix = useMemo(() => labelLayerMatrix(labelled, metric), [labelled, metric]);
  const labelSeries = useMemo(() => labelledActivitySeries(labelled), [labelled]);
  const leakage = useMemo(() => leakageMatrix(labelled), [labelled]);
  const preferred = useMemo(() => preferredLabelMap(labelled, state), [labelled, state]);
  const activityClusters = useMemo(() => clusterPresentations(labelled), [labelled]);
  const hierarchy = useMemo(() => conceptSimilarityPairs(leakage), [leakage]);

  return (
    <div className="analysisStack">
      <div className="analysisControlRow">
        <label className="field">
          <span>metric</span>
          <select
            data-testid="label-metric-select"
            value={metric}
            onChange={(event) => setMetric(event.target.value as ScalarMetric)}
          >
            {METRICS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <MiniCard title="Label × layer heatmap" value={`${labelMatrix.labels.length} labels`}>
        {labelMatrix.cells.length > 0 ? (
          <Heatmap
            cells={labelMatrix.cells}
            xLabels={labelMatrix.layers}
            yLabels={labelMatrix.labels}
          />
        ) : (
          <EmptyViz text="No labelled samples" />
        )}
      </MiniCard>

      <MiniCard title="Label time series" value={`${labelSeries.length} labels`}>
        {labelSeries.length > 0 ? <MultiSparkline series={labelSeries} /> : <EmptyViz text="No labelled samples" />}
      </MiniCard>

      <MiniCard title="Preferred label map" value={`${preferred.active} active`}>
        {preferred.points.length > 0 ? (
          <ScatterPlot points={preferred.points} xLabel="layer" yLabel="neuron" />
        ) : (
          <EmptyViz text="No labelled samples" />
        )}
      </MiniCard>

      <MiniCard title="Presentation clusters" value={`${activityClusters.clusters} clusters`}>
        {activityClusters.points.length > 0 ? (
          <ScatterPlot points={activityClusters.points} xLabel="S" yLabel="F_avg" />
        ) : (
          <EmptyViz text="No presentations" />
        )}
      </MiniCard>

      <MiniCard title="Attractor leakage" value={`${leakage.cells.length} cells`}>
        {leakage.cells.length > 0 ? (
          <Heatmap cells={leakage.cells} xLabels={leakage.predicted} yLabels={leakage.actual} />
        ) : (
          <EmptyViz text="No predictions" />
        )}
      </MiniCard>

      <MiniCard title="Concept similarity" value={`${hierarchy.length} links`}>
        {hierarchy.length > 0 ? <PairList pairs={hierarchy} /> : <EmptyViz text="No prediction matrix" />}
      </MiniCard>
    </div>
  );
}

function AssembliesView({
  state,
  history,
  selectedId,
}: LiveAnalysisPanelProps) {
  const nodes = useMemo(() => neuronNodes(state), [state]);
  const featureClusters = useMemo(() => clusterNeuronsByFeatures(nodes), [nodes]);
  const synchrony = useMemo(() => synchronyAssemblies(history), [history]);
  const preferred = useMemo(() => preferredLabelMap(labelledHistory(history), state), [
    history,
    state,
  ]);

  return (
    <div className="analysisStack">
      <MiniCard title="Feature assemblies" value={`${featureClusters.clusters.length} clusters`}>
        <ScatterPlot
          points={featureClusters.points.map((point) => ({
            id: point.id,
            x: point.x,
            y: point.y,
            color: colorForIndex(point.cluster),
            r: point.id === selectedId ? 4.8 : 3.2,
          }))}
          xLabel="S"
          yLabel="F_avg"
        />
      </MiniCard>

      <MiniCard title="Synchrony assemblies" value={`${synchrony.clusters.length} clusters`}>
        <CorrelationGraph graph={synchrony.graph} />
        <ClusterBars clusters={synchrony.clusters} />
      </MiniCard>

      <MiniCard title="Preferred-label assemblies" value={`${preferred.labels.length} labels`}>
        {preferred.points.length > 0 ? (
          <ScatterPlot points={preferred.points} xLabel="layer" yLabel="neuron" />
        ) : (
          <EmptyViz text="No labelled samples" />
        )}
      </MiniCard>
    </div>
  );
}

function ConnectivityView({
  state,
  history,
}: {
  state: NetworkState | null;
  history: NetworkState[];
}) {
  const connectivity = useMemo(() => connectivityAnalysis(state), [state]);
  const weightSeries = useMemo(() => edgeWeightSeries(history), [history]);
  const byLabel = useMemo(() => connectivityByLabel(history), [history]);

  return (
    <div className="analysisStack">
      <MiniCard title="Synaptic weight matrix" value={`${connectivity.nodeCount}n`}>
        {connectivity.cells.length > 0 ? (
          <Heatmap
            cells={connectivity.cells}
            xLabels={connectivity.labels}
            yLabels={connectivity.labels}
          />
        ) : (
          <EmptyViz text="No weighted edges" />
        )}
      </MiniCard>

      <MiniCard title="Connectivity communities" value={`${connectivity.clusters.length} groups`}>
        <ClusterBars clusters={connectivity.clusters} />
      </MiniCard>

      <MiniCard title="Temporal edge weight" value={`${weightSeries.length}t`}>
        <Sparkline values={weightSeries} color="#5ee090" height={72} />
      </MiniCard>

      <MiniCard title="Cross-label connectivity" value={`${byLabel.length} labels`}>
        {byLabel.length > 0 ? (
          <BarChart
            items={byLabel.map((item, index) => ({
              key: item.label,
              label: item.label,
              value: item.meanWeight,
              color: colorForIndex(index),
            }))}
          />
        ) : (
          <EmptyViz text="No labelled edge history" />
        )}
      </MiniCard>
    </div>
  );
}

function MiniCard({
  title,
  value,
  children,
}: {
  title: string;
  value: string;
  children: ReactNode;
}) {
  return (
    <section className="analysisCard">
      <header>
        <span>{title}</span>
        <strong>{value}</strong>
      </header>
      {children}
    </section>
  );
}

function EmptyViz({ text }: { text: string }) {
  return <div className="analysisEmpty">{text}</div>;
}

function ScatterPlot({
  points,
  xLabel,
  yLabel,
  polyline = false,
}: {
  points: Point2D[];
  xLabel: string;
  yLabel: string;
  polyline?: boolean;
}) {
  const width = 260;
  const height = 142;
  const bounds = bounds2D(points);
  const projected = points.map((point) => ({
    ...point,
    sx: scale(point.x, bounds.minX, bounds.maxX, 22, width - 12),
    sy: scale(point.y, bounds.minY, bounds.maxY, height - 22, 12),
  }));
  const line = projected.map((point) => `${point.sx.toFixed(1)},${point.sy.toFixed(1)}`).join(" ");

  return (
    <svg className="analysisSvg" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <PlotGrid width={width} height={height} />
      {polyline && projected.length > 1 && (
        <polyline points={line} fill="none" stroke="#74d4ff" strokeWidth="1.4" opacity="0.65" />
      )}
      {projected.map((point, index) => (
        <circle
          key={`${point.id ?? "p"}-${index}`}
          cx={point.sx}
          cy={point.sy}
          r={point.r ?? 3.2}
          fill={point.color ?? "#74d4ff"}
          opacity={0.82}
        />
      ))}
      <text x={width - 8} y={height - 5} textAnchor="end" className="analysisAxisText">
        {xLabel}
      </text>
      <text x={8} y={13} className="analysisAxisText">
        {yLabel}
      </text>
    </svg>
  );
}

function PlotGrid({ width, height }: { width: number; height: number }) {
  const lines = [];
  for (let index = 0; index <= 4; index += 1) {
    const x = 22 + ((width - 34) * index) / 4;
    const y = 12 + ((height - 34) * index) / 4;
    lines.push(<line key={`x-${index}`} x1={x} x2={x} y1={12} y2={height - 22} />);
    lines.push(<line key={`y-${index}`} x1={22} x2={width - 12} y1={y} y2={y} />);
  }
  return <g className="analysisGridLines">{lines}</g>;
}

function Sparkline({
  values,
  color,
  height = 56,
}: {
  values: number[];
  color: string;
  height?: number;
}) {
  const width = 260;
  const finite = values.filter(Number.isFinite);
  const min = finite.length ? Math.min(...finite) : 0;
  const max = finite.length ? Math.max(...finite) : 1;
  const path = values
    .map((value, index) => {
      const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * width;
      const y = scale(value, min, max, height - 8, 8);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="analysisSvg spark" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <path d={path} stroke={color} />
    </svg>
  );
}

function MultiSparkline({
  series,
}: {
  series: Array<{ label: string; values: number[] }>;
}) {
  return (
    <div className="analysisMultiSpark">
      {series.slice(0, 8).map((item, index) => (
        <div key={item.label} className="analysisSparkRow">
          <span>{item.label}</span>
          <Sparkline values={item.values} color={colorForIndex(index)} height={34} />
        </div>
      ))}
    </div>
  );
}

function BarChart({
  items,
}: {
  items: Array<{ key: string; label: string; value: number; color?: string }>;
}) {
  const max = Math.max(1e-9, ...items.map((item) => Math.abs(item.value)));
  return (
    <div className="analysisBars">
      {items.slice(0, 16).map((item) => (
        <div key={item.key} className="analysisBarRow">
          <span>{item.label}</span>
          <div>
            <i
              style={{
                width: `${Math.min(100, (Math.abs(item.value) / max) * 100)}%`,
                background: item.color ?? "#74d4ff",
              }}
            />
          </div>
          <strong>{formatScalar(item.value)}</strong>
        </div>
      ))}
    </div>
  );
}

function Heatmap({
  cells,
  xLabels,
  yLabels,
}: {
  cells: HeatCell[];
  xLabels: string[];
  yLabels: string[];
}) {
  const width = 260;
  const left = 42;
  const top = 12;
  const cellW = Math.max(9, (width - left - 8) / Math.max(1, xLabels.length));
  const cellH = Math.max(10, 124 / Math.max(1, yLabels.length));
  const valueMap = new Map(cells.map((cell) => [`${cell.x}|${cell.y}`, cell.value]));
  const max = Math.max(1e-9, ...cells.map((cell) => Math.abs(cell.value)));
  const height = top + yLabels.length * cellH + 22;

  return (
    <svg className="analysisSvg heat" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      {yLabels.map((yLabel, yIndex) => (
        <g key={yLabel}>
          <text x={left - 5} y={top + yIndex * cellH + cellH * 0.72} textAnchor="end" className="analysisAxisText">
            {compactLabel(yLabel)}
          </text>
          {xLabels.map((xLabel, xIndex) => {
            const value = valueMap.get(`${xLabel}|${yLabel}`) ?? 0;
            return (
              <rect
                key={`${xLabel}-${yLabel}`}
                x={left + xIndex * cellW}
                y={top + yIndex * cellH}
                width={Math.max(1, cellW - 1)}
                height={Math.max(1, cellH - 1)}
                fill={heatColor(value / max)}
              />
            );
          })}
        </g>
      ))}
      {xLabels.slice(0, 12).map((label, index) => (
        <text
          key={label}
          x={left + index * cellW + cellW / 2}
          y={height - 5}
          textAnchor="middle"
          className="analysisAxisText"
        >
          {compactLabel(label)}
        </text>
      ))}
    </svg>
  );
}

function RasterPlot({ raster }: { raster: ReturnType<typeof spikeRaster> }) {
  const width = 260;
  const height = 132;
  const xScale = Math.max(1, raster.ticks.length - 1);
  const yScale = Math.max(1, raster.ids.length - 1);
  return (
    <svg className="analysisSvg raster" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <PlotGrid width={width} height={height} />
      {raster.points.map((point, index) => (
        <rect
          key={`${point.tick}-${point.id}-${index}`}
          x={22 + (point.tickIndex / xScale) * (width - 34)}
          y={12 + (point.neuronIndex / yScale) * (height - 34)}
          width={2}
          height={2}
          fill="#f87171"
          opacity="0.9"
        />
      ))}
    </svg>
  );
}

function LayerTimeline({
  history,
  metric,
}: {
  history: NetworkState[];
  metric: ScalarMetric;
}) {
  const layers = useMemo(() => layerSeries(history, metric), [history, metric]);
  return (
    <div className="analysisMultiSpark">
      {layers.slice(0, 6).map((layer, index) => (
        <div key={layer.key} className="analysisSparkRow">
          <span>{compactLabel(layer.label)}</span>
          <Sparkline values={layer.values} color={colorForIndex(index)} height={34} />
        </div>
      ))}
    </div>
  );
}

function CorrelationGraph({
  graph,
}: {
  graph: CorrelationGraphData;
}) {
  const data = graph;
  const width = 260;
  const height = 130;
  const radius = 48;
  const cx = width / 2;
  const cy = height / 2;
  const positions = new Map<string, { x: number; y: number }>();
  data.nodes.forEach((node, index) => {
    const angle = (index / Math.max(1, data.nodes.length)) * Math.PI * 2 - Math.PI / 2;
    positions.set(node.id, { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius });
  });
  return (
    <svg className="analysisSvg graph" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      {data.edges.slice(0, 80).map((edge, index) => {
        const a = positions.get(edge.source);
        const b = positions.get(edge.target);
        if (!a || !b) return null;
        return (
          <line
            key={`${edge.source}-${edge.target}-${index}`}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke="#74d4ff"
            strokeOpacity={Math.min(0.85, Math.max(0.14, edge.weight))}
            strokeWidth={1 + Math.max(0, edge.weight) * 2}
          />
        );
      })}
      {data.nodes.map((node, index) => {
        const point = positions.get(node.id);
        if (!point) return null;
        return (
          <circle
            key={node.id}
            cx={point.x}
            cy={point.y}
            r={4}
            fill={colorForIndex(index)}
            opacity="0.9"
          />
        );
      })}
    </svg>
  );
}

function ClusterBars({ clusters }: { clusters: Array<{ id: string; count: number }> }) {
  return (
    <div className="analysisClusterBars">
      {clusters.slice(0, 10).map((cluster, index) => (
        <span key={cluster.id} style={{ background: colorForIndex(index) }}>
          {cluster.id}:{cluster.count}
        </span>
      ))}
    </div>
  );
}

function PairList({
  pairs,
}: {
  pairs: Array<{ a: string; b: string; value: number }>;
}) {
  return (
    <div className="analysisPairList">
      {pairs.slice(0, 8).map((pair) => (
        <div key={`${pair.a}-${pair.b}`}>
          <span>{pair.a} ↔ {pair.b}</span>
          <strong>{formatScalar(pair.value)}</strong>
        </div>
      ))}
    </div>
  );
}

function neuronNodes(state: NetworkState | null): GraphNode[] {
  return (state?.elements.nodes ?? []).filter((node) => node.data.type === "neuron");
}

function neuronEdges(state: NetworkState | null): GraphEdge[] {
  return (state?.elements.edges ?? []).filter((edge) => edge.data.type === "neuron");
}

function layerStats(state: NetworkState | null): LayerStats[] {
  const groups = new Map<string, GraphNode[]>();
  for (const node of neuronNodes(state)) {
    const key = getLayerKey(node.data);
    groups.set(key, [...(groups.get(key) ?? []), node]);
  }
  return Array.from(groups.entries())
    .map(([key, nodes]) => {
      const s = nodes.map((node) => numberValue(node.data.membrane_potential));
      const f = nodes.map((node) => numberValue(node.data.firing_rate));
      const tref = nodes.map((node) => numberValue(node.data.t_ref));
      const fired = nodes.filter((node) => numberValue(node.data.output) > 0).length;
      return {
        key,
        label: nodes[0]?.data.layer_name ?? `layer ${key}`,
        count: nodes.length,
        meanS: average(s),
        meanF: average(f),
        meanTRef: average(tref),
        activeRatio: nodes.length ? fired / nodes.length : 0,
        varianceS: variance(s),
        fired,
        tRefMin: tref.length ? Math.min(...tref) : 0,
        tRefMax: tref.length ? Math.max(...tref) : 0,
      };
    })
    .sort((a, b) => layerSortKey(a.key) - layerSortKey(b.key));
}

function globalSeries(history: NetworkState[]) {
  return history.map((sample) => {
    const stats = layerStats(sample);
    const nodes = neuronNodes(sample);
    return {
      tick: sample.current_tick,
      meanS: numberValue(sample.statistics.avg_potential),
      meanF: numberValue(sample.statistics.avg_firing_rate),
      meanTRef: numberValue(sample.statistics.avg_t_ref),
      activeRatio: sample.statistics.num_neurons
        ? sample.statistics.active_neurons / sample.statistics.num_neurons
        : 0,
      varianceS: variance(nodes.map((node) => numberValue(node.data.membrane_potential))),
      layers: stats,
    };
  });
}

function spikeRaster(history: NetworkState[], selectedId: string | null) {
  const samples = history.slice(-120);
  const latestNodes = neuronNodes(samples.at(-1) ?? null);
  const selectedFirst = selectedId
    ? latestNodes.filter((node) => node.data.id === selectedId)
    : [];
  const ids = [...selectedFirst, ...latestNodes.filter((node) => node.data.id !== selectedId)]
    .slice(0, 72)
    .map((node) => node.data.id);
  const idSet = new Set(ids);
  const idToIndex = new Map(ids.map((id, index) => [id, index]));
  const points: Array<{ tick: number; tickIndex: number; id: string; neuronIndex: number }> = [];
  samples.forEach((sample, tickIndex) => {
    for (const node of neuronNodes(sample)) {
      if (!idSet.has(node.data.id) || numberValue(node.data.output) <= 0) continue;
      points.push({
        tick: sample.current_tick,
        tickIndex,
        id: node.data.id,
        neuronIndex: idToIndex.get(node.data.id) ?? 0,
      });
    }
  });
  return { ticks: samples.map((sample) => sample.current_tick), ids, neurons: ids.length, points };
}

function temporalCorrelations(history: NetworkState[]) {
  const samples = history.slice(-96);
  const latest = samples.at(-1) ?? null;
  const ids = neuronNodes(latest)
    .slice(0, 28)
    .map((node) => node.data.id);
  const series = buildOutputSeries(samples, ids);
  const edges: Array<{ source: string; target: string; weight: number }> = [];
  for (let i = 0; i < ids.length; i += 1) {
    for (let j = i + 1; j < ids.length; j += 1) {
      const corr = pearson(series.get(ids[i]) ?? [], series.get(ids[j]) ?? []);
      if (corr > 0.28) edges.push({ source: ids[i], target: ids[j], weight: corr });
    }
  }
  edges.sort((a, b) => b.weight - a.weight);
  return {
    nodes: ids.map((id) => ({ id })),
    edges: edges.slice(0, 120),
  };
}

function homeostaticCurve(nodes: GraphNode[]) {
  const bins = 10;
  const bucketed = Array.from({ length: bins }, () => [] as number[]);
  for (const node of nodes) {
    const f = clamp(numberValue(node.data.firing_rate), 0, 1);
    const index = Math.min(bins - 1, Math.floor(f * bins));
    bucketed[index].push(numberValue(node.data.t_ref));
  }
  return bucketed.map((values, index) => ({
    x: (index + 0.5) / bins,
    y: average(values),
  }));
}

function layerSeries(history: NetworkState[], metric: ScalarMetric) {
  const map = new Map<string, { key: string; label: string; values: number[] }>();
  for (const sample of history) {
    for (const layer of layerStats(sample)) {
      const row = map.get(layer.key) ?? { key: layer.key, label: layer.label, values: [] };
      row.values.push(layer[metric]);
      map.set(layer.key, row);
    }
  }
  return Array.from(map.values()).sort((a, b) => layerSortKey(a.key) - layerSortKey(b.key));
}

function labelledHistory(history: NetworkState[]) {
  return history.filter((sample) => labelKey(sample.stimulus) !== null);
}

function labelLayerMatrix(history: NetworkState[], metric: ScalarMetric) {
  const labels = unique(history.map((sample) => labelKey(sample.stimulus)).filter(isString));
  const layers = unique(history.flatMap((sample) => layerStats(sample).map((layer) => layer.key)));
  const accum = new Map<string, number[]>();
  for (const sample of history) {
    const label = labelKey(sample.stimulus);
    if (!label) continue;
    for (const layer of layerStats(sample)) {
      const key = `${layer.key}|${label}`;
      const values = accum.get(key) ?? [];
      values.push(layer[metric]);
      accum.set(key, values);
    }
  }
  const cells: HeatCell[] = [];
  for (const label of labels) {
    for (const layer of layers) {
      cells.push({ x: layer, y: label, value: average(accum.get(`${layer}|${label}`) ?? []) });
    }
  }
  return { labels, layers, cells };
}

function labelledActivitySeries(history: NetworkState[]) {
  const labels = unique(history.map((sample) => labelKey(sample.stimulus)).filter(isString));
  return labels.map((label) => ({
    label,
    values: history
      .filter((sample) => labelKey(sample.stimulus) === label)
      .map((sample) => sample.statistics.avg_firing_rate ?? 0),
  }));
}

function leakageMatrix(history: NetworkState[]) {
  const actual = unique(history.map((sample) => labelKey(sample.stimulus)).filter(isString));
  const predicted = unique(
    history
      .map((sample) => firstPrediction(sample.stimulus)?.label)
      .filter((value): value is string | number => value !== undefined && value !== null)
      .map(String),
  );
  const accum = new Map<string, number[]>();
  for (const sample of history) {
    const label = labelKey(sample.stimulus);
    const pred = firstPrediction(sample.stimulus);
    if (!label || pred?.label == null) continue;
    const key = `${String(pred.label)}|${label}`;
    const values = accum.get(key) ?? [];
    values.push(numberValue(pred.confidence, 1));
    accum.set(key, values);
  }
  const cells: HeatCell[] = [];
  for (const y of actual) {
    for (const x of predicted) {
      cells.push({ x, y, value: average(accum.get(`${x}|${y}`) ?? []) });
    }
  }
  return { actual, predicted, cells };
}

function conceptSimilarityPairs(leakage: ReturnType<typeof leakageMatrix>) {
  const rowVectors = leakage.actual.map((label) =>
    leakage.predicted.map(
      (pred) => leakage.cells.find((cell) => cell.x === pred && cell.y === label)?.value ?? 0,
    ),
  );
  const pairs: Array<{ a: string; b: string; value: number }> = [];
  for (let i = 0; i < leakage.actual.length; i += 1) {
    for (let j = i + 1; j < leakage.actual.length; j += 1) {
      pairs.push({
        a: leakage.actual[i],
        b: leakage.actual[j],
        value: cosine(rowVectors[i], rowVectors[j]),
      });
    }
  }
  return pairs.sort((a, b) => b.value - a.value);
}

function preferredLabelMap(history: NetworkState[], state: NetworkState | null) {
  const labels = unique(history.map((sample) => labelKey(sample.stimulus)).filter(isString));
  const byNeuron = new Map<string, Map<string, number[]>>();
  for (const sample of history) {
    const label = labelKey(sample.stimulus);
    if (!label) continue;
    for (const node of neuronNodes(sample)) {
      const row = byNeuron.get(node.data.id) ?? new Map<string, number[]>();
      const values = row.get(label) ?? [];
      values.push(numberValue(node.data.output) + numberValue(node.data.firing_rate));
      row.set(label, values);
      byNeuron.set(node.data.id, row);
    }
  }
  const current = neuronNodes(state);
  let active = 0;
  const points = current.map((node, index) => {
    const row = byNeuron.get(node.data.id);
    let bestLabel = "";
    let bestValue = -Infinity;
    if (row) {
      for (const [label, values] of row.entries()) {
        const value = average(values);
        if (value > bestValue) {
          bestValue = value;
          bestLabel = label;
        }
      }
    }
    if (bestLabel) active += 1;
    return {
      id: node.data.id,
      x: layerSortKey(getLayerKey(node.data)),
      y: index,
      color: bestLabel ? colorForIndex(labels.indexOf(bestLabel)) : "rgba(141,152,168,0.35)",
      r: 2.9,
    };
  });
  return { labels, points, active };
}

function clusterPresentations(history: NetworkState[]) {
  const grouped = new Map<string, { label: string; samples: NetworkState[] }>();
  for (const sample of history) {
    const label = labelKey(sample.stimulus);
    if (!label) continue;
    const presentation = sample.stimulus?.presentation_id ?? sample.current_tick;
    const key = `${label}|${String(presentation)}`;
    const row = grouped.get(key) ?? { label, samples: [] };
    row.samples.push(sample);
    grouped.set(key, row);
  }
  const rows = Array.from(grouped.entries()).map(([key, row]) => {
    const features = row.samples.map((sample) => globalSeries([sample])[0]);
    return {
      id: key,
      label: row.label,
      x: average(features.map((item) => item.meanS)),
      y: average(features.map((item) => item.meanF)),
      active: average(features.map((item) => item.activeRatio)),
      tRef: average(features.map((item) => item.meanTRef)),
    };
  });
  const vectors = rows.map((row) => [row.x, row.y, row.active, row.tRef]);
  const assignments = simpleKMeans(vectors, chooseK(vectors, 8)).assignments;
  const clusterCount = unique(assignments.map(String)).length;
  return {
    clusters: clusterCount,
    points: rows.map((row, index) => ({
      id: row.id,
      x: row.x,
      y: row.y,
      color: colorForIndex(assignments[index] ?? 0),
      r: 3.6,
    })),
  };
}

function clusterNeuronsByFeatures(nodes: GraphNode[]) {
  const vectors = nodes.map((node) => [
    numberValue(node.data.membrane_potential),
    numberValue(node.data.firing_rate),
    numberValue(node.data.t_ref),
    numberValue(node.data.output),
  ]);
  const normalized = normalizeRows(vectors);
  const k = chooseK(normalized, 8);
  const solution = simpleKMeans(normalized, k);
  const clusters = clusterCounts(solution.assignments);
  return {
    clusters,
    points: nodes.map((node, index) => ({
      id: node.data.id,
      x: normalized[index]?.[0] ?? 0,
      y: normalized[index]?.[1] ?? 0,
      cluster: solution.assignments[index] ?? 0,
    })),
  };
}

function synchronyAssemblies(history: NetworkState[]) {
  const graph = temporalCorrelations(history);
  const union = new UnionFind(graph.nodes.map((node) => node.id));
  for (const edge of graph.edges) {
    if (edge.weight >= 0.42) union.union(edge.source, edge.target);
  }
  const counts = new Map<string, number>();
  for (const node of graph.nodes) {
    const root = union.find(node.id);
    counts.set(root, (counts.get(root) ?? 0) + 1);
  }
  return {
    graph,
    clusters: Array.from(counts.entries())
      .map(([id, count]) => ({ id: compactId(id), count }))
      .sort((a, b) => b.count - a.count),
  };
}

function connectivityAnalysis(state: NetworkState | null) {
  const allNodes = neuronNodes(state);
  const byId = new Map(allNodes.map((node) => [node.data.id, node]));
  const edges = neuronEdges(state);
  const selectedIds: string[] = [];
  for (const edge of edges) {
    for (const id of [edge.data.source, edge.data.target]) {
      if (!byId.has(id) || selectedIds.includes(id)) continue;
      selectedIds.push(id);
      if (selectedIds.length >= 72) break;
    }
    if (selectedIds.length >= 72) break;
  }
  const nodes = (selectedIds.length ? selectedIds.map((id) => byId.get(id)!) : allNodes).slice(0, 72);
  const labels = nodes.map((node) => compactId(String(node.data.neuron_id ?? node.data.id)));
  const idToIndex = new Map(nodes.map((node, index) => [node.data.id, index]));
  const cells: HeatCell[] = [];
  const union = new UnionFind(nodes.map((node) => node.data.id));
  for (const edge of edges) {
    const source = idToIndex.get(edge.data.source);
    const target = idToIndex.get(edge.data.target);
    if (source === undefined || target === undefined) continue;
    const weight = Math.abs(numberValue(edge.data.weight, edge.data.source_firing ? 1 : 0.1));
    cells.push({ x: labels[target], y: labels[source], value: weight });
    if (weight > 0.05) union.union(edge.data.source, edge.data.target);
  }
  const counts = new Map<string, number>();
  for (const node of nodes) {
    const root = union.find(node.data.id);
    counts.set(root, (counts.get(root) ?? 0) + 1);
  }
  const clusters = Array.from(counts.entries())
    .map(([id, count]) => ({ id: compactId(id), count }))
    .sort((a, b) => b.count - a.count);
  return {
    nodeCount: nodes.length,
    labels,
    cells,
    clusters,
  };
}

function edgeWeightSeries(history: NetworkState[]) {
  return history.map((sample) =>
    average(neuronEdges(sample).map((edge) => Math.abs(numberValue(edge.data.weight)))),
  );
}

function connectivityByLabel(history: NetworkState[]) {
  const byLabel = new Map<string, number[]>();
  for (const sample of history) {
    const label = labelKey(sample.stimulus);
    if (!label) continue;
    const weights = neuronEdges(sample).map((edge) => Math.abs(numberValue(edge.data.weight)));
    const row = byLabel.get(label) ?? [];
    row.push(average(weights));
    byLabel.set(label, row);
  }
  return Array.from(byLabel.entries()).map(([label, values]) => ({
    label,
    meanWeight: average(values),
  }));
}

function buildOutputSeries(samples: NetworkState[], ids: string[]) {
  const out = new Map(ids.map((id) => [id, [] as number[]]));
  for (const sample of samples) {
    const byId = new Map(neuronNodes(sample).map((node) => [node.data.id, node]));
    for (const id of ids) {
      out.get(id)?.push(numberValue(byId.get(id)?.data.output));
    }
  }
  return out;
}

function firstPrediction(stimulus?: StimulusMetadata) {
  if (stimulus?.predictions?.length) return stimulus.predictions[0];
  if (stimulus?.predicted_label !== undefined && stimulus.predicted_label !== null) {
    return { label: stimulus.predicted_label, confidence: stimulus.confidence };
  }
  return null;
}

function predictionLabel(stimulus?: StimulusMetadata) {
  const pred = firstPrediction(stimulus);
  if (!pred || pred.label == null) return "-";
  const conf = pred.confidence == null ? "" : ` ${formatPercent(numberValue(pred.confidence))}`;
  return `${String(pred.label)}${conf}`;
}

function labelKey(stimulus?: StimulusMetadata): string | null {
  if (!stimulus?.active) return null;
  if (stimulus.label !== undefined && stimulus.label !== null && stimulus.label !== "") {
    return String(stimulus.label);
  }
  if (stimulus.class_name) return String(stimulus.class_name);
  return null;
}

function getLayerKey(data: GraphNodeData) {
  if (data.layer_key) return String(data.layer_key);
  if (data.layer !== undefined && data.layer !== null) return String(data.layer);
  return "unknown";
}

function layerSortKey(key: string) {
  const parsed = Number(key);
  return Number.isFinite(parsed) ? parsed : 1e9;
}

function numberValue(value: unknown, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function average(values: number[]) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
}

function variance(values: number[]) {
  const mean = average(values);
  return average(values.map((value) => (value - mean) ** 2));
}

function pearson(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  if (n < 3) return 0;
  const aa = a.slice(0, n);
  const bb = b.slice(0, n);
  const ma = average(aa);
  const mb = average(bb);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let index = 0; index < n; index += 1) {
    const x = aa[index] - ma;
    const y = bb[index] - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  const den = Math.sqrt(da * db);
  return den > 1e-9 ? num / den : 0;
}

function cosine(a: number[], b: number[]) {
  let dot = 0;
  let da = 0;
  let db = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    dot += a[index] * b[index];
    da += a[index] ** 2;
    db += b[index] ** 2;
  }
  const den = Math.sqrt(da * db);
  return den > 1e-9 ? dot / den : 0;
}

function normalizeRows(rows: number[][]) {
  if (rows.length === 0) return [];
  const dims = rows[0].length;
  const ranges = Array.from({ length: dims }, (_, dim) => {
    const values = rows.map((row) => row[dim]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { min, span: Math.max(1e-9, max - min) };
  });
  return rows.map((row) => row.map((value, dim) => (value - ranges[dim].min) / ranges[dim].span));
}

function chooseK(rows: number[][], maxK: number) {
  if (rows.length <= 2) return Math.max(1, rows.length);
  const distinct = new Set(rows.map((row) => row.map((value) => value.toFixed(4)).join("|"))).size;
  return clamp(Math.round(Math.sqrt(distinct / 2)) + 1, 2, Math.min(maxK, distinct, rows.length));
}

function simpleKMeans(rows: number[][], k: number) {
  const clusterCount = clamp(Math.floor(k), 1, Math.max(1, Math.min(rows.length, k)));
  if (rows.length === 0) return { assignments: [] as number[], centroids: [] as number[][] };
  let centroids = Array.from({ length: clusterCount }, (_, index) => [
    ...rows[Math.round((index / Math.max(1, clusterCount - 1)) * (rows.length - 1))],
  ]);
  const assignments = Array(rows.length).fill(0);
  for (let iteration = 0; iteration < 14; iteration += 1) {
    rows.forEach((row, rowIndex) => {
      assignments[rowIndex] = nearest(row, centroids);
    });
    centroids = recompute(rows, assignments, clusterCount, rows[0].length, iteration);
  }
  return { assignments, centroids };
}

function nearest(row: number[], centroids: number[][]) {
  let best = 0;
  let bestDistance = Infinity;
  centroids.forEach((centroid, index) => {
    const distance = squaredDistance(row, centroid);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

function recompute(
  rows: number[][],
  assignments: number[],
  k: number,
  dims: number,
  iteration: number,
) {
  const sums = Array.from({ length: k }, () => Array(dims).fill(0));
  const counts = Array(k).fill(0);
  rows.forEach((row, index) => {
    const cluster = assignments[index];
    counts[cluster] += 1;
    for (let dim = 0; dim < dims; dim += 1) sums[cluster][dim] += row[dim];
  });
  return sums.map((sum, index) => {
    if (!counts[index]) return [...rows[(iteration + index * 5) % rows.length]];
    return sum.map((value) => value / counts[index]);
  });
}

function squaredDistance(a: number[], b: number[]) {
  let out = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    out += (a[index] - b[index]) ** 2;
  }
  return out;
}

function clusterCounts(assignments: number[]) {
  const counts = new Map<number, number>();
  for (const assignment of assignments) counts.set(assignment, (counts.get(assignment) ?? 0) + 1);
  return Array.from(counts.entries())
    .map(([id, count]) => ({ id: `C${id + 1}`, count }))
    .sort((a, b) => b.count - a.count);
}

class UnionFind {
  private parent = new Map<string, string>();

  constructor(ids: string[]) {
    for (const id of ids) this.parent.set(id, id);
  }

  find(id: string): string {
    const parent = this.parent.get(id) ?? id;
    if (parent === id) return id;
    const root = this.find(parent);
    this.parent.set(id, root);
    return root;
  }

  union(a: string, b: string) {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent.set(rb, ra);
  }
}

function bounds2D(points: Point2D[]) {
  const xs = points.map((point) => point.x).filter(Number.isFinite);
  const ys = points.map((point) => point.y).filter(Number.isFinite);
  return {
    minX: xs.length ? Math.min(...xs) : 0,
    maxX: xs.length ? Math.max(...xs) : 1,
    minY: ys.length ? Math.min(...ys) : 0,
    maxY: ys.length ? Math.max(...ys) : 1,
  };
}

function scale(value: number, min: number, max: number, outMin: number, outMax: number) {
  if (!Number.isFinite(value)) return (outMin + outMax) / 2;
  if (Math.abs(max - min) <= 1e-9) return (outMin + outMax) / 2;
  return outMin + ((value - min) / (max - min)) * (outMax - outMin);
}

function heatColor(value: number) {
  const v = clamp(Math.abs(value), 0, 1);
  const r = Math.round(18 + v * 230);
  const g = Math.round(26 + v * 96);
  const b = Math.round(42 + (1 - v) * 170);
  return `rgb(${r}, ${g}, ${b})`;
}

function colorForKey(key: string) {
  return colorForIndex(Math.abs(hash(key)) % PALETTE.length);
}

function colorForIndex(index: number) {
  return PALETTE[Math.abs(index) % PALETTE.length];
}

function hash(value: string) {
  let out = 0;
  for (let index = 0; index < value.length; index += 1) {
    out = (out * 31 + value.charCodeAt(index)) | 0;
  }
  return out;
}

function unique<T>(values: T[]) {
  return Array.from(new Set(values));
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function compactLabel(value: string) {
  return value.length > 8 ? `${value.slice(0, 7)}…` : value;
}

function compactId(value: string) {
  return value.replace(/^neuron_/, "").slice(0, 8);
}

function formatText(value: unknown) {
  if (value === undefined || value === null || value === "") return "-";
  return String(value);
}

function formatScalar(value: number) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}
