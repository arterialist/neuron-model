import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { RotateCcw } from "lucide-react";
import * as THREE from "three";
import type { GraphNode, GraphNodeData, NetworkState } from "./types";

type PopulationMode = "all" | "selected" | "layer";

interface DynamicsPanelProps {
  state: NetworkState | null;
  history: NetworkState[];
  selectedId: string | null;
  onResetHistory: () => void;
}

interface LayerOption {
  key: string;
  label: string;
  count: number;
}

interface Population {
  mode: PopulationMode;
  label: string;
  ids: Set<string> | null;
  count: number;
  ready: boolean;
}

interface DynamicsSample {
  tick: number;
  count: number;
  meanS: number;
  meanF: number;
  meanTRef: number;
  meanO: number;
  activeRatio: number;
  varianceS: number;
  energy: number;
}

interface LayerFreeEnergySeries {
  key: string;
  label: string;
  count: number;
  samples: Array<{ tick: number; value: number }>;
}

type MetricKey =
  | "meanS"
  | "meanF"
  | "meanTRef"
  | "meanO"
  | "activeRatio"
  | "varianceS"
  | "energy";

const METRIC_DEFS: Array<{ key: MetricKey; label: string }> = [
  { key: "meanS", label: "S mean" },
  { key: "meanF", label: "F_avg mean" },
  { key: "meanTRef", label: "t_ref mean" },
  { key: "meanO", label: "Output mean" },
  { key: "activeRatio", label: "Active ratio" },
  { key: "varianceS", label: "S variance" },
  { key: "energy", label: "Energy" },
];

export function DynamicsPanel({
  state,
  history,
  selectedId,
  onResetHistory,
}: DynamicsPanelProps) {
  const [mode, setMode] = useState<PopulationMode>("all");
  const layerOptions = useMemo(() => getLayerOptions(state), [state]);
  const [layerKey, setLayerKey] = useState("");
  const [xMetric, setXMetric] = useState<MetricKey>("meanS");
  const [yMetric, setYMetric] = useState<MetricKey>("meanF");
  const [zMetric, setZMetric] = useState<MetricKey>("energy");

  useEffect(() => {
    if (layerOptions.length === 0) return;
    if (!layerOptions.some((option) => option.key === layerKey)) {
      setLayerKey(layerOptions[0].key);
    }
  }, [layerKey, layerOptions]);

  const population = useMemo(
    () => resolvePopulation(state, mode, selectedId, layerKey),
    [layerKey, mode, selectedId, state],
  );
  const samples = useMemo(
    () => buildSamples(history, population),
    [history, population],
  );
  const analysis = useMemo(() => analyzeSamples(samples), [samples]);

  return (
    <section className="panelSection dynamicsPanel">
      <h2>Dynamics</h2>
      <FreeEnergyGraph history={history} />
      <LayerFreeEnergyGraphs history={history} layers={layerOptions} />
      <div className="segmented dynamicsModes">
        <button
          className={clsx(mode === "all" && "active")}
          onClick={() => setMode("all")}
        >
          All
        </button>
        <button
          className={clsx(mode === "selected" && "active")}
          onClick={() => setMode("selected")}
        >
          Selected
        </button>
        <button
          className={clsx(mode === "layer" && "active")}
          onClick={() => setMode("layer")}
        >
          Layer
        </button>
      </div>

      {mode === "layer" && (
        <label className="field dynamicsLayerField">
          <span>layer</span>
          <select
            data-testid="layer-select"
            value={layerKey}
            onChange={(event) => setLayerKey(event.target.value)}
          >
            {layerOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label} ({option.count})
              </option>
            ))}
          </select>
        </label>
      )}

      <div className="axisGrid">
        <MetricSelect axis="X" value={xMetric} onChange={setXMetric} />
        <MetricSelect axis="Y" value={yMetric} onChange={setYMetric} />
        <MetricSelect axis="Z" value={zMetric} onChange={setZMetric} />
      </div>

      <AttractorCanvas
        samples={samples}
        ready={population.ready}
        xMetric={xMetric}
        yMetric={yMetric}
        zMetric={zMetric}
      />

      <div className="regimeHeader">
        <strong>{analysis.regime}</strong>
        <span>{population.label}</span>
      </div>
      <div className="statsGrid dynamicsStats">
        <DynMetric label="pop" value={String(population.count)} />
        <DynMetric label="history" value={String(samples.length)} />
        <DynMetric label="speed" value={analysis.speed.toFixed(3)} />
        <DynMetric label="recur" value={formatPercent(analysis.recurrence)} />
        <DynMetric label="stable" value={formatPercent(analysis.stability)} />
        <DynMetric label="var S" value={analysis.variance.toFixed(3)} />
      </div>
      <button className="iconButton full dynamicsReset" onClick={onResetHistory}>
        <RotateCcw size={16} />
        <span>Reset History</span>
      </button>
    </section>
  );
}

function FreeEnergyGraph({ history }: { history: NetworkState[] }) {
  const samples = useMemo(() => buildFreeEnergySamples(history), [history]);
  const latest = samples[samples.length - 1]?.value ?? 0;
  const path = useMemo(() => sparkPath(samples, 282, 76), [samples]);
  const area = path ? `${path} L 282 76 L 0 76 Z` : "";

  return (
    <div className="freeEnergyPanel" data-testid="free-energy-graph">
      <div className="freeEnergyHeader">
        <span>Free Energy</span>
        <strong>{formatScalar(latest)}</strong>
      </div>
      <svg className="freeEnergyGraph" viewBox="0 0 282 76" aria-hidden="true">
        <path className="freeEnergyArea" d={area} />
        <path className="freeEnergyLine" d={path} />
      </svg>
    </div>
  );
}

function LayerFreeEnergyGraphs({
  history,
  layers,
}: {
  history: NetworkState[];
  layers: LayerOption[];
}) {
  const series = useMemo(
    () => buildLayerFreeEnergySeries(history, layers),
    [history, layers],
  );
  const bounds = useMemo(() => layerSeriesBounds(series), [series]);

  if (series.length === 0) {
    return null;
  }

  return (
    <section className="layerEnergyPanel" data-testid="layer-free-energy-graphs">
      <div className="layerEnergyTitle">
        <span>Layer Free Energy</span>
        <strong>{series.length}</strong>
      </div>
      <div className="layerEnergyList">
        {series.map((item) => (
          <LayerFreeEnergyRow key={item.key} item={item} bounds={bounds} />
        ))}
      </div>
    </section>
  );
}

function LayerFreeEnergyRow({
  item,
  bounds,
}: {
  item: LayerFreeEnergySeries;
  bounds: { min: number; max: number };
}) {
  const latest = item.samples[item.samples.length - 1]?.value ?? 0;
  const path = useMemo(
    () => sparkPath(item.samples, 162, 34, bounds),
    [bounds, item.samples],
  );
  const area = path ? `${path} L 162 34 L 0 34 Z` : "";

  return (
    <div className="layerEnergyRow" data-layer-key={item.key}>
      <div className="layerEnergyMeta">
        <span>{item.label}</span>
        <small>{item.count}n</small>
      </div>
      <svg className="layerEnergyGraph" viewBox="0 0 162 34" aria-hidden="true">
        <path className="layerEnergyArea" d={area} />
        <path className="layerEnergyLine" d={path} />
      </svg>
      <strong>{formatScalar(latest)}</strong>
    </div>
  );
}

function AttractorCanvas({
  samples,
  ready,
  xMetric,
  yMetric,
  zMetric,
}: {
  samples: DynamicsSample[];
  ready: boolean;
  xMetric: MetricKey;
  yMetric: MetricKey;
  zMetric: MetricKey;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const rotationRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const sceneRef = useRef<ThreeSceneHandles | null>(null);
  const sceneData = useMemo(
    () => buildSceneData(samples, ready, xMetric, yMetric, zMetric),
    [ready, samples, xMetric, yMetric, zMetric],
  );
  const sceneDataRef = useRef(sceneData);
  sceneDataRef.current = sceneData;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(2.6, 1.9, 3.3);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setClearColor(0x080b10, 1);
    renderer.domElement.className = "landscapeCanvas dynamicsCanvas3d";
    renderer.domElement.dataset.testid = "dynamics-3d";
    host.appendChild(renderer.domElement);

    const world = new THREE.Group();
    scene.add(world);
    const handles: ThreeSceneHandles = { scene, camera, renderer, world };
    sceneRef.current = handles;
    updateThreeScene(handles, sceneDataRef.current);

    const resize = () => {
      const rect = host.getBoundingClientRect();
      const width = Math.max(220, Math.floor(rect.width));
      const height = 210;
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    let raf = 0;
    const animate = () => {
      world.rotation.x = -0.46 + rotationRef.current.x;
      world.rotation.y = 0.72 + rotationRef.current.y;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      clearGroup(world);
      renderer.dispose();
      renderer.domElement.remove();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (sceneRef.current) {
      updateThreeScene(sceneRef.current, sceneData);
    }
  }, [sceneData]);

  return (
    <div
      ref={hostRef}
      className="landscapeFrame landscapeFrame3d"
      data-x-metric={xMetric}
      data-y-metric={yMetric}
      data-z-metric={zMetric}
      onPointerDown={(event) => {
        try {
          event.currentTarget.setPointerCapture(event.pointerId);
        } catch {
          // Pointer capture can fail in older embedded browsers.
        }
        dragRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
      }}
      onPointerMove={(event) => {
        const drag = dragRef.current;
        if (!drag || drag.id !== event.pointerId) return;
        rotationRef.current.y += (event.clientX - drag.x) * 0.01;
        rotationRef.current.x += (event.clientY - drag.y) * 0.008;
        rotationRef.current.x = clamp(rotationRef.current.x, -1.1, 1.1);
        drag.x = event.clientX;
        drag.y = event.clientY;
      }}
      onPointerUp={(event) => {
        if (dragRef.current?.id === event.pointerId) dragRef.current = null;
      }}
      onPointerCancel={() => {
        dragRef.current = null;
      }}
      onDoubleClick={() => {
        rotationRef.current = { x: 0, y: 0 };
      }}
    >
      <div className="axisLegend">
        <span>X {metricLabel(xMetric)}</span>
        <span>Y {metricLabel(yMetric)}</span>
        <span>Z {metricLabel(zMetric)}</span>
      </div>
      {!sceneData.ready && (
        <div className="empty3dText">
          {ready ? "Awaiting trajectory" : "No population"}
        </div>
      )}
    </div>
  );
}

function MetricSelect({
  axis,
  value,
  onChange,
}: {
  axis: "X" | "Y" | "Z";
  value: MetricKey;
  onChange: (value: MetricKey) => void;
}) {
  return (
    <label className="field axisField">
      <span>{axis} axis</span>
      <select
        data-testid={`axis-${axis.toLowerCase()}`}
        value={value}
        onChange={(event) => onChange(event.target.value as MetricKey)}
      >
        {METRIC_DEFS.map((metric) => (
          <option key={metric.key} value={metric.key}>
            {metric.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function buildFreeEnergySamples(history: NetworkState[]) {
  return history
    .map((snapshot) => ({
      tick: snapshot.current_tick,
      value: stateFreeEnergy(snapshot),
    }))
    .filter((sample) => Number.isFinite(sample.value))
    .slice(-180);
}

function stateFreeEnergy(snapshot: NetworkState) {
  const stats = snapshot.statistics;
  if (typeof stats.free_energy === "number" && Number.isFinite(stats.free_energy)) {
    return stats.free_energy;
  }
  if (typeof stats.state_energy === "number" && Number.isFinite(stats.state_energy)) {
    return stats.state_energy;
  }
  const activeRatio = stats.num_neurons > 0 ? stats.active_neurons / stats.num_neurons : 0;
  return Math.sqrt(
    stats.avg_potential * stats.avg_potential +
      stats.avg_firing_rate * stats.avg_firing_rate +
      activeRatio * activeRatio,
  );
}

function sparkPath(
  samples: Array<{ tick: number; value: number }>,
  width: number,
  height: number,
  forcedBounds?: { min: number; max: number },
) {
  if (samples.length === 0) return "";
  if (samples.length === 1) {
    const y = height / 2;
    return `M 0 ${y.toFixed(2)} L ${width} ${y.toFixed(2)}`;
  }
  const values = samples.map((sample) => sample.value);
  const min = forcedBounds?.min ?? Math.min(...values);
  const max = forcedBounds?.max ?? Math.max(...values);
  const span = Math.max(1e-9, max - min);
  return samples
    .map((sample, index) => {
      const x = (index / (samples.length - 1)) * width;
      const y = height - ((sample.value - min) / span) * (height - 8) - 4;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function buildLayerFreeEnergySeries(
  history: NetworkState[],
  layers: LayerOption[],
): LayerFreeEnergySeries[] {
  return layers
    .map((layer) => {
      const samples = history
        .map((snapshot) => {
          const nodes = neuronNodes(snapshot).filter(
            (node) => getLayerKey(node.data) === layer.key,
          );
          if (nodes.length === 0) return null;
          return {
            tick: snapshot.current_tick,
            value: samplePopulation(snapshot.current_tick, nodes).energy,
          };
        })
        .filter((sample): sample is { tick: number; value: number } => {
          return !!sample && Number.isFinite(sample.value);
        })
        .slice(-180);
      return { ...layer, samples };
    })
    .filter((layer) => layer.samples.length > 0);
}

function layerSeriesBounds(series: LayerFreeEnergySeries[]) {
  const values = series.flatMap((item) => item.samples.map((sample) => sample.value));
  if (values.length === 0) return { min: 0, max: 1 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-6, max - min);
  return { min: min - span * 0.08, max: max + span * 0.08 };
}

interface ThreeSceneHandles {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  world: THREE.Group;
}

interface ScenePoint {
  x: number;
  y: number;
  z: number;
  heat: number;
}

interface SceneData {
  ready: boolean;
  points: ScenePoint[];
  latest: ScenePoint | null;
}

function updateThreeScene(handles: ThreeSceneHandles, data: SceneData) {
  clearGroup(handles.world);
  handles.world.add(makeAxisBox());
  if (!data.ready || data.points.length < 2) return;

  const positions = new Float32Array(data.points.length * 3);
  const colors = new Float32Array(data.points.length * 3);
  data.points.forEach((point, index) => {
    positions[index * 3] = point.x;
    positions[index * 3 + 1] = point.y;
    positions[index * 3 + 2] = point.z;
    const color = heatThreeColor(point.heat);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  });

  const pointGeometry = new THREE.BufferGeometry();
  pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  pointGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const pointMaterial = new THREE.PointsMaterial({
    size: 0.055,
    vertexColors: true,
    sizeAttenuation: true,
  });
  handles.world.add(new THREE.Points(pointGeometry, pointMaterial));

  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0xe8edf4,
    transparent: true,
    opacity: 0.74,
  });
  handles.world.add(new THREE.Line(lineGeometry, lineMaterial));

  if (data.latest) {
    const latestGeometry = new THREE.SphereGeometry(0.07, 18, 12);
    const latestMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const latestMesh = new THREE.Mesh(latestGeometry, latestMaterial);
    latestMesh.position.set(data.latest.x, data.latest.y, data.latest.z);
    handles.world.add(latestMesh);
  }
}

function makeAxisBox() {
  const group = new THREE.Group();
  const gridVertices: number[] = [];
  const ticks = [-1, -0.5, 0, 0.5, 1];
  for (const tick of ticks) {
    gridVertices.push(-1, -1, tick, 1, -1, tick);
    gridVertices.push(tick, -1, -1, tick, -1, 1);
    gridVertices.push(-1, tick, -1, 1, tick, -1);
    gridVertices.push(tick, -1, -1, tick, 1, -1);
  }
  const gridGeometry = new THREE.BufferGeometry();
  gridGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(gridVertices, 3),
  );
  group.add(
    new THREE.LineSegments(
      gridGeometry,
      new THREE.LineBasicMaterial({
        color: 0x2b3440,
        transparent: true,
        opacity: 0.55,
      }),
    ),
  );

  group.add(axisLine(-1.08, 0, 0, 1.08, 0, 0, 0x74d4ff));
  group.add(axisLine(0, -1.08, 0, 0, 1.08, 0, 0x5ee090));
  group.add(axisLine(0, 0, -1.08, 0, 0, 1.08, 0xfb923c));
  return group;
}

function axisLine(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  color: number,
) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([ax, ay, az, bx, by, bz], 3),
  );
  return new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }),
  );
}

function buildSceneData(
  samples: DynamicsSample[],
  ready: boolean,
  xMetric: MetricKey,
  yMetric: MetricKey,
  zMetric: MetricKey,
): SceneData {
  if (!ready || samples.length < 2) {
    return { ready: false, points: [], latest: null };
  }

  const bounds = {
    x: metricBounds(samples, xMetric),
    y: metricBounds(samples, yMetric),
    z: metricBounds(samples, zMetric),
  };
  const rawPoints = samples.map((sample) => ({
    x: normalizeMetric(sample[xMetric], bounds.x),
    y: normalizeMetric(sample[yMetric], bounds.y),
    z: normalizeMetric(sample[zMetric], bounds.z),
  }));
  const density = densityMap(rawPoints, 10);
  const points = rawPoints.map((point) => ({
    ...point,
    heat: density.get(densityKey(point, 10)) ?? 0,
  }));
  const maxHeat = Math.max(1, ...points.map((point) => point.heat));
  const scaled = points.map((point) => ({ ...point, heat: point.heat / maxHeat }));
  return {
    ready: true,
    points: scaled,
    latest: scaled[scaled.length - 1] ?? null,
  };
}

function metricBounds(samples: DynamicsSample[], metric: MetricKey) {
  const values = samples.map((sample) => sample[metric]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(0.05, max - min);
  return { min: min - span * 0.12, max: max + span * 0.12 };
}

function normalizeMetric(value: number, bounds: { min: number; max: number }) {
  return ((value - bounds.min) / (bounds.max - bounds.min)) * 2 - 1;
}

function densityMap(points: Array<{ x: number; y: number; z: number }>, bins: number) {
  const density = new Map<string, number>();
  for (const point of points) {
    const key = densityKey(point, bins);
    density.set(key, (density.get(key) ?? 0) + 1);
  }
  return density;
}

function densityKey(point: { x: number; y: number; z: number }, bins: number) {
  const bx = clamp(Math.floor(((point.x + 1) / 2) * bins), 0, bins - 1);
  const by = clamp(Math.floor(((point.y + 1) / 2) * bins), 0, bins - 1);
  const bz = clamp(Math.floor(((point.z + 1) / 2) * bins), 0, bins - 1);
  return `${bx}:${by}:${bz}`;
}

function heatThreeColor(value: number) {
  if (value > 0.66) return new THREE.Color(0xf87171);
  if (value > 0.33) return new THREE.Color(0xfb923c);
  return new THREE.Color(0x74d4ff);
}

function metricLabel(metric: MetricKey) {
  return METRIC_DEFS.find((item) => item.key === metric)?.label ?? metric;
}

function clearGroup(group: THREE.Group) {
  while (group.children.length) {
    const child = group.children.pop();
    if (child) {
      disposeObject(child);
    }
  }
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const maybeMesh = child as THREE.Object3D & {
      geometry?: THREE.BufferGeometry;
      material?: THREE.Material | THREE.Material[];
    };
    maybeMesh.geometry?.dispose();
    if (Array.isArray(maybeMesh.material)) {
      maybeMesh.material.forEach((material) => material.dispose());
    } else {
      maybeMesh.material?.dispose();
    }
  });
}

function drawLandscape(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  samples: DynamicsSample[],
  ready: boolean,
) {
  ctx.fillStyle = "#080b10";
  ctx.fillRect(0, 0, width, height);

  const padL = 34;
  const padR = 12;
  const padT = 12;
  const padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  drawLandscapeGrid(ctx, padL, padT, plotW, plotH);

  if (!ready || samples.length < 2) {
    ctx.fillStyle = "rgba(198, 208, 220, 0.72)";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(ready ? "Awaiting trajectory" : "No population", width / 2, height / 2);
    drawAxisLabels(ctx, width, height);
    return;
  }

  const bounds = getBounds(samples);
  const bins = buildBins(samples, bounds, 26);
  const maxBin = Math.max(1, ...bins.grid);
  const cellW = plotW / bins.size;
  const cellH = plotH / bins.size;

  for (let y = 0; y < bins.size; y += 1) {
    for (let x = 0; x < bins.size; x += 1) {
      const count = bins.grid[y * bins.size + x];
      if (count === 0) continue;
      const heat = count / maxBin;
      ctx.fillStyle = heatColor(heat);
      ctx.fillRect(
        padL + x * cellW,
        padT + (bins.size - 1 - y) * cellH,
        Math.ceil(cellW) + 1,
        Math.ceil(cellH) + 1,
      );
    }
  }

  ctx.beginPath();
  samples.forEach((sample, index) => {
    const [x, y] = projectSample(sample, bounds, padL, padT, plotW, plotH);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "rgba(230, 237, 244, 0.58)";
  ctx.lineWidth = 1.4;
  ctx.stroke();

  const newest = samples[samples.length - 1];
  const [latestX, latestY] = projectSample(newest, bounds, padL, padT, plotW, plotH);
  ctx.fillStyle = "#f8fafc";
  ctx.strokeStyle = "rgba(116, 212, 255, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(latestX, latestY, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  if (bins.peak) {
    const px = padL + (bins.peak.x + 0.5) * cellW;
    const py = padT + (bins.size - 0.5 - bins.peak.y) * cellH;
    ctx.strokeStyle = "rgba(251, 146, 60, 0.92)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.stroke();
  }

  drawAxisLabels(ctx, width, height);
}

function drawLandscapeGrid(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
) {
  ctx.strokeStyle = "rgba(255,255,255,0.055)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const gx = x + (i / 4) * w;
    const gy = y + (i / 4) * h;
    ctx.moveTo(gx, y);
    ctx.lineTo(gx, y + h);
    ctx.moveTo(x, gy);
    ctx.lineTo(x + w, gy);
  }
  ctx.stroke();
}

function drawAxisLabels(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.fillStyle = "rgba(198, 208, 220, 0.62)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText("F_avg", 29, 20);
  ctx.textAlign = "right";
  ctx.fillText("S mean", width - 12, height - 7);
}

function getBounds(samples: DynamicsSample[]) {
  const xs = samples.map((sample) => sample.meanS);
  const ys = samples.map((sample) => sample.meanF);
  return expandBounds(
    Math.min(...xs),
    Math.max(...xs),
    Math.min(...ys),
    Math.max(...ys),
  );
}

function expandBounds(minX: number, maxX: number, minY: number, maxY: number) {
  const spanX = Math.max(0.05, maxX - minX);
  const spanY = Math.max(0.05, maxY - minY);
  return {
    minX: minX - spanX * 0.12,
    maxX: maxX + spanX * 0.12,
    minY: minY - spanY * 0.12,
    maxY: maxY + spanY * 0.12,
  };
}

function projectSample(
  sample: DynamicsSample,
  bounds: ReturnType<typeof expandBounds>,
  x: number,
  y: number,
  w: number,
  h: number,
): [number, number] {
  const px = x + ((sample.meanS - bounds.minX) / (bounds.maxX - bounds.minX)) * w;
  const py = y + h - ((sample.meanF - bounds.minY) / (bounds.maxY - bounds.minY)) * h;
  return [px, py];
}

function buildBins(
  samples: DynamicsSample[],
  bounds: ReturnType<typeof expandBounds>,
  size: number,
) {
  const grid = new Array(size * size).fill(0);
  let peak: { x: number; y: number } | null = null;
  let peakValue = 0;
  for (const sample of samples) {
    const x = clamp(
      Math.floor(((sample.meanS - bounds.minX) / (bounds.maxX - bounds.minX)) * size),
      0,
      size - 1,
    );
    const y = clamp(
      Math.floor(((sample.meanF - bounds.minY) / (bounds.maxY - bounds.minY)) * size),
      0,
      size - 1,
    );
    const index = y * size + x;
    grid[index] += 1;
    if (grid[index] > peakValue) {
      peakValue = grid[index];
      peak = { x, y };
    }
  }
  return { grid, peak, size };
}

function heatColor(value: number) {
  const alpha = 0.12 + value * 0.72;
  if (value > 0.66) return `rgba(248, 113, 113, ${alpha})`;
  if (value > 0.33) return `rgba(251, 146, 60, ${alpha})`;
  return `rgba(116, 212, 255, ${alpha})`;
}

function resolvePopulation(
  state: NetworkState | null,
  mode: PopulationMode,
  selectedId: string | null,
  layerKey: string,
): Population {
  const nodes = neuronNodes(state);
  if (mode === "all") {
    return { mode, label: "all neurons", ids: null, count: nodes.length, ready: nodes.length > 0 };
  }

  if (mode === "selected") {
    const selected = selectedId
      ? nodes.find((node) => node.data.id === selectedId)
      : null;
    const ids = new Set<string>();
    if (selected) ids.add(selected.data.id);
    return {
      mode,
      label: selected ? selectedLabel(selected.data) : "selected",
      ids,
      count: ids.size,
      ready: ids.size > 0,
    };
  }

  const ids = new Set(
    nodes
      .filter((node) => getLayerKey(node.data) === layerKey)
      .map((node) => node.data.id),
  );
  const option = getLayerOptions(state).find((layer) => layer.key === layerKey);
  return {
    mode,
    label: option?.label ?? "layer",
    ids,
    count: ids.size,
    ready: ids.size > 0,
  };
}

function buildSamples(history: NetworkState[], population: Population): DynamicsSample[] {
  if (!population.ready) return [];
  const out: DynamicsSample[] = [];
  for (const snapshot of history) {
    const nodes = population.ids
      ? neuronNodes(snapshot).filter((node) => population.ids?.has(node.data.id))
      : neuronNodes(snapshot);
    if (nodes.length === 0) continue;
    out.push(samplePopulation(snapshot.current_tick, nodes));
  }
  return out;
}

function samplePopulation(tick: number, nodes: GraphNode[]): DynamicsSample {
  let sumS = 0;
  let sumS2 = 0;
  let sumF = 0;
  let sumTRef = 0;
  let sumO = 0;
  let active = 0;
  for (const node of nodes) {
    const s = numberValue(node.data.membrane_potential);
    const f = numberValue(node.data.firing_rate);
    const tRef = numberValue(node.data.t_ref);
    const o = numberValue(node.data.output);
    sumS += s;
    sumS2 += s * s;
    sumF += f;
    sumTRef += tRef;
    sumO += o;
    if (Math.abs(o) > 1e-9 || f > 1e-6) active += 1;
  }
  const count = Math.max(1, nodes.length);
  const meanS = sumS / count;
  const meanF = sumF / count;
  const meanTRef = sumTRef / count;
  const meanO = sumO / count;
  const varianceS = Math.max(0, sumS2 / count - meanS * meanS);
  return {
    tick,
    count,
    meanS,
    meanF,
    meanTRef,
    meanO,
    activeRatio: active / count,
    varianceS,
    energy: Math.sqrt(sumS2 / count + meanF * meanF + meanO * meanO),
  };
}

function analyzeSamples(samples: DynamicsSample[]) {
  if (samples.length < 2) {
    return {
      regime: "insufficient",
      speed: 0,
      recurrence: 0,
      stability: 0,
      variance: 0,
    };
  }

  const window = samples.slice(-96);
  const bounds = getBounds(window);
  const scaleX = bounds.maxX - bounds.minX;
  const scaleY = bounds.maxY - bounds.minY;
  const deltas: number[] = [];
  const signs: number[] = [];
  for (let i = 1; i < window.length; i += 1) {
    const dx = (window[i].meanS - window[i - 1].meanS) / scaleX;
    const dy = (window[i].meanF - window[i - 1].meanF) / scaleY;
    deltas.push(Math.sqrt(dx * dx + dy * dy));
    signs.push(Math.sign(dx || dy));
  }
  const speed = average(deltas);
  const variance = average(window.map((sample) => sample.varianceS));
  const bins = buildBins(window, bounds, 14);
  const recurrence = Math.max(...bins.grid) / window.length;
  let signChanges = 0;
  for (let i = 1; i < signs.length; i += 1) {
    if (signs[i] !== 0 && signs[i - 1] !== 0 && signs[i] !== signs[i - 1]) {
      signChanges += 1;
    }
  }
  const oscillation = signs.length > 0 ? signChanges / signs.length : 0;
  const latest = window[window.length - 1];
  const stability = clamp(1 - speed * 3.4 - variance * 1.8, 0, 1);

  let regime = "settling";
  if (latest.activeRatio < 0.02 && latest.meanF < 0.01 && Math.abs(latest.meanS) < 0.03) {
    regime = "quiescent";
  } else if (speed < 0.006 && recurrence > 0.35) {
    regime = "fixed point";
  } else if (recurrence > 0.32 && speed < 0.04) {
    regime = "attractor basin";
  } else if (oscillation > 0.38 && speed > 0.006) {
    regime = "limit cycle";
  } else if (speed > 0.12 && stability < 0.45) {
    regime = "transient";
  }

  return { regime, speed, recurrence, stability, variance };
}

function getLayerOptions(state: NetworkState | null): LayerOption[] {
  const byLayer = new Map<string, LayerOption>();
  for (const node of neuronNodes(state)) {
    const key = getLayerKey(node.data);
    const current = byLayer.get(key);
    if (current) {
      current.count += 1;
    } else {
      byLayer.set(key, {
        key,
        label: getLayerLabel(node.data),
        count: 1,
      });
    }
  }
  return Array.from(byLayer.values()).sort((a, b) =>
    layerSortValue(a.key, a.label).localeCompare(layerSortValue(b.key, b.label)),
  );
}

function neuronNodes(state: NetworkState | null) {
  return (state?.elements.nodes ?? []).filter((node) => node.data.type === "neuron");
}

function getLayerKey(data: GraphNodeData) {
  if (data.layer_key) return data.layer_key;
  if (data.layer == null) return "unlayered";
  return String(data.layer);
}

function getLayerLabel(data: GraphNodeData) {
  const key = getLayerKey(data);
  if (data.layer_name && data.layer_name !== "unknown") return data.layer_name;
  return key === "unlayered" ? "unlayered" : `layer ${key}`;
}

function selectedLabel(data: GraphNodeData) {
  return `selected ${data.label ?? data.neuron_id ?? data.id}`;
}

function layerSortValue(key: string, label: string) {
  const numeric = Number(key);
  if (Number.isFinite(numeric)) return `${String(numeric).padStart(8, "0")} ${label}`;
  return `zz ${label}`;
}

function DynMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function numberValue(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function average(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatPercent(value: number) {
  return `${Math.round(clamp(value, 0, 1) * 100)}%`;
}

function formatScalar(value: number) {
  if (!Number.isFinite(value)) return "0.000";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
