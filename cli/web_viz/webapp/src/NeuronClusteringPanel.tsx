import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import * as THREE from "three";
import type { GraphNode, GraphNodeData, NetworkState } from "./types";

type ClusterScope = "all" | "layer";
type ClusterViewMode = "2d" | "3d";
type ClusterCountMode = "auto" | "manual";

interface LayerOption {
  key: string;
  label: string;
  count: number;
}

interface FeatureDef {
  key: string;
  label: string;
  group: string;
  value: (node: GraphNode) => number;
}

interface ClusterPoint {
  id: string;
  label: string;
  node: GraphNode;
  raw: number[];
  normalized: number[];
  cluster: number;
}

interface ClusterSummary {
  id: number;
  count: number;
  centroid: number[];
}

interface KMeansResult {
  ready: boolean;
  mode: ClusterCountMode;
  features: string[];
  dimension: number;
  requestedClusters: number;
  points: ClusterPoint[];
  centroids: number[][];
  clusters: ClusterSummary[];
  inertia: number;
  autoScore: number | null;
}

interface NeuronClusteringPanelProps {
  state: NetworkState | null;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

const CLUSTER_COLORS = [
  "#74d4ff",
  "#f87171",
  "#5ee090",
  "#fb923c",
  "#c084fc",
  "#facc15",
  "#38bdf8",
  "#f472b6",
  "#a3e635",
  "#fb7185",
  "#60a5fa",
  "#f59e0b",
];

const DEFAULT_FEATURES = ["membrane_potential", "firing_rate", "t_ref"];
const DEFAULT_AUTO_MAX_CLUSTERS = 8;
const DEFAULT_MANUAL_CLUSTERS = 4;

const FEATURE_DEFS: FeatureDef[] = [
  {
    key: "membrane_potential",
    label: "S",
    group: "Runtime",
    value: (node) => numberValue(node.data.membrane_potential),
  },
  {
    key: "firing_rate",
    label: "F_avg",
    group: "Runtime",
    value: (node) => numberValue(node.data.firing_rate),
  },
  {
    key: "output",
    label: "O",
    group: "Runtime",
    value: (node) => numberValue(node.data.output),
  },
  {
    key: "r",
    label: "r",
    group: "Runtime",
    value: (node) => numberValue(node.data.r),
  },
  {
    key: "b",
    label: "b",
    group: "Runtime",
    value: (node) => numberValue(node.data.b),
  },
  {
    key: "t_ref",
    label: "t_ref",
    group: "Runtime",
    value: (node) => numberValue(node.data.t_ref),
  },
  {
    key: "m0",
    label: "M[0]",
    group: "Runtime",
    value: (node) => vectorValue(node.data.M_vector, 0),
  },
  {
    key: "m1",
    label: "M[1]",
    group: "Runtime",
    value: (node) => vectorValue(node.data.M_vector, 1),
  },
  {
    key: "pq_len",
    label: "pq_len",
    group: "Runtime",
    value: (node) => numberValue(node.data.pq_len),
  },
  {
    key: "synapse_count",
    label: "synapses",
    group: "Topology",
    value: (node) => countValue(node.data.synapses),
  },
  {
    key: "terminal_count",
    label: "terminals",
    group: "Topology",
    value: (node) => countValue(node.data.terminals),
  },
  {
    key: "r_base",
    label: "r_base",
    group: "Params",
    value: (node) => paramNumber(node.data, "r_base"),
  },
  {
    key: "b_base",
    label: "b_base",
    group: "Params",
    value: (node) => paramNumber(node.data, "b_base"),
  },
  {
    key: "c",
    label: "c",
    group: "Params",
    value: (node) => paramNumber(node.data, "c"),
  },
  {
    key: "lambda_param",
    label: "lambda_param",
    group: "Params",
    value: (node) => paramNumber(node.data, "lambda_param", "lambda"),
  },
  {
    key: "p",
    label: "p",
    group: "Params",
    value: (node) => paramNumber(node.data, "p"),
  },
  {
    key: "eta_post",
    label: "eta_post",
    group: "Params",
    value: (node) => paramNumber(node.data, "eta_post"),
  },
  {
    key: "eta_retro",
    label: "eta_retro",
    group: "Params",
    value: (node) => paramNumber(node.data, "eta_retro"),
  },
  {
    key: "delta_decay",
    label: "delta_decay",
    group: "Params",
    value: (node) => paramNumber(node.data, "delta_decay"),
  },
  {
    key: "beta_avg",
    label: "beta_avg",
    group: "Params",
    value: (node) => paramNumber(node.data, "beta_avg"),
  },
  {
    key: "gamma0",
    label: "gamma[0]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "gamma", 0),
  },
  {
    key: "gamma1",
    label: "gamma[1]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "gamma", 1),
  },
  {
    key: "w_r0",
    label: "w_r[0]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "w_r", 0),
  },
  {
    key: "w_r1",
    label: "w_r[1]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "w_r", 1),
  },
  {
    key: "w_b0",
    label: "w_b[0]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "w_b", 0),
  },
  {
    key: "w_b1",
    label: "w_b[1]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "w_b", 1),
  },
  {
    key: "w_tref0",
    label: "w_tref[0]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "w_tref", 0),
  },
  {
    key: "w_tref1",
    label: "w_tref[1]",
    group: "Params",
    value: (node) => paramVectorValue(node.data, "w_tref", 1),
  },
  {
    key: "num_inputs",
    label: "num_inputs",
    group: "Params",
    value: (node) => paramNumber(node.data, "num_inputs"),
  },
  {
    key: "num_neuromodulators",
    label: "num_neuromodulators",
    group: "Params",
    value: (node) => paramNumber(node.data, "num_neuromodulators"),
  },
];

export function NeuronClusteringPanel({
  state,
  selectedId,
  onSelect,
}: NeuronClusteringPanelProps) {
  const [scope, setScope] = useState<ClusterScope>("all");
  const [viewMode, setViewMode] = useState<ClusterViewMode>("2d");
  const [clusterMode, setClusterMode] = useState<ClusterCountMode>("auto");
  const [layerKey, setLayerKey] = useState("");
  const [manualClusterCount, setManualClusterCount] = useState(DEFAULT_MANUAL_CLUSTERS);
  const [autoMaxClusters, setAutoMaxClusters] = useState(DEFAULT_AUTO_MAX_CLUSTERS);
  const [features, setFeatures] = useState<string[]>(DEFAULT_FEATURES);
  const dimension = viewMode === "3d" ? 3 : 2;
  const requestedClusters =
    clusterMode === "auto" ? autoMaxClusters : manualClusterCount;
  const activeFeatures = useMemo(
    () => makeUniqueFeatures(features, dimension),
    [dimension, features],
  );
  const layerOptions = useMemo(() => getLayerOptions(state), [state]);

  useEffect(() => {
    if (layerOptions.length === 0) return;
    if (!layerOptions.some((option) => option.key === layerKey)) {
      setLayerKey(layerOptions[0].key);
    }
  }, [layerKey, layerOptions]);

  useEffect(() => {
    setFeatures((previous) => makeUniqueFeatures(previous, dimension));
  }, [dimension]);

  const scopedNodes = useMemo(
    () => resolveClusterNodes(state, scope, layerKey),
    [layerKey, scope, state],
  );
  const result = useMemo(
    () => runKMeans(scopedNodes, activeFeatures, requestedClusters, clusterMode),
    [activeFeatures, clusterMode, requestedClusters, scopedNodes],
  );
  const selectedCluster = useMemo(
    () => result.points.find((point) => point.id === selectedId)?.cluster ?? null,
    [result.points, selectedId],
  );

  const setFeatureAt = (axisIndex: number, nextKey: string) => {
    setFeatures((previous) => {
      if (previous.some((key, index) => index !== axisIndex && key === nextKey)) {
        return previous;
      }
      const next = [...previous];
      next[axisIndex] = nextKey;
      return makeUniqueFeatures(next, dimension);
    });
  };

  return (
    <section className="panelSection clusteringPanel" data-testid="clustering-panel">
      <h2>Clustering</h2>
      <div className="segmented clusteringScopeModes">
        <button
          className={clsx(scope === "all" && "active")}
          onClick={() => setScope("all")}
        >
          All
        </button>
        <button
          className={clsx(scope === "layer" && "active")}
          onClick={() => setScope("layer")}
        >
          Layer
        </button>
      </div>

      {scope === "layer" && (
        <label className="field clusteringLayerField">
          <span>layer</span>
          <select
            data-testid="cluster-layer-select"
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

      <div className="clusteringControlGrid">
        <div className="segmented clusteringModeSwitch" aria-label="cluster count mode">
          <button
            data-testid="cluster-mode-auto"
            className={clsx(clusterMode === "auto" && "active")}
            onClick={() => setClusterMode("auto")}
          >
            Auto
          </button>
          <button
            data-testid="cluster-mode-manual"
            className={clsx(clusterMode === "manual" && "active")}
            onClick={() => setClusterMode("manual")}
          >
            Manual
          </button>
        </div>
        <label className="field">
          <span>view</span>
          <select
            data-testid="cluster-view-select"
            value={viewMode}
            onChange={(event) => setViewMode(event.target.value as ClusterViewMode)}
          >
            <option value="2d">2D</option>
            <option value="3d">3D</option>
          </select>
        </label>
        <label className="field">
          <span>{clusterMode === "auto" ? "max K" : "clusters"}</span>
          <input
            data-testid="cluster-count"
            type="number"
            min={clusterMode === "auto" ? 2 : 1}
            max={12}
            step={1}
            value={requestedClusters}
            onChange={(event) => {
              const next = Number(event.target.value);
              if (clusterMode === "auto") setAutoMaxClusters(next);
              else setManualClusterCount(next);
            }}
          />
        </label>
      </div>

      <div className={clsx("clusterFeatureGrid", viewMode === "3d" && "three")}>
        {activeFeatures.map((feature, index) => (
          <FeatureSelect
            key={index}
            axis={index === 0 ? "X" : index === 1 ? "Y" : "Z"}
            value={feature}
            selected={activeFeatures}
            onChange={(nextFeature) => setFeatureAt(index, nextFeature)}
          />
        ))}
      </div>

      {viewMode === "3d" ? (
        <ClusterScatter3d
          result={result}
          selectedId={selectedId}
        />
      ) : (
        <ClusterScatter2d
          result={result}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      )}

      <div className="statsGrid clusteringStats">
        <ClusterMetric label="neurons" value={String(result.points.length)} />
        <ClusterMetric label="clusters" value={String(result.clusters.length)} />
        <ClusterMetric
          label="mode"
          value={result.mode === "auto" ? `auto/${result.requestedClusters}` : "manual"}
        />
        <ClusterMetric label="inertia" value={formatScalar(result.inertia)} />
        <ClusterMetric
          label="score"
          value={result.autoScore == null ? "-" : formatScalar(result.autoScore)}
        />
        <ClusterMetric
          label="selected"
          value={selectedCluster == null ? "-" : `C${selectedCluster + 1}`}
        />
      </div>

      <div className="clusterList">
        {result.clusters.map((cluster) => (
          <div
            key={cluster.id}
            className={clsx(
              "clusterRow",
              selectedCluster === cluster.id && "selected",
            )}
            style={{ ["--cluster-color" as string]: clusterColor(cluster.id) }}
          >
            <span>C{cluster.id + 1}</span>
            <strong>{cluster.count}</strong>
            <small>{cluster.centroid.map(formatScalar).join(" / ")}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function FeatureSelect({
  axis,
  value,
  selected,
  onChange,
}: {
  axis: "X" | "Y" | "Z";
  value: string;
  selected: string[];
  onChange: (value: string) => void;
}) {
  const groups = useMemo(() => groupedFeatures(), []);
  return (
    <label className="field clusterFeatureField">
      <span>{axis} parameter</span>
      <select
        data-testid={`cluster-${axis.toLowerCase()}-feature`}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {groups.map((group) => (
          <optgroup key={group.name} label={group.name}>
            {group.features.map((feature) => (
              <option
                key={feature.key}
                value={feature.key}
                disabled={selected.includes(feature.key) && feature.key !== value}
              >
                {feature.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

function ClusterMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ClusterScatter2d({
  result,
  selectedId,
  onSelect,
}: {
  result: KMeansResult;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const resultRef = useRef(result);
  resultRef.current = result;

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    if (!host || !canvas) return;
    const ctx =
      canvas.getContext("2d", { alpha: false, desynchronized: true }) ||
      canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const rect = host.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const width = Math.max(220, Math.floor(rect.width));
      const height = 238;
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawCluster2d(ctx, width, height, resultRef.current, selectedId);
    };

    const observer = new ResizeObserver(draw);
    observer.observe(host);
    draw();
    return () => observer.disconnect();
  }, [result, selectedId]);

  return (
    <div
      ref={hostRef}
      className="clusterFrame clusterFrame2d"
      data-testid="cluster-2d"
      onPointerUp={(event) => {
        const host = hostRef.current;
        if (!host) return;
        const rect = host.getBoundingClientRect();
        const hit = hitTest2d(
          resultRef.current,
          event.clientX - rect.left,
          event.clientY - rect.top,
          rect.width,
          238,
        );
        if (hit) onSelect(hit === selectedId ? null : hit);
      }}
    >
      <canvas ref={canvasRef} className="clusterCanvas2d" />
      {!result.ready && <div className="empty3dText">No cluster data</div>}
    </div>
  );
}

function ClusterScatter3d({
  result,
  selectedId,
}: {
  result: KMeansResult;
  selectedId: string | null;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const rotationRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const sceneRef = useRef<ClusterSceneHandles | null>(null);
  const sceneData = useMemo(
    () => buildClusterSceneData(result, selectedId),
    [result, selectedId],
  );
  const sceneDataRef = useRef(sceneData);
  sceneDataRef.current = sceneData;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(2.45, 1.8, 3.1);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setClearColor(0x080b10, 1);
    renderer.domElement.className = "clusterCanvas3d";
    renderer.domElement.dataset.testid = "cluster-3d-canvas";
    host.appendChild(renderer.domElement);

    const world = new THREE.Group();
    scene.add(world);
    const handles: ClusterSceneHandles = { camera, renderer, world };
    sceneRef.current = handles;
    updateClusterScene(handles, sceneDataRef.current);

    const resize = () => {
      const rect = host.getBoundingClientRect();
      const width = Math.max(220, Math.floor(rect.width));
      const height = 238;
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
      world.rotation.x = -0.42 + rotationRef.current.x;
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
    if (sceneRef.current) updateClusterScene(sceneRef.current, sceneData);
  }, [sceneData]);

  return (
    <div
      ref={hostRef}
      className="clusterFrame clusterFrame3d"
      data-testid="cluster-3d"
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
      {!result.ready && <div className="empty3dText">No cluster data</div>}
    </div>
  );
}

function drawCluster2d(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  result: KMeansResult,
  selectedId: string | null,
) {
  ctx.fillStyle = "#080b10";
  ctx.fillRect(0, 0, width, height);
  drawGrid2d(ctx, width, height);
  if (!result.ready) return;

  for (const point of result.points) {
    const [x, y] = project2d(point.normalized[0], point.normalized[1], width, height);
    const selected = point.id === selectedId;
    ctx.fillStyle = clusterColor(point.cluster);
    ctx.globalAlpha = selected ? 1 : 0.82;
    ctx.beginPath();
    ctx.arc(x, y, selected ? 5.4 : 3.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (selected) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
  }

  for (let index = 0; index < result.centroids.length; index += 1) {
    const centroid = result.centroids[index];
    const [x, y] = project2d(centroid[0], centroid[1], width, height);
    ctx.strokeStyle = clusterColor(index);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 7, y);
    ctx.lineTo(x + 7, y);
    ctx.moveTo(x, y - 7);
    ctx.lineTo(x, y + 7);
    ctx.stroke();
  }

  drawAxisLabels2d(ctx, result, width, height);
}

function drawGrid2d(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.strokeStyle = "rgba(255,255,255,0.055)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const x = 28 + ((width - 40) * i) / 4;
    const y = 14 + ((height - 44) * i) / 4;
    ctx.moveTo(x, 14);
    ctx.lineTo(x, height - 30);
    ctx.moveTo(28, y);
    ctx.lineTo(width - 12, y);
  }
  ctx.stroke();
}

function drawAxisLabels2d(
  ctx: CanvasRenderingContext2D,
  result: KMeansResult,
  width: number,
  height: number,
) {
  ctx.fillStyle = "rgba(198, 208, 220, 0.72)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(featureLabel(result.features[0]), width - 12, height - 8);
  ctx.save();
  ctx.translate(12, 18);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "right";
  ctx.fillText(featureLabel(result.features[1]), 0, 0);
  ctx.restore();
}

function hitTest2d(
  result: KMeansResult,
  x: number,
  y: number,
  width: number,
  height: number,
) {
  let best: { id: string; d2: number } | null = null;
  for (const point of result.points) {
    const [px, py] = project2d(point.normalized[0], point.normalized[1], width, height);
    const d2 = (px - x) ** 2 + (py - y) ** 2;
    if (d2 > 12 ** 2) continue;
    if (!best || d2 < best.d2) best = { id: point.id, d2 };
  }
  return best?.id ?? null;
}

function project2d(x: number, y: number, width: number, height: number): [number, number] {
  const left = 28;
  const right = width - 12;
  const top = 14;
  const bottom = height - 30;
  return [left + x * (right - left), bottom - y * (bottom - top)];
}

interface ClusterSceneHandles {
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  world: THREE.Group;
}

interface ClusterSceneData {
  ready: boolean;
  points: Array<{
    x: number;
    y: number;
    z: number;
    color: string;
    selected: boolean;
  }>;
  centroids: Array<{ x: number; y: number; z: number; color: string }>;
}

function buildClusterSceneData(
  result: KMeansResult,
  selectedId: string | null,
): ClusterSceneData {
  if (!result.ready || result.dimension < 3) {
    return { ready: false, points: [], centroids: [] };
  }
  return {
    ready: true,
    points: result.points.map((point) => ({
      x: point.normalized[0] * 2 - 1,
      y: point.normalized[1] * 2 - 1,
      z: point.normalized[2] * 2 - 1,
      color: clusterColor(point.cluster),
      selected: point.id === selectedId,
    })),
    centroids: result.centroids.map((centroid, index) => ({
      x: centroid[0] * 2 - 1,
      y: centroid[1] * 2 - 1,
      z: centroid[2] * 2 - 1,
      color: clusterColor(index),
    })),
  };
}

function updateClusterScene(handles: ClusterSceneHandles, data: ClusterSceneData) {
  clearGroup(handles.world);
  handles.world.add(makeAxisBox());
  if (!data.ready) return;

  const positions = new Float32Array(data.points.length * 3);
  const colors = new Float32Array(data.points.length * 3);
  data.points.forEach((point, index) => {
    positions[index * 3] = point.x;
    positions[index * 3 + 1] = point.y;
    positions[index * 3 + 2] = point.z;
    const color = new THREE.Color(point.color);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  handles.world.add(
    new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 0.065,
        vertexColors: true,
        sizeAttenuation: true,
      }),
    ),
  );

  for (const point of data.points) {
    if (!point.selected) continue;
    const selectedGeometry = new THREE.SphereGeometry(0.082, 18, 12);
    const selectedMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const selectedMesh = new THREE.Mesh(selectedGeometry, selectedMaterial);
    selectedMesh.position.set(point.x, point.y, point.z);
    handles.world.add(selectedMesh);
  }

  for (const centroid of data.centroids) {
    const color = new THREE.Color(centroid.color);
    const centroidGeometry = new THREE.SphereGeometry(0.06, 16, 10);
    const centroidMaterial = new THREE.MeshBasicMaterial({
      color,
      wireframe: true,
    });
    const centroidMesh = new THREE.Mesh(centroidGeometry, centroidMaterial);
    centroidMesh.position.set(centroid.x, centroid.y, centroid.z);
    handles.world.add(centroidMesh);
  }
}

function makeAxisBox() {
  const group = new THREE.Group();
  const vertices: number[] = [];
  const ticks = [-1, -0.5, 0, 0.5, 1];
  for (const tick of ticks) {
    vertices.push(-1, -1, tick, 1, -1, tick);
    vertices.push(tick, -1, -1, tick, -1, 1);
    vertices.push(-1, tick, -1, 1, tick, -1);
    vertices.push(tick, -1, -1, tick, 1, -1);
  }
  const grid = new THREE.BufferGeometry();
  grid.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  group.add(
    new THREE.LineSegments(
      grid,
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

function clearGroup(group: THREE.Group) {
  while (group.children.length) {
    const child = group.children.pop();
    if (child) disposeObject(child);
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

function runKMeans(
  nodes: GraphNode[],
  featureKeys: string[],
  requestedClusters: number,
  mode: ClusterCountMode,
): KMeansResult {
  const defs = featureKeys
    .map((key) => FEATURE_DEFS.find((feature) => feature.key === key))
    .filter((feature): feature is FeatureDef => !!feature);
  const dimension = defs.length;
  if (nodes.length === 0 || dimension < 2) {
    return {
      ready: false,
      mode,
      features: featureKeys,
      dimension,
      requestedClusters,
      points: [],
      centroids: [],
      clusters: [],
      inertia: 0,
      autoScore: null,
    };
  }

  const raw = nodes.map((node) => defs.map((def) => finiteValue(def.value(node))));
  const bounds = defs.map((_, featureIndex) => {
    const values = raw.map((row) => row[featureIndex]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(1e-9, max - min);
    return { min, max, span };
  });
  const normalized = raw.map((row) =>
    row.map((value, featureIndex) => {
      const bound = bounds[featureIndex];
      return bound.span <= 1e-9 ? 0.5 : (value - bound.min) / bound.span;
    }),
  );
  const maxK = Math.min(12, nodes.length, distinctRowCount(normalized));
  const requestedK = clampInt(requestedClusters, 1, Math.max(1, maxK));
  const selected = mode === "auto"
    ? selectAutomaticK(normalized, requestedK, dimension)
    : solveKMeans(normalized, requestedK, dimension);
  const { assignments, centroids } = selected;

  const points = nodes.map((node, index) => ({
    id: node.data.id,
    label: node.data.label ?? String(node.data.neuron_id ?? node.data.id),
    node,
    raw: raw[index],
    normalized: normalized[index],
    cluster: assignments[index],
  }));
  const clusters = centroids.map((centroid, clusterIndex) => {
    const members = points.filter((point) => point.cluster === clusterIndex);
    return {
      id: clusterIndex,
      count: members.length,
      centroid: centroid.map((value, featureIndex) => {
        const bound = bounds[featureIndex];
        return bound.min + value * bound.span;
      }),
    };
  });
  return {
    ready: true,
    mode,
    features: featureKeys,
    dimension,
    requestedClusters: requestedK,
    points,
    centroids,
    clusters,
    inertia: selected.inertia,
    autoScore: mode === "auto" ? selected.score : null,
  };
}

interface KMeansSolution {
  assignments: number[];
  centroids: number[][];
  inertia: number;
  score: number | null;
}

function selectAutomaticK(
  rows: number[][],
  maxK: number,
  dimension: number,
): KMeansSolution {
  const cappedMax = clampInt(maxK, 1, Math.min(12, rows.length));
  if (cappedMax <= 1) {
    return solveKMeans(rows, 1, dimension, 0);
  }

  const candidates: KMeansSolution[] = [solveKMeans(rows, 1, dimension, -0.05)];
  for (let k = 2; k <= cappedMax; k += 1) {
    const candidate = solveKMeans(rows, k, dimension);
    const score = automaticClusterScore(rows, candidate, k);
    candidate.score = score;
    candidates.push(candidate);
  }

  const bestScore = Math.max(...candidates.map((candidate) => candidate.score ?? -Infinity));
  if (bestScore < 0.08) return candidates[0];

  const tolerance = Math.max(0.045, Math.abs(bestScore) * 0.07);
  return (
    candidates.find((candidate) => (candidate.score ?? -Infinity) >= bestScore - tolerance) ??
    candidates[candidates.length - 1]
  );
}

function solveKMeans(
  rows: number[][],
  k: number,
  dimension: number,
  score: number | null = null,
): KMeansSolution {
  const clusterCount = clampInt(k, 1, Math.min(12, rows.length));
  const assignments = new Array(rows.length).fill(0);
  let centroids = initialCentroids(rows, clusterCount);

  for (let iteration = 0; iteration < 18; iteration += 1) {
    let changed = false;
    rows.forEach((row, rowIndex) => {
      const nextCluster = nearestCentroid(row, centroids);
      if (assignments[rowIndex] !== nextCluster) {
        assignments[rowIndex] = nextCluster;
        changed = true;
      }
    });
    centroids = recomputeCentroids(rows, assignments, clusterCount, dimension, iteration);
    if (!changed && iteration > 0) break;
  }

  const inertia =
    rows.length > 0
      ? average(rows.map((row, index) => squaredDistance(row, centroids[assignments[index]])))
      : 0;
  return { assignments, centroids, inertia, score };
}

function automaticClusterScore(rows: number[][], solution: KMeansSolution, k: number) {
  const counts = new Array(k).fill(0);
  for (const assignment of solution.assignments) counts[assignment] += 1;
  const nonEmpty = counts.filter((count) => count > 0).length;
  if (nonEmpty < Math.min(k, 2)) return -1;

  const silhouette = centroidSilhouette(rows, solution.assignments, solution.centroids);
  const emptyPenalty = (k - nonEmpty) * 0.18;
  const largestShare = Math.max(...counts) / Math.max(1, rows.length);
  const dominancePenalty = largestShare > 0.92 ? (largestShare - 0.92) * 0.75 : 0;
  const complexityPenalty = Math.log2(k + 1) * 0.038;
  return silhouette - emptyPenalty - dominancePenalty - complexityPenalty;
}

function centroidSilhouette(
  rows: number[][],
  assignments: number[],
  centroids: number[][],
) {
  if (centroids.length <= 1 || rows.length === 0) return 0;
  return average(
    rows.map((row, index) => {
      const own = assignments[index];
      const ownDistance = Math.sqrt(squaredDistance(row, centroids[own] ?? centroids[0]));
      let otherDistance = Number.POSITIVE_INFINITY;
      for (let centroidIndex = 0; centroidIndex < centroids.length; centroidIndex += 1) {
        if (centroidIndex === own) continue;
        otherDistance = Math.min(
          otherDistance,
          Math.sqrt(squaredDistance(row, centroids[centroidIndex])),
        );
      }
      const scale = Math.max(ownDistance, otherDistance, 1e-9);
      return (otherDistance - ownDistance) / scale;
    }),
  );
}

function distinctRowCount(rows: number[][]) {
  const seen = new Set<string>();
  for (const row of rows) {
    seen.add(row.map((value) => value.toFixed(6)).join("|"));
  }
  return Math.max(1, seen.size);
}

function initialCentroids(rows: number[][], k: number) {
  const sorted = [...rows].sort((a, b) => a[0] - b[0]);
  return Array.from({ length: k }, (_, index) => {
    const pick = Math.round((index / Math.max(1, k - 1)) * (sorted.length - 1));
    return [...sorted[pick]];
  });
}

function nearestCentroid(row: number[], centroids: number[][]) {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  centroids.forEach((centroid, index) => {
    const distance = squaredDistance(row, centroid);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function recomputeCentroids(
  rows: number[][],
  assignments: number[],
  k: number,
  dimension: number,
  iteration: number,
) {
  const sums = Array.from({ length: k }, () => new Array(dimension).fill(0));
  const counts = new Array(k).fill(0);
  rows.forEach((row, index) => {
    const cluster = assignments[index];
    counts[cluster] += 1;
    for (let featureIndex = 0; featureIndex < dimension; featureIndex += 1) {
      sums[cluster][featureIndex] += row[featureIndex];
    }
  });
  return sums.map((sum, clusterIndex) => {
    if (counts[clusterIndex] === 0) {
      return [...rows[(iteration + clusterIndex * 7) % rows.length]];
    }
    return sum.map((value) => value / counts[clusterIndex]);
  });
}

function resolveClusterNodes(
  state: NetworkState | null,
  scope: ClusterScope,
  layerKey: string,
) {
  const nodes = neuronNodes(state);
  if (scope === "all") return nodes;
  return nodes.filter((node) => getLayerKey(node.data) === layerKey);
}

function neuronNodes(state: NetworkState | null) {
  return (state?.elements.nodes ?? []).filter((node) => node.data.type === "neuron");
}

function getLayerOptions(state: NetworkState | null): LayerOption[] {
  const map = new Map<string, LayerOption>();
  for (const node of neuronNodes(state)) {
    const key = getLayerKey(node.data);
    const label = node.data.layer_name ?? `layer ${String(node.data.layer ?? key)}`;
    const existing = map.get(key);
    if (existing) existing.count += 1;
    else map.set(key, { key, label, count: 1 });
  }
  return Array.from(map.values()).sort((a, b) => layerSortKey(a.key) - layerSortKey(b.key));
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

function groupedFeatures() {
  const groups = new Map<string, FeatureDef[]>();
  for (const feature of FEATURE_DEFS) {
    const list = groups.get(feature.group) ?? [];
    list.push(feature);
    groups.set(feature.group, list);
  }
  return Array.from(groups.entries()).map(([name, features]) => ({ name, features }));
}

function makeUniqueFeatures(features: string[], dimension: number) {
  const out: string[] = [];
  for (let index = 0; index < dimension; index += 1) {
    let candidate = features[index] ?? DEFAULT_FEATURES[index] ?? FEATURE_DEFS[0].key;
    if (!FEATURE_DEFS.some((feature) => feature.key === candidate) || out.includes(candidate)) {
      candidate = FEATURE_DEFS.find((feature) => !out.includes(feature.key))?.key ?? FEATURE_DEFS[0].key;
    }
    out.push(candidate);
  }
  return out;
}

function featureLabel(key: string) {
  return FEATURE_DEFS.find((feature) => feature.key === key)?.label ?? key;
}

function clusterColor(index: number) {
  return CLUSTER_COLORS[index % CLUSTER_COLORS.length];
}

function paramNumber(data: GraphNodeData, key: string, fallbackKey?: string) {
  const params = data.params ?? {};
  const primary = params[key];
  const fallback = fallbackKey ? params[fallbackKey] : undefined;
  return numberValue(primary ?? fallback);
}

function paramVectorValue(data: GraphNodeData, key: string, index: number) {
  const value = data.params?.[key];
  return Array.isArray(value) ? numberValue(value[index]) : 0;
}

function vectorValue(values: unknown, index: number) {
  return Array.isArray(values) ? numberValue(values[index]) : 0;
}

function countValue(value: unknown) {
  return Array.isArray(value) ? value.length : 0;
}

function numberValue(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function finiteValue(value: number) {
  return Number.isFinite(value) ? value : 0;
}

function squaredDistance(a: number[], b: number[]) {
  let out = 0;
  for (let index = 0; index < a.length; index += 1) {
    out += (a[index] - b[index]) ** 2;
  }
  return out;
}

function average(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function clampInt(value: number, min: number, max: number) {
  const parsed = Math.floor(Number(value));
  return clamp(Number.isFinite(parsed) ? parsed : min, min, max);
}

function formatScalar(value: number) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}
