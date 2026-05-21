import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from "react";
import type {
  FrameStats,
  GraphEdge,
  GraphNode,
  GraphNodeData,
  LayoutName,
  NetworkState,
} from "./types";

const PAD = 26;
const ZOOM_MIN = 0.12;
const ZOOM_MAX = 18;
const BASE_NODE_RADIUS = 5;
const SELECTED_RADIUS = 8;

interface RenderPoint {
  id: string;
  x: number;
  y: number;
  node: GraphNode;
}

interface CanvasNetworkProps {
  state: NetworkState | null;
  selectedId: string | null;
  layout: LayoutName;
  showEdges: boolean;
  edgeOpacity: number;
  nodeScale: number;
  onSelect: (id: string | null) => void;
  onFrameStats: (stats: FrameStats) => void;
}

export interface CanvasNetworkHandle {
  fit: () => void;
}

export const CanvasNetwork = forwardRef<CanvasNetworkHandle, CanvasNetworkProps>(
  function CanvasNetwork(
    {
      state,
      selectedId,
      layout,
      showEdges,
      edgeOpacity,
      nodeScale,
      onSelect,
      onFrameStats,
    },
    ref,
  ) {
    const hostRef = useRef<HTMLDivElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const viewRef = useRef({ cx: 0, cy: 0, zoom: 1 });
    const dragRef = useRef<{
      pointerId: number;
      lastX: number;
      lastY: number;
      moved: boolean;
    } | null>(null);
    const latestRef = useRef({
      selectedId,
      showEdges,
      edgeOpacity,
      nodeScale,
      onSelect,
      onFrameStats,
    });

    const graph = useMemo(() => buildRenderGraph(state, layout), [state, layout]);
    const graphRef = useRef(graph);
    graphRef.current = graph;

    latestRef.current = {
      selectedId,
      showEdges,
      edgeOpacity,
      nodeScale,
      onSelect,
      onFrameStats,
    };

    useImperativeHandle(ref, () => ({
      fit: () => {
        viewRef.current = { cx: 0, cy: 0, zoom: 1 };
      },
    }));

    useEffect(() => {
      viewRef.current = { cx: 0, cy: 0, zoom: 1 };
    }, [layout, state?.elements.nodes.length]);

    useEffect(() => {
      const canvas = canvasRef.current;
      const host = hostRef.current;
      if (!canvas || !host) return;
      const ctx =
        canvas.getContext("2d", { alpha: false, desynchronized: true }) ||
        canvas.getContext("2d");
      if (!ctx) return;

      let raf = 0;
      let frameCount = 0;
      let lastFpsAt = performance.now();
      let lastFps = 0;

      const draw = (now: number) => {
        const rect = host.getBoundingClientRect();
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        const w = Math.max(180, Math.floor(rect.width));
        const h = Math.max(180, Math.floor(rect.height));
        if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
          canvas.width = Math.floor(w * dpr);
          canvas.height = Math.floor(h * dpr);
          canvas.style.width = `${w}px`;
          canvas.style.height = `${h}px`;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = "#07090d";
        ctx.fillRect(0, 0, w, h);

        const current = latestRef.current;
        const currentGraph = graphRef.current;
        const transform = makeTransform(w, h, viewRef.current);

        drawGrid(ctx, w, h);
        drawEdges(ctx, currentGraph, transform, current);
        drawTravelingSignals(ctx, currentGraph, transform);
        drawNodes(ctx, currentGraph, transform, current);
        drawHud(ctx, currentGraph, w, h, lastFps);

        frameCount += 1;
        if (now - lastFpsAt > 500) {
          lastFps = (frameCount * 1000) / (now - lastFpsAt);
          current.onFrameStats({
            fps: lastFps,
            nodes: currentGraph.points.length,
            edges: currentGraph.edges.length,
          });
          frameCount = 0;
          lastFpsAt = now;
        }

        raf = requestAnimationFrame(draw);
      };

      raf = requestAnimationFrame(draw);
      return () => cancelAnimationFrame(raf);
    }, []);

    return (
      <div
        ref={hostRef}
        className="canvasHost"
        data-edge-opacity={edgeOpacity.toFixed(2)}
        data-node-scale={nodeScale.toFixed(2)}
        data-show-edges={showEdges ? "true" : "false"}
        onPointerDown={(event) => {
          const host = hostRef.current;
          if (!host) return;
          try {
            host.setPointerCapture(event.pointerId);
          } catch {
            // Pointer capture can fail in older embedded browsers.
          }
          dragRef.current = {
            pointerId: event.pointerId,
            lastX: event.clientX,
            lastY: event.clientY,
            moved: false,
          };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          const host = hostRef.current;
          if (!drag || drag.pointerId !== event.pointerId || !host) return;
          const rect = host.getBoundingClientRect();
          const dx = event.clientX - drag.lastX;
          const dy = event.clientY - drag.lastY;
          if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
          drag.lastX = event.clientX;
          drag.lastY = event.clientY;
          panView(dx, dy, rect.width, rect.height, viewRef.current);
        }}
        onPointerUp={(event) => {
          const drag = dragRef.current;
          const host = hostRef.current;
          if (!drag || drag.pointerId !== event.pointerId || !host) return;
          try {
            host.releasePointerCapture(event.pointerId);
          } catch {
            // Ignore release failures.
          }

          if (!drag.moved) {
            const rect = host.getBoundingClientRect();
            const selected = hitTest(
              graphRef.current.points,
              event.clientX - rect.left,
              event.clientY - rect.top,
              rect.width,
              rect.height,
              viewRef.current,
              latestRef.current.nodeScale,
            );
            latestRef.current.onSelect(
              selected === latestRef.current.selectedId ? null : selected,
            );
          }
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
        onWheel={(event) => {
          const host = hostRef.current;
          if (!host) return;
          event.preventDefault();
          const rect = host.getBoundingClientRect();
          zoomView(
            event.clientX - rect.left,
            event.clientY - rect.top,
            rect.width,
            rect.height,
            Math.exp(-event.deltaY * 0.0015),
            viewRef.current,
          );
        }}
        onDoubleClick={() => {
          viewRef.current = { cx: 0, cy: 0, zoom: 1 };
        }}
      >
        <canvas ref={canvasRef} className="networkCanvas" />
      </div>
    );
  },
);

function buildRenderGraph(state: NetworkState | null, layout: LayoutName) {
  const nodes = state?.elements.nodes ?? [];
  const edges = state?.elements.edges ?? [];
  const points = layoutNodes(nodes, edges, layout);
  const pointById = new Map(points.map((p) => [p.id, p]));
  return {
    points,
    pointById,
    edges,
    signals: state?.traveling_signals ?? [],
  };
}

function layoutNodes(nodes: GraphNode[], edges: GraphEdge[], layout: LayoutName) {
  if (nodes.length === 0) return [];
  if (layout === "grid") return gridLayout(nodes);
  if (layout === "circle") return circleLayout(nodes);
  if (layout === "concentric") return concentricLayout(nodes, edges);
  return layerLayout(nodes);
}

function layerLayout(nodes: GraphNode[]): RenderPoint[] {
  const raw = nodes.map((node, index) => {
    const pos = node.position ?? node.data.position;
    return {
      id: node.data.id,
      x: typeof pos?.x === "number" ? pos.x : fallbackLayerX(node.data, index),
      y: typeof pos?.y === "number" ? pos.y : fallbackLayerY(node.data, index),
      node,
    };
  });
  return normalize(raw);
}

function gridLayout(nodes: GraphNode[]): RenderPoint[] {
  const cols = Math.ceil(Math.sqrt(nodes.length));
  const rows = Math.ceil(nodes.length / cols);
  const out = nodes.map((node, index) => ({
    id: node.data.id,
    x: cols <= 1 ? 0 : (index % cols) / (cols - 1) * 2 - 1,
    y: rows <= 1 ? 0 : Math.floor(index / cols) / (rows - 1) * 2 - 1,
    node,
  }));
  return out;
}

function circleLayout(nodes: GraphNode[]): RenderPoint[] {
  const radius = 0.88;
  return nodes.map((node, index) => {
    const a = (index / nodes.length) * Math.PI * 2 - Math.PI / 2;
    return {
      id: node.data.id,
      x: Math.cos(a) * radius,
      y: Math.sin(a) * radius,
      node,
    };
  });
}

function concentricLayout(nodes: GraphNode[], edges: GraphEdge[]): RenderPoint[] {
  const degree = new Map<string, number>();
  for (const edge of edges) {
    degree.set(edge.data.source, (degree.get(edge.data.source) ?? 0) + 1);
    degree.set(edge.data.target, (degree.get(edge.data.target) ?? 0) + 1);
  }
  const maxDegree = Math.max(1, ...Array.from(degree.values()));
  return nodes.map((node, index) => {
    const d = degree.get(node.data.id) ?? 0;
    const radius = 0.16 + (1 - d / maxDegree) * 0.78;
    const a = (index / nodes.length) * Math.PI * 2 - Math.PI / 2;
    return {
      id: node.data.id,
      x: Math.cos(a) * radius,
      y: Math.sin(a) * radius,
      node,
    };
  });
}

function fallbackLayerX(data: GraphNodeData, index: number) {
  const layer = numericLayer(data.layer, index);
  return layer * 160;
}

function fallbackLayerY(data: GraphNodeData, index: number) {
  const layer = numericLayer(data.layer, 0);
  return (index % 31) * 32 + layer * 11;
}

function numericLayer(layer: GraphNodeData["layer"], fallback: number) {
  if (typeof layer === "number" && Number.isFinite(layer)) return layer;
  if (typeof layer === "string" && layer.trim() !== "") {
    const parsed = Number(layer);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function normalize(points: RenderPoint[]) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  return points.map((p) => ({
    ...p,
    x: ((p.x - minX) / spanX) * 2 - 1,
    y: ((p.y - minY) / spanY) * 2 - 1,
  }));
}

function makeTransform(
  w: number,
  h: number,
  view: { cx: number; cy: number; zoom: number },
) {
  const scale = (Math.min(w - PAD * 2, h - PAD * 2) / 2) * view.zoom;
  return (x: number, y: number): [number, number] => [
    w / 2 + (x - view.cx) * scale,
    h / 2 + (y - view.cy) * scale,
  ];
}

function panView(
  dx: number,
  dy: number,
  w: number,
  h: number,
  view: { cx: number; cy: number; zoom: number },
) {
  const scale = (Math.min(w - PAD * 2, h - PAD * 2) / 2) * view.zoom;
  view.cx -= dx / scale;
  view.cy -= dy / scale;
  view.cx = clamp(view.cx, -4, 4);
  view.cy = clamp(view.cy, -4, 4);
}

function zoomView(
  px: number,
  py: number,
  w: number,
  h: number,
  factor: number,
  view: { cx: number; cy: number; zoom: number },
) {
  const oldZoom = view.zoom;
  const newZoom = clamp(oldZoom * factor, ZOOM_MIN, ZOOM_MAX);
  if (Math.abs(newZoom - oldZoom) < 1e-6) return;

  const oldScale = (Math.min(w - PAD * 2, h - PAD * 2) / 2) * oldZoom;
  const worldX = view.cx + (px - w / 2) / oldScale;
  const worldY = view.cy + (py - h / 2) / oldScale;
  const newScale = (Math.min(w - PAD * 2, h - PAD * 2) / 2) * newZoom;
  view.cx = worldX - (px - w / 2) / newScale;
  view.cy = worldY - (py - h / 2) / newScale;
  view.zoom = newZoom;
}

function hitTest(
  points: RenderPoint[],
  x: number,
  y: number,
  w: number,
  h: number,
  view: { cx: number; cy: number; zoom: number },
  nodeScale: number,
) {
  const transform = makeTransform(w, h, view);
  let best: { id: string; d2: number } | null = null;
  for (const p of points) {
    const [px, py] = transform(p.x, p.y);
    const r = radiusForNode(p.node.data, false, nodeScale) + 8;
    const d2 = (px - x) ** 2 + (py - y) ** 2;
    if (d2 > r ** 2) continue;
    if (!best || d2 < best.d2) best = { id: p.id, d2 };
  }
  return best?.id ?? null;
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
) {
  ctx.strokeStyle = "rgba(255,255,255,0.035)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= w; x += 48) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  for (let y = 0; y <= h; y += 48) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();
}

function drawEdges(
  ctx: CanvasRenderingContext2D,
  graph: ReturnType<typeof buildRenderGraph>,
  transform: (x: number, y: number) => [number, number],
  current: {
    selectedId: string | null;
    showEdges: boolean;
    edgeOpacity: number;
  },
) {
  if (!current.showEdges || current.edgeOpacity <= 0) return;
  const selected = current.selectedId;
  const normalAlpha = clamp(current.edgeOpacity, 0, 1);
  const dimAlpha = normalAlpha * 0.2;
  const selectedAlpha = clamp(normalAlpha * 1.7, 0, 0.95);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.beginPath();
  ctx.strokeStyle = selected
    ? `rgba(100, 118, 136, ${dimAlpha})`
    : `rgba(108, 132, 156, ${normalAlpha})`;
  ctx.lineWidth = 0.7;
  for (const edge of graph.edges) {
    if (selected && (edge.data.source === selected || edge.data.target === selected)) {
      continue;
    }
    const a = graph.pointById.get(edge.data.source);
    const b = graph.pointById.get(edge.data.target);
    if (!a || !b) continue;
    const [ax, ay] = transform(a.x, a.y);
    const [bx, by] = transform(b.x, b.y);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();

  if (!selected) {
    if (graph.edges.length < 800) {
      for (const edge of graph.edges) drawArrowForEdge(ctx, graph, edge, transform, normalAlpha);
    }
    return;
  }

  ctx.beginPath();
  ctx.strokeStyle = `rgba(234, 244, 255, ${selectedAlpha})`;
  ctx.lineWidth = 1.45;
  for (const edge of graph.edges) {
    if (edge.data.source !== selected && edge.data.target !== selected) continue;
    const a = graph.pointById.get(edge.data.source);
    const b = graph.pointById.get(edge.data.target);
    if (!a || !b) continue;
    const [ax, ay] = transform(a.x, a.y);
    const [bx, by] = transform(b.x, b.y);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();

  for (const edge of graph.edges) {
    if (edge.data.source === selected || edge.data.target === selected) {
      drawArrowForEdge(ctx, graph, edge, transform, selectedAlpha);
    }
  }
}

function drawArrowForEdge(
  ctx: CanvasRenderingContext2D,
  graph: ReturnType<typeof buildRenderGraph>,
  edge: GraphEdge,
  transform: (x: number, y: number) => [number, number],
  alpha: number,
) {
  const a = graph.pointById.get(edge.data.source);
  const b = graph.pointById.get(edge.data.target);
  if (!a || !b) return;
  const [ax, ay] = transform(a.x, a.y);
  const [bx, by] = transform(b.x, b.y);
  const angle = Math.atan2(by - ay, bx - ax);
  const r = radiusForNode(b.node.data, false, 1) + 3;
  const tipX = bx - Math.cos(angle) * r;
  const tipY = by - Math.sin(angle) * r;
  const size = 5;
  ctx.fillStyle = edge.data.type === "external"
    ? `rgba(74, 222, 128, ${alpha})`
    : `rgba(160, 178, 198, ${alpha})`;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - Math.cos(angle - 0.55) * size, tipY - Math.sin(angle - 0.55) * size);
  ctx.lineTo(tipX - Math.cos(angle + 0.55) * size, tipY - Math.sin(angle + 0.55) * size);
  ctx.closePath();
  ctx.fill();
}

function drawTravelingSignals(
  ctx: CanvasRenderingContext2D,
  graph: ReturnType<typeof buildRenderGraph>,
  transform: (x: number, y: number) => [number, number],
) {
  for (const signal of graph.signals) {
    const a = graph.pointById.get(signal.source);
    const b = graph.pointById.get(signal.target);
    if (!a || !b) continue;
    const progress = clamp(signal.progress, 0, 1);
    const x = a.x + (b.x - a.x) * progress;
    const y = a.y + (b.y - a.y) * progress;
    const [px, py] = transform(x, y);
    ctx.fillStyle = signal.color || "#f59e0b";
    ctx.shadowColor = signal.color || "#f59e0b";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(px, py, signal.event_type === "RetrogradeSignalEvent" ? 4 : 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

function drawNodes(
  ctx: CanvasRenderingContext2D,
  graph: ReturnType<typeof buildRenderGraph>,
  transform: (x: number, y: number) => [number, number],
  current: { selectedId: string | null; nodeScale: number },
) {
  const shouldLabel = graph.points.length <= 160;
  for (const p of graph.points) {
    const selected = p.id === current.selectedId;
    const [px, py] = transform(p.x, p.y);
    const r = radiusForNode(p.node.data, selected, current.nodeScale);
    ctx.fillStyle = colorForNode(p.node.data);
    if (p.node.data.type === "external") {
      ctx.fillRect(px - r, py - r, r * 2, r * 2);
      if (selected) {
        ctx.strokeStyle = "rgba(246, 247, 255, 0.96)";
        ctx.lineWidth = 2;
        ctx.strokeRect(px - r, py - r, r * 2, r * 2);
      }
    } else {
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
      if (selected) {
        ctx.strokeStyle = "rgba(246, 247, 255, 0.96)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    if (shouldLabel || selected) {
      const label = p.node.data.label || String(p.node.data.neuron_id ?? p.id);
      ctx.fillStyle = selected ? "#ffffff" : "rgba(232, 237, 244, 0.65)";
      ctx.font = selected ? "600 12px ui-monospace, monospace" : "10px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(label, px, py + r + 4);
    }
  }
}

function drawHud(
  ctx: CanvasRenderingContext2D,
  graph: ReturnType<typeof buildRenderGraph>,
  w: number,
  h: number,
  fps: number,
) {
  ctx.fillStyle = "rgba(12, 16, 22, 0.82)";
  ctx.fillRect(w - 158, h - 35, 144, 22);
  ctx.fillStyle = "rgba(225, 230, 236, 0.72)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(
    `${graph.points.length}n ${graph.edges.length}e ${fps.toFixed(0)}fps`,
    w - 24,
    h - 24,
  );
}

function radiusForNode(data: GraphNodeData, selected: boolean, nodeScale: number) {
  const output = Math.abs(data.output ?? 0);
  const potential = Math.abs(data.membrane_potential ?? 0);
  const activity = Math.min(5, output * 4 + potential * 2);
  const base = data.type === "external" ? 4 : selected ? SELECTED_RADIUS : BASE_NODE_RADIUS;
  const payloadBase = data.base_size ? data.base_size / 6 : base;
  return clamp((Math.max(base, payloadBase) + activity) * nodeScale, 3, 18);
}

function colorForNode(data: GraphNodeData) {
  if (data.base_color && (data.output ?? 0) <= 0) return data.base_color;
  if (data.type === "external") return "rgba(74, 222, 128, 0.9)";
  if ((data.output ?? 0) > 0) return "rgba(248, 113, 113, 0.96)";
  const potential = data.membrane_potential ?? 0;
  if (potential > 0.5) return "rgba(251, 146, 60, 0.9)";
  if (potential > 0) return "rgba(250, 204, 21, 0.82)";
  return "rgba(96, 165, 250, 0.72)";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
