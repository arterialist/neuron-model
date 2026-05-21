import { useEffect, useMemo, useState } from "react";
import { Activity, CircleStop, X } from "lucide-react";
import { getNeuronDetail } from "./api";
import type {
  GraphEdge,
  GraphNode,
  GraphNodeData,
  NetworkState,
  NeuronDetailData,
  NeuronPresynapticTargetView,
} from "./types";

interface NeuronBottomSheetProps {
  node: GraphNode | null;
  state: NetworkState | null;
  history: NetworkState[];
  onClose: () => void;
}

const PARAM_ORDER = [
  "delta_decay",
  "beta_avg",
  "eta_post",
  "eta_retro",
  "c",
  "lambda_param",
  "p",
  "r_base",
  "b_base",
  "gamma",
  "w_r",
  "w_b",
  "w_tref",
  "num_neuromodulators",
  "num_inputs",
];

type SheetNeuronData = GraphNodeData & Partial<NeuronDetailData>;

export function NeuronBottomSheet({
  node,
  state,
  history,
  onClose,
}: NeuronBottomSheetProps) {
  const liveData = node?.data ?? null;
  const [detail, setDetail] = useState<NeuronDetailData | null>(null);
  const [detailError, setDetailError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setDetailError("");

    const neuronId = liveData?.neuron_id;
    if (!liveData || liveData.type !== "neuron" || neuronId == null) return;

    void getNeuronDetail(neuronId)
      .then((nextDetail) => {
        if (!cancelled) setDetail(nextDetail);
      })
      .catch((error) => {
        if (!cancelled) {
          setDetailError(error instanceof Error ? error.message : String(error));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [liveData?.id, liveData?.neuron_id, liveData?.type]);

  const data = useMemo(() => mergeNeuronData(liveData, detail), [detail, liveData]);
  const incident = useMemo(() => incidentEdges(node, state), [node, state]);
  const posts = useMemo(() => postsynapticRows(data, incident.incoming), [data, incident]);
  const pres = useMemo(() => presynapticRows(data, incident.outgoing), [data, incident]);
  const series = useMemo(() => selectedSeries(data, history), [data, history]);

  if (!data) return null;

  const params = normalizeParams(data.params);
  const mVector = Array.isArray(data.M_vector) ? data.M_vector : [];
  const paramGroups = groupParams(params);

  return (
    <section className="neuronSheet" data-testid="neuron-bottom-sheet">
      <div className="neuronSheetGrip" />
      <header className="neuronSheetHeader">
        <div>
          <h2>{data.label ?? data.id}</h2>
          <span>
            paula_id={String(data.paula_id ?? data.neuron_id ?? data.id)} ·{" "}
            {data.layer_name ?? data.layer ?? "-"}
          </span>
          {detailError && <span className="sheetDetailStatus">{detailError}</span>}
        </div>
        <button className="sheetClose" onClick={onClose} aria-label="Close neuron details">
          <X size={16} />
        </button>
      </header>

      <div className="neuronSheetBody">
        <section className="sheetSection sheetLive">
          <h3>Live state</h3>
          <SparkRow label="S" values={series.S} color="#74d4ff" />
          <SparkRow label="fire" values={series.fire} color="#f87171" min={0} max={1} />
          <SparkRow label="r" values={series.r} color="#5ee090" />
          <SparkRow label="b" values={series.b} color="#38bdf8" />
          <SparkRow label="t_ref" values={series.tRef} color="#fb923c" min={0} />
          <SparkRow label="M0" values={series.m0} color="#fb923c" />
          <SparkRow label="M1" values={series.m1} color="#5ee090" />
          <div className="sheetMiniMetric">
            <span>pq_len</span>
            <strong>{formatScalar(numberValue(data.pq_len))}</strong>
          </div>
        </section>

        <section className="sheetSection">
          <h3>Runtime state</h3>
          <div className="sheetMetricGrid">
            <SheetMetric label="S" value={data.membrane_potential} />
            <SheetMetric label="O" value={data.output} />
            <SheetMetric label="r" value={data.r} />
            <SheetMetric label="b" value={data.b} />
            <SheetMetric label="t_ref" value={data.t_ref} />
            <SheetMetric label="F_avg" value={data.firing_rate} />
            <SheetMetric label="t_last_fire" value={data.t_last_fire} />
            <SheetMetric label="activity" value={data.activity_level ?? "-"} />
            <SheetMetric label="synapses" value={data.synapses?.length} />
            <SheetMetric label="terminals" value={data.terminals?.length} />
          </div>
          <VectorReadout label="M_vector" values={mVector} />
        </section>

        <section className="sheetSection">
          <h3>Scalar params</h3>
          <div className="sheetMetricGrid">
            {paramGroups.scalars.map((entry) => (
              <ParamMetric key={entry.key} label={entry.key} value={entry.value} />
            ))}
          </div>
        </section>

        <section className="sheetSection">
          <h3>Parameter vectors</h3>
          <div className="sheetVectorStack">
            {paramGroups.vectors.map((entry) => (
              <VectorReadout
                key={entry.key}
                label={entry.key}
                values={Array.isArray(entry.value) ? entry.value : []}
              />
            ))}
          </div>
        </section>

        {paramGroups.other.length > 0 && (
          <section className="sheetSection">
            <h3>Other params</h3>
            <div className="sheetMetricGrid">
              {paramGroups.other.map((entry) => (
                <ParamMetric key={entry.key} label={entry.key} value={entry.value} />
              ))}
            </div>
          </section>
        )}

        <section className="sheetSection sheetTableSection">
          <h3>Postsynaptic weights</h3>
          <SheetTable
            columns={["slot", "from", "info", "plast", "V", "dist", "adapt"]}
            rows={posts}
          />
        </section>

        <section className="sheetSection sheetTableSection">
          <h3>Presynaptic terminals</h3>
          <SheetTable
            columns={["term", "to", "u_o info", "u_i retro", "dist", "mod"]}
            rows={pres}
          />
        </section>
      </div>
    </section>
  );
}

function mergeNeuronData(
  liveData: GraphNodeData | null,
  detail: NeuronDetailData | null,
): SheetNeuronData | null {
  if (!liveData && !detail) return null;
  const merged: SheetNeuronData = {
    id: liveData?.id ?? `neuron_${String(detail?.paula_id ?? detail?.id ?? "")}`,
    label:
      detail?.name ??
      liveData?.label ??
      String(detail?.paula_id ?? detail?.id ?? "neuron"),
    type: liveData?.type ?? "neuron",
    neuron_id: detail?.paula_id ?? detail?.id ?? liveData?.neuron_id,
    paula_id: detail?.paula_id ?? liveData?.neuron_id,
    membrane_potential: detail?.S ?? detail?.membrane_potential,
    firing_rate: detail?.F_avg ?? detail?.firing_rate,
    output: detail?.O ?? detail?.output,
    r: detail?.r,
    b: detail?.b,
    t_ref: detail?.t_ref,
    t_last_fire: detail?.t_last_fire,
    M_vector: detail?.M_vector,
    pq_len: detail?.pq_len,
    params: detail?.params,
    synapses: detail?.synapses,
    terminals: detail?.terminals,
    postsynaptic: detail?.postsynaptic,
    presynaptic: detail?.presynaptic,
  };

  Object.assign(merged, liveData);
  if (detail) {
    const params = normalizeParams(detail.params);
    merged.label = detail.name ?? merged.label;
    merged.paula_id = detail.paula_id ?? detail.id ?? merged.paula_id;
    merged.neuron_id = detail.paula_id ?? detail.id ?? merged.neuron_id;
    merged.membrane_potential = liveData?.membrane_potential ?? detail.S ?? detail.membrane_potential;
    merged.firing_rate = liveData?.firing_rate ?? detail.F_avg ?? detail.firing_rate;
    merged.output = liveData?.output ?? detail.O ?? detail.output;
    merged.r = liveData?.r ?? detail.r ?? numericParam(params.r_base);
    merged.b = liveData?.b ?? detail.b ?? numericParam(params.b_base);
    merged.t_ref =
      liveData?.t_ref ??
      detail.t_ref ??
      derivedTRef(params.c, params.num_inputs);
    merged.t_last_fire = liveData?.t_last_fire ?? detail.t_last_fire;
    merged.M_vector =
      liveData?.M_vector ??
      detail.M_vector ??
      zeroVector(params.num_neuromodulators);
    merged.pq_len = liveData?.pq_len ?? detail.pq_len;
    merged.params = detail.params ? params : merged.params;
    merged.synapses = detail.synapses ?? merged.synapses;
    merged.terminals = detail.terminals ?? merged.terminals;
    merged.postsynaptic = detail.postsynaptic ?? merged.postsynaptic;
    merged.presynaptic = detail.presynaptic ?? merged.presynaptic;
  }
  return merged;
}

function normalizeParams(params: unknown) {
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return {};
  }
  const out = { ...(params as Record<string, unknown>) };
  if (out.lambda_param === undefined && out.lambda !== undefined) {
    out.lambda_param = out.lambda;
  }
  return out;
}

function groupParams(params: Record<string, unknown>) {
  const ordered = orderedParamEntries(params);
  const groups = {
    scalars: [] as Array<{ key: string; value: unknown }>,
    vectors: [] as Array<{ key: string; value: unknown }>,
    other: [] as Array<{ key: string; value: unknown }>,
  };
  for (const entry of ordered) {
    if (Array.isArray(entry.value)) {
      groups.vectors.push(entry);
    } else if (isDisplayScalar(entry.value)) {
      groups.scalars.push(entry);
    } else {
      groups.other.push(entry);
    }
  }
  return groups;
}

function orderedParamEntries(params: Record<string, unknown>) {
  const seen = new Set<string>();
  const entries: Array<{ key: string; value: unknown }> = [];
  for (const key of PARAM_ORDER) {
    if (params[key] === undefined) continue;
    entries.push({ key, value: params[key] });
    seen.add(key);
  }
  return entries.concat(
    Object.keys(params)
      .filter((key) => !seen.has(key) && key !== "lambda")
      .sort((a, b) => a.localeCompare(b))
      .map((key) => ({ key, value: params[key] })),
  );
}

function isDisplayScalar(value: unknown) {
  return value === null || ["number", "string", "boolean", "undefined"].includes(typeof value);
}

function numericParam(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function derivedTRef(c: unknown, numInputs: unknown) {
  const cValue = numericParam(c);
  const inputCount = numericParam(numInputs);
  return cValue !== undefined && inputCount !== undefined ? cValue * inputCount : undefined;
}

function zeroVector(length: unknown) {
  const size = Math.max(0, Math.min(16, Math.floor(Number(length) || 0)));
  return size > 0 ? Array.from({ length: size }, () => 0) : undefined;
}

function incidentEdges(node: GraphNode | null, state: NetworkState | null) {
  const empty = { incoming: [] as GraphEdge[], outgoing: [] as GraphEdge[] };
  if (!node || !state) return empty;
  const incoming: GraphEdge[] = [];
  const outgoing: GraphEdge[] = [];
  for (const edge of state.elements.edges) {
    if (edge.data.target === node.data.id) incoming.push(edge);
    if (edge.data.source === node.data.id) outgoing.push(edge);
  }
  return { incoming: incoming.slice(0, 64), outgoing: outgoing.slice(0, 64) };
}

function postsynapticRows(data: SheetNeuronData | null, incoming: GraphEdge[]) {
  const explicit = Array.isArray(data?.postsynaptic) ? data.postsynaptic : [];
  if (explicit.length > 0) {
    return explicit.map((row, index) => [
      String(row.id ?? index),
      String(row.pre_name ?? row.pre_paula_id ?? "-"),
      formatScalar(row.info),
      formatScalar(row.plast),
      formatScalar(row.potential),
      formatScalar(row.distance_to_hillock),
      formatVector(row.adapt),
    ]);
  }
  return incoming.map((edge, index) => [
    String(edge.data.target_synapse ?? index),
    edge.data.source,
    "-",
    "-",
    "-",
    "-",
    "-",
  ]);
}

function presynapticRows(data: SheetNeuronData | null, outgoing: GraphEdge[]) {
  const explicit = Array.isArray(data?.presynaptic) ? data.presynaptic : [];
  const targetLabelsByTerminal = terminalTargetLabels(outgoing);
  if (explicit.length > 0) {
    return explicit.map((row, index) => [
      String(row.id ?? index),
      formatTargets(row.targets) || targetLabelsByTerminal.get(String(row.id)) || "-",
      formatScalar(row.u_o_info),
      formatScalar(row.u_i_retro),
      formatScalar(row.distance_from_hillock),
      formatVector(row.u_o_mod),
    ]);
  }
  return outgoing.map((edge, index) => [
    String(edge.data.source_terminal ?? index),
    edge.data.target,
    "-",
    "-",
    "-",
    "-",
  ]);
}

function terminalTargetLabels(outgoing: GraphEdge[]) {
  const labels = new Map<string, string[]>();
  for (const edge of outgoing) {
    const key = String(edge.data.source_terminal ?? "");
    if (!key) continue;
    const next = labels.get(key) ?? [];
    next.push(`${edge.data.target}:${edge.data.target_synapse ?? "-"}`);
    labels.set(key, next);
  }
  return new Map([...labels.entries()].map(([key, value]) => [key, value.join(", ")]));
}

function formatTargets(targets: NeuronPresynapticTargetView[] | undefined) {
  if (!Array.isArray(targets) || targets.length === 0) return "";
  return targets
    .map((target) => {
      const name = target.target_name ?? target.target_paula_id ?? "-";
      return `${name}:${target.target_synapse ?? "-"}`;
    })
    .join(", ");
}

function selectedSeries(data: GraphNodeData | null, history: NetworkState[]) {
  const base = {
    S: [] as number[],
    fire: [] as number[],
    r: [] as number[],
    b: [] as number[],
    tRef: [] as number[],
    m0: [] as number[],
    m1: [] as number[],
  };
  if (!data) return base;
  for (const snapshot of history.slice(-160)) {
    const node = snapshot.elements.nodes.find((candidate) => candidate.data.id === data.id);
    if (!node) continue;
    base.S.push(numberValue(node.data.membrane_potential));
    base.fire.push(Math.abs(numberValue(node.data.output)) > 1e-9 ? 1 : 0);
    base.r.push(numberValue(node.data.r));
    base.b.push(numberValue(node.data.b));
    base.tRef.push(numberValue(node.data.t_ref));
    base.m0.push(Array.isArray(node.data.M_vector) ? numberValue(node.data.M_vector[0]) : 0);
    base.m1.push(Array.isArray(node.data.M_vector) ? numberValue(node.data.M_vector[1]) : 0);
  }
  return base;
}

function SparkRow({
  label,
  values,
  color,
  min,
  max,
}: {
  label: string;
  values: number[];
  color: string;
  min?: number;
  max?: number;
}) {
  const latest = values[values.length - 1] ?? 0;
  const path = sparkPath(values, 152, 30, min, max);
  return (
    <div className="sheetSpark">
      <span>{label}</span>
      <svg viewBox="0 0 152 30" aria-hidden="true">
        <path d={path} style={{ stroke: color }} />
      </svg>
      <strong>{formatScalar(latest)}</strong>
    </div>
  );
}

function SheetMetric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="sheetMetric">
      <span>{label}</span>
      <strong>{typeof value === "string" ? value : formatScalar(value)}</strong>
    </div>
  );
}

function ParamMetric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="sheetMetric">
      <span>{label}</span>
      <strong>{formatParamValue(value)}</strong>
    </div>
  );
}

function VectorReadout({ label, values }: { label: string; values: unknown[] }) {
  return (
    <div className="sheetVector">
      <span>{label}</span>
      <strong>{formatVector(values)}</strong>
    </div>
  );
}

function SheetTable({ columns, rows }: { columns: string[]; rows: string[][] }) {
  if (rows.length === 0) {
    return (
      <div className="sheetEmpty">
        <Activity size={16} />
        <span>None</span>
      </div>
    );
  }
  return (
    <div className="sheetTableWrap">
      <table className="sheetTable">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex}>
                  {cellIndex === 0 && <CircleStop size={9} />}
                  <span>{cell}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function sparkPath(
  values: number[],
  width: number,
  height: number,
  forcedMin?: number,
  forcedMax?: number,
) {
  if (values.length === 0) return "";
  if (values.length === 1) return `M 0 ${height / 2} L ${width} ${height / 2}`;
  const min = forcedMin ?? Math.min(...values);
  const max = forcedMax ?? Math.max(...values);
  const span = Math.max(1e-9, max - min);
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / span) * (height - 6) - 3;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function numberValue(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatScalar(value: unknown) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (Math.abs(number) >= 100) return number.toFixed(1);
  if (Math.abs(number) >= 10) return number.toFixed(2);
  return number.toFixed(3);
}

function formatVector(values: unknown) {
  if (!Array.isArray(values) || values.length === 0) return "-";
  return `[${values.map((value) => formatScalar(value)).join(", ")}]`;
}

function formatParamValue(value: unknown) {
  if (typeof value === "number") return formatScalar(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "-";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
