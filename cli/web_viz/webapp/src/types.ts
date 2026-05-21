export type LayoutName = "layers" | "grid" | "circle" | "concentric";

export interface ClientConfig {
  renderer: "canvas";
  websocket_url: string;
  update_interval: number;
}

export interface GraphNodeData {
  id: string;
  label: string;
  type: "neuron" | "external" | string;
  neuron_id?: number | string;
  target_neuron?: number | string;
  target_synapse?: number | string;
  membrane_potential?: number;
  firing_rate?: number;
  t_ref?: number;
  output?: number;
  r?: number;
  b?: number;
  t_last_fire?: number | null;
  M_vector?: number[];
  pq_len?: number;
  postsynaptic?: NeuronPostsynapticView[];
  presynaptic?: NeuronPresynapticView[];
  params?: NeuronParamsView;
  activity_level?: string;
  synapses?: Array<number | string>;
  terminals?: Array<number | string>;
  layer?: number | string | null;
  layer_name?: string;
  layer_key?: string;
  layer_type?: string;
  filter?: number | string;
  kernel_size?: number | string | Array<number | string>;
  stride?: number | string | Array<number | string>;
  in_channels?: number | string;
  in_height?: number | string;
  in_width?: number | string;
  out_height?: number | string;
  out_width?: number | string;
  x_index?: number | string;
  y_index?: number | string;
  position?: { x: number; y: number };
  base_color?: string;
  base_size?: number;
}

export interface GraphNode {
  data: GraphNodeData;
  position?: { x: number; y: number };
}

export interface GraphEdgeData {
  id: string;
  source: string;
  target: string;
  type: "neuron" | "external" | string;
  source_terminal?: number;
  target_synapse?: number;
  source_firing?: boolean;
  weight?: number;
  info?: number;
  plast?: number;
  potential?: number;
  base_color?: string;
  base_width?: number;
}

export interface GraphEdge {
  data: GraphEdgeData;
}

export interface TravelingSignal {
  id: string;
  source: string;
  target: string;
  progress: number;
  event_type: string;
  color: string;
  size: number;
  arrival_tick: number;
  start_tick?: number;
}

export interface NetworkStatistics {
  current_tick: number;
  is_running: boolean;
  tick_rate: number;
  num_neurons: number;
  num_connections: number;
  num_external_inputs: number;
  num_traveling_signals: number;
  active_neurons: number;
  avg_potential: number;
  avg_firing_rate: number;
  avg_t_ref: number;
  max_potential: number;
  free_energy?: number | null;
  state_energy?: number;
  synaptic_density: number;
  graph_density: number;
}

export interface NeuronPostsynapticView {
  id?: number | string;
  pre_paula_id?: number | string | null;
  pre_terminal?: number | string | null;
  pre_name?: string | null;
  info?: number;
  plast?: number;
  adapt?: number[];
  potential?: number;
  distance_to_hillock?: number;
}

export interface NeuronPresynapticView {
  id?: number | string;
  u_o_info?: number;
  u_o_mod?: number[];
  u_i_retro?: number;
  distance_from_hillock?: number;
  targets?: NeuronPresynapticTargetView[];
}

export interface NeuronPresynapticTargetView {
  target_paula_id?: number | string | null;
  target_synapse?: number | string | null;
  target_name?: string | null;
}

export interface NeuronParamsView {
  r_base?: number;
  b_base?: number;
  c?: number;
  lambda_param?: number;
  p?: number;
  eta_post?: number;
  eta_retro?: number;
  delta_decay?: number;
  beta_avg?: number;
  gamma?: number[];
  w_r?: number[];
  w_b?: number[];
  w_tref?: number[];
  num_neuromodulators?: number;
  num_inputs?: number;
  [key: string]: unknown;
}

export interface NetworkState {
  elements: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  traveling_signals: TravelingSignal[];
  statistics: NetworkStatistics;
  stimulus?: StimulusMetadata;
  current_tick: number;
  is_running: boolean;
}

export interface StimulusPrediction {
  rank?: number;
  label?: number | string | null;
  confidence?: number | null;
}

export interface StimulusMetadata {
  active?: boolean;
  sequence?: number;
  updated_at_tick?: number | string | null;
  server_time_ms?: number;
  label?: number | string | null;
  class_name?: string | null;
  presentation_id?: number | string | null;
  sample_index?: number | null;
  dataset_name?: string | null;
  epoch?: number | string | null;
  source?: string | null;
  predicted_label?: number | string | null;
  confidence?: number | null;
  second_predicted_label?: number | string | null;
  second_confidence?: number | null;
  third_predicted_label?: number | string | null;
  third_confidence?: number | null;
  predictions?: StimulusPrediction[];
  tags?: string[];
  extra?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface NeuronDetailData {
  id: number | string;
  paula_id?: number | string;
  name?: string | null;
  S?: number;
  O?: number;
  r?: number;
  b?: number;
  t_ref?: number;
  F_avg?: number;
  t_last_fire?: number | null;
  M_vector?: number[];
  pq_len?: number;
  membrane_potential?: number;
  firing_rate?: number;
  output?: number;
  metadata?: Record<string, unknown>;
  params?: NeuronParamsView;
  synapses?: Array<number | string>;
  terminals?: Array<number | string>;
  postsynaptic?: NeuronPostsynapticView[];
  presynaptic?: NeuronPresynapticView[];
}

export interface TickResult {
  error?: string;
  tick?: number;
  total_activity?: number;
}

export interface FrameStats {
  fps: number;
  nodes: number;
  edges: number;
}
