import axios from "axios";
import type {
  ClientConfig,
  NetworkState,
  NeuronDetailData,
  StimulusMetadata,
  TickResult,
} from "./types";

const http = axios.create({
  baseURL: "/api",
  timeout: 10_000,
  headers: { "Content-Type": "application/json" },
});

export async function getClientConfig(): Promise<ClientConfig> {
  const r = await http.get<ClientConfig>("/config");
  return r.data;
}

export async function getNetworkState(): Promise<NetworkState> {
  const r = await http.get<NetworkState>("/network/state");
  return r.data;
}

export async function getNeuronDetail(neuronId: number | string): Promise<NeuronDetailData> {
  const r = await http.get<NeuronDetailData>(`/neuron/${encodeURIComponent(String(neuronId))}`);
  return r.data;
}

export async function executeTick(): Promise<TickResult> {
  const r = await http.post<TickResult>("/network/tick", {});
  return r.data;
}

export async function executeTicks(nTicks: number): Promise<{ count: number }> {
  const r = await http.post<{ count: number }>("/network/ticks", {
    n_ticks: nTicks,
  });
  return r.data;
}

export async function startNetwork(tickRate: number): Promise<{ success: boolean }> {
  const r = await http.post<{ success: boolean }>("/network/start", {
    tick_rate: tickRate,
  });
  return r.data;
}

export async function stopNetwork(): Promise<{ success: boolean }> {
  const r = await http.post<{ success: boolean }>("/network/stop", {});
  return r.data;
}

export async function sendSignal(input: {
  neuronId: number;
  synapseId: number;
  strength: number;
}): Promise<{ success: boolean }> {
  const r = await http.post<{ success: boolean }>("/network/signal", {
    neuron_id: input.neuronId,
    synapse_id: input.synapseId,
    strength: input.strength,
  });
  return r.data;
}

export async function setStimulusMetadata(
  metadata: Partial<StimulusMetadata>,
): Promise<StimulusMetadata> {
  const r = await http.post<StimulusMetadata>("/stimulus/metadata", metadata);
  return r.data;
}

export async function clearStimulusMetadata(): Promise<StimulusMetadata> {
  const r = await http.delete<StimulusMetadata>("/stimulus/metadata");
  return r.data;
}
