export type AOI = {
  aoi_name: string;
  min_lon: number; min_lat: number; max_lon: number; max_lat: number;
};

export type PredictionCell = {
  h3: string;
  flood_prob: number;
  min_elev: number | null;
  slope_deg: number | null;
  dist_to_water_m: number | null;
  annual_precip_mm: number | null;
  max24h_precip_mm: number | null;
  building_count: number | null;
  residential_count: number | null;
  expected_buildings_at_risk: number | null;
  label_real: number;
  geometry: GeoJSON.Polygon | Record<string, unknown>;
};

export type PredictionsResponse = {
  aoi_name: string;
  scenario_24h_mm: number;
  threshold: number;
  count: number;
  cells: PredictionCell[];
};

export type FloodEvent = { year: string; geometry: GeoJSON.Geometry };
export type FloodEventsResponse = { aoi_name: string; count: number; events: FloodEvent[] };

export type ScenariosResponse = { aoi_name: string; scenarios_24h_mm: number[] };

export type Metrics = {
  aoi_name: string;
  scenario_24h_mm: number;
  total_cells: number;
  mean_prob: number;
  high_risk_rate: number;
  real_flood_cells: number;
  precision_vs_real: number | null;
  recall_vs_real: number | null;
  expected_buildings_at_risk: number;
  expected_residential_at_risk: number;
  high_risk_cells_with_buildings: number;
};

export type RiskTier = "low" | "moderate" | "high" | "severe";

export type BuildingExposure = {
  osm_id: string;
  building_type: string | null;
  residential: number;
  h3: string;
  flood_prob: number;
  risk_tier: RiskTier;
  brc_usd: number;
  loss_severity: number;
  expected_loss_usd: number;
  geometry: GeoJSON.Polygon | Record<string, unknown>;
};

export type BuildingsAtRiskResponse = {
  aoi_name: string;
  scenario_24h_mm: number;
  threshold: number;
  count: number;
  total_expected_loss_usd: number;
  footprint_source: "real" | "synthesized";
  buildings: BuildingExposure[];
};

export type AddressLookup = {
  query: string;
  resolved_name: string;
  lat: number;
  lon: number;
  h3: string;
  scenario_24h_mm: number;
  flood_prob: number;
  min_elev: number | null;
  slope_deg: number | null;
  dist_to_water_m: number | null;
  annual_precip_mm: number | null;
  max24h_precip_mm: number | null;
  building_count: number | null;
  residential_count: number | null;
  expected_buildings_at_risk: number | null;
  sweep: [number, number][];
};

export type ChatStatus = "PENDING" | "COMPLETED" | "FAILED";

export type ChatMessage = {
  conversation_id: string;
  message_id: string;
  status: ChatStatus;
  text: string | null;
  sql: string | null;
  columns: string[] | null;
  rows: (string | number | boolean | null)[][] | null;
  row_count: number | null;
  truncated: boolean;
  error: string | null;
};

export type ChatHealth = { enabled: boolean; space_id: string | null };

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${url} -> ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

function postJson<T>(url: string, body: unknown): Promise<T> {
  return fetchJson<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  listAoi: () => fetchJson<AOI[]>("/api/aoi"),
  listScenarios: (aoi: string) =>
    fetchJson<ScenariosResponse>(`/api/scenarios?aoi=${encodeURIComponent(aoi)}`),
  predictions: (aoi: string, scenarioMm: number, threshold: number, limit = 20000) =>
    fetchJson<PredictionsResponse>(
      `/api/predictions?aoi=${encodeURIComponent(aoi)}&scenario_mm=${scenarioMm}` +
        `&threshold=${threshold}&limit=${limit}`,
    ),
  floodEventYears: (aoi: string) =>
    fetchJson<string[]>(`/api/flood_event_years?aoi=${encodeURIComponent(aoi)}`),
  floodEvents: (aoi: string, year?: string) => {
    const qs = new URLSearchParams({ aoi });
    if (year) qs.set("year", year);
    return fetchJson<FloodEventsResponse>(`/api/flood_events?${qs}`);
  },
  metrics: (aoi: string, scenarioMm: number, threshold: number) =>
    fetchJson<Metrics>(
      `/api/metrics?aoi=${encodeURIComponent(aoi)}&scenario_mm=${scenarioMm}&threshold=${threshold}`,
    ),
  buildingsAtRisk: (
    aoi: string,
    scenarioMm: number,
    threshold: number,
    bbox?: { minLon: number; minLat: number; maxLon: number; maxLat: number },
    limit = 4000,
  ) => {
    const qs = new URLSearchParams({
      aoi,
      scenario_mm: String(scenarioMm),
      threshold: String(threshold),
      limit: String(limit),
    });
    if (bbox) {
      qs.set("min_lon", String(bbox.minLon));
      qs.set("min_lat", String(bbox.minLat));
      qs.set("max_lon", String(bbox.maxLon));
      qs.set("max_lat", String(bbox.maxLat));
    }
    return fetchJson<BuildingsAtRiskResponse>(`/api/buildings_at_risk?${qs}`);
  },
  lookup: (aoi: string, scenarioMm: number, query: string) =>
    fetchJson<AddressLookup>(
      `/api/lookup?aoi=${encodeURIComponent(aoi)}&scenario_mm=${scenarioMm}` +
        `&q=${encodeURIComponent(query)}`,
    ),
  chatHealth: () => fetchJson<ChatHealth>("/api/chat/health"),
  chatStart: (content: string, aoi?: string, scenarioMm?: number) =>
    postJson<ChatMessage>("/api/chat/start", {
      content,
      aoi: aoi ?? null,
      scenario_mm: scenarioMm ?? null,
    }),
  chatSend: (conversationId: string, content: string, aoi?: string, scenarioMm?: number) =>
    postJson<ChatMessage>("/api/chat/message", {
      conversation_id: conversationId,
      content,
      aoi: aoi ?? null,
      scenario_mm: scenarioMm ?? null,
    }),
  chatPoll: (conversationId: string, messageId: string) =>
    fetchJson<ChatMessage>(
      `/api/chat/message/${encodeURIComponent(conversationId)}/${encodeURIComponent(messageId)}`,
    ),
};
