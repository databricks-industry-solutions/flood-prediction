import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PickingInfo } from "@deck.gl/core";
import { FloodMap, THEMES, type ViewState, type MapTheme } from "./FloodMap";
import { AddressCard } from "./AddressCard";
import { Chat } from "./Chat";
import {
  api,
  type AOI,
  type AddressLookup,
  type PredictionCell,
  type FloodEvent,
  type Metrics,
  type BuildingExposure,
} from "./api";

const RISK_TIER_LABEL: Record<string, string> = {
  low: "Low",
  moderate: "Moderate",
  high: "High",
  severe: "Severe",
};

function fmtUsd(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000)     return `$${(n / 1_000).toFixed(1)}K`;
  return `$${Math.round(n)}`;
}

function isBuildingExposure(o: unknown): o is BuildingExposure {
  return !!o && typeof o === "object"
    && "expected_loss_usd" in (o as Record<string, unknown>)
    && "risk_tier" in (o as Record<string, unknown>);
}

const DEFAULT_VIEW: ViewState = {
  longitude: -73.65,
  latitude: 45.55,
  zoom: 10,
  pitch: 30,
  bearing: 0,
};

function snap(target: number, options: number[]): number {
  if (!options.length) return target;
  return options.reduce((best, v) =>
    Math.abs(v - target) < Math.abs(best - target) ? v : best,
  );
}

export default function App() {
  const [aois, setAois] = useState<AOI[]>([]);
  const [aoi, setAoi] = useState<string>("greater_montreal");
  const [scenarios, setScenarios] = useState<number[]>([]);
  const [scenarioMm, setScenarioMm] = useState(60);
  const [threshold, setThreshold] = useState(0.4);
  const [showFloods, setShowFloods] = useState(false);
  const [theme, setTheme] = useState<MapTheme>(() => {
    const saved = localStorage.getItem("flood-map-theme");
    return (saved === "dark" || saved === "light" || saved === "voyager") ? saved : "dark";
  });

  useEffect(() => {
    localStorage.setItem("flood-map-theme", theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const [cells, setCells] = useState<PredictionCell[]>([]);
  const [events, setEvents] = useState<FloodEvent[]>([]);
  // Cache the (potentially large) flood-overlay polygons per AOI so toggling
  // the overlay off/on doesn't refetch.
  const eventsCacheRef = useRef<Map<string, FloodEvent[]>>(new Map());
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  // Start true so the initial AOI → scenarios → predictions chain shows a
  // loader from first paint instead of an empty map.
  const [loading, setLoading] = useState(true);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [view, setView] = useState<ViewState>(DEFAULT_VIEW);
  const [hover, setHover] = useState<PickingInfo | null>(null);
  const [addressHit, setAddressHit] = useState<AddressLookup | null>(null);
  const [showBuildings, setShowBuildings] = useState(false);
  // Buildings layer has its own probability floor, decoupled from the hex
  // layer's `threshold`. Default 0.2 keeps the layer focused on cells with
  // material risk; drop to 0 to paint every building (slow at full AOI zoom).
  const [buildingsThreshold, setBuildingsThreshold] = useState(0.2);
  const [buildings, setBuildings] = useState<BuildingExposure[]>([]);
  const [buildingsLoading, setBuildingsLoading] = useState(false);
  const [buildingsTotalEl, setBuildingsTotalEl] = useState(0);
  const [footprintSource, setFootprintSource] = useState<"real" | "synthesized">("synthesized");

  const onAddressResult = useCallback((hit: AddressLookup | null) => {
    setAddressHit(hit);
    if (hit) {
      setView({ longitude: hit.lon, latitude: hit.lat, zoom: 14, pitch: 30, bearing: 0 });
    }
  }, []);

  useEffect(() => {
    api.listAoi()
      .then((rows) => {
        setAois(rows);
        if (rows.length && !rows.find((r) => r.aoi_name === aoi)) setAoi(rows[0].aoi_name);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    const found = aois.find((a) => a.aoi_name === aoi);
    if (found) {
      setView({
        longitude: (found.min_lon + found.max_lon) / 2,
        latitude: (found.min_lat + found.max_lat) / 2,
        zoom: 10, pitch: 30, bearing: 0,
      });
    }
  }, [aoi, aois]);

  useEffect(() => {
    setLoading(true);
    api.listScenarios(aoi)
      .then((r) => {
        setScenarios(r.scenarios_24h_mm);
        if (r.scenarios_24h_mm.length) {
          setScenarioMm((prev) => snap(prev, r.scenarios_24h_mm));
        } else {
          setCells([]);
          setMetrics(null);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, [aoi]);

  useEffect(() => {
    if (!scenarios.length) return;
    let cancelled = false;
    setLoading(true);
    const snapped = snap(scenarioMm, scenarios);
    Promise.all([
      api.predictions(aoi, snapped, threshold),
      api.metrics(aoi, snapped, Math.max(0.5, threshold)),
    ])
      .then(([p, m]) => {
        if (cancelled) return;
        setCells(p.cells);
        setMetrics(m);
      })
      .catch((err) => console.error(err))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [aoi, scenarioMm, threshold, scenarios]);

  // Lazy + batched flood-overlay loader. Only runs when the user actually
  // toggles the overlay on, fetches one year at a time in parallel, and
  // appends each batch to state so the map paints them progressively
  // instead of waiting for one giant payload.
  useEffect(() => {
    if (!showFloods) {
      setOverlayLoading(false);
      return;
    }
    const cached = eventsCacheRef.current.get(aoi);
    if (cached) {
      setEvents(cached);
      setOverlayLoading(false);
      return;
    }
    let cancelled = false;
    setEvents([]);
    setOverlayLoading(true);
    const acc: FloodEvent[] = [];
    (async () => {
      try {
        const years = await api.floodEventYears(aoi);
        if (cancelled) return;
        const list = years.length ? years : [undefined];
        await Promise.all(
          list.map(async (y) => {
            const r = await api.floodEvents(aoi, y);
            if (cancelled) return;
            acc.push(...r.events);
            setEvents([...acc]);
          }),
        );
        if (!cancelled) eventsCacheRef.current.set(aoi, acc);
      } catch (err) {
        console.error(err);
      } finally {
        if (!cancelled) setOverlayLoading(false);
      }
    })();
    return () => {
      cancelled = true;
      setOverlayLoading(false);
    };
  }, [aoi, showFloods]);

  // Approximate the current map viewport as a lon/lat bbox so the buildings
  // endpoint only ships footprints the user is actually looking at. This is
  // intentionally crude (Web-Mercator-flat, no rotation/pitch correction) -
  // we pad by 25% so panning a little doesn't immediately refetch, and we
  // round to the nearest snap value so tiny zoom drifts don't churn the API.
  const viewportBbox = useMemo(() => {
    const z = view.zoom;
    const halfLatDeg = 180 / Math.pow(2, z) * 0.6;
    const halfLonDeg = halfLatDeg / Math.max(Math.cos(view.latitude * Math.PI / 180), 0.1);
    const pad = 1.25;
    return {
      minLon: view.longitude - halfLonDeg * pad,
      maxLon: view.longitude + halfLonDeg * pad,
      minLat: view.latitude  - halfLatDeg * pad,
      maxLat: view.latitude  + halfLatDeg * pad,
    };
  }, [view.longitude, view.latitude, view.zoom]);

  // Debounce + lazy-fetch the buildings layer. Only runs when the toggle is
  // on; refetches when AOI / scenario / threshold / viewport change.
  useEffect(() => {
    if (!showBuildings || !scenarios.length) {
      setBuildingsLoading(false);
      return;
    }
    const snapped = snap(scenarioMm, scenarios);
    let cancelled = false;
    setBuildingsLoading(true);
    const handle = window.setTimeout(() => {
      api.buildingsAtRisk(aoi, snapped, buildingsThreshold, viewportBbox)
        .then((r) => {
          if (cancelled) return;
          setBuildings(r.buildings);
          setBuildingsTotalEl(r.total_expected_loss_usd);
          setFootprintSource(r.footprint_source);
        })
        .catch((err) => console.error(err))
        .finally(() => !cancelled && setBuildingsLoading(false));
    }, 350);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [showBuildings, aoi, scenarioMm, buildingsThreshold, scenarios, viewportBbox]);

  // Drop stale buildings when the layer is toggled off so the next enable
  // doesn't flash old polygons before the new fetch lands.
  useEffect(() => {
    if (!showBuildings) {
      setBuildings([]);
      setBuildingsTotalEl(0);
    }
  }, [showBuildings]);

  const onHover = useCallback((info: PickingInfo) => setHover(info), []);

  const snappedScenario = useMemo(
    () => (scenarios.length ? snap(scenarioMm, scenarios) : scenarioMm),
    [scenarioMm, scenarios],
  );

  const tooltip = useMemo(() => {
    if (!hover || !hover.object) return null;

    // GeoJsonLayer wraps the building in a Feature, so unwrap properties.
    const obj = hover.object as { properties?: unknown } | BuildingExposure | PredictionCell;
    const maybeBuilding = "properties" in obj ? obj.properties : obj;
    if (isBuildingExposure(maybeBuilding)) {
      const b = maybeBuilding;
      return (
        <div className="tooltip tooltip-uw" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <div className="tooltip-title">
            {b.residential ? "Residential" : "Commercial / other"}
            {b.building_type && b.building_type !== "yes" && (
              <span className="tooltip-sub"> · {b.building_type}</span>
            )}
          </div>
          <div className={`tooltip-tier tier-${b.risk_tier}`}>
            {RISK_TIER_LABEL[b.risk_tier]} risk
          </div>
          <div>Flood prob: <b>{(b.flood_prob * 100).toFixed(1)}%</b></div>
          <div>Replacement cost: <b>{fmtUsd(b.brc_usd)}</b></div>
          <div>Loss severity: <b>{(b.loss_severity * 100).toFixed(0)}%</b></div>
          <div className="tooltip-el">
            Expected loss: <b>{fmtUsd(b.expected_loss_usd)}</b>
          </div>
          <div className="tooltip-foot">H3 {b.h3} · {b.osm_id}</div>
        </div>
      );
    }

    const d = hover.object as PredictionCell;
    return (
      <div className="tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
        <div>H3: <b>{d.h3}</b></div>
        <div>Flood prob: <b>{(d.flood_prob * 100).toFixed(1)}%</b></div>
        {d.min_elev !== null && <div>Min elev: {d.min_elev.toFixed(1)} m</div>}
        {d.slope_deg !== null && <div>Slope: {d.slope_deg.toFixed(2)}&deg;</div>}
        {d.dist_to_water_m !== null && (
          <div>Distance to water: {Math.round(d.dist_to_water_m)} m</div>
        )}
        {d.annual_precip_mm !== null && (
          <div>Annual precip: {Math.round(d.annual_precip_mm)} mm</div>
        )}
        {d.max24h_precip_mm !== null && (
          <div>Historic max 24h: {Math.round(d.max24h_precip_mm)} mm</div>
        )}
        {d.building_count != null && d.building_count > 0 && (
          <div>Buildings: <b>{d.building_count.toLocaleString()}</b></div>
        )}
        {d.expected_buildings_at_risk != null && d.expected_buildings_at_risk > 0.5 && (
          <div>Expected at risk: <b>{d.expected_buildings_at_risk.toFixed(1)}</b></div>
        )}
        <div>Historical flood: {d.label_real ? "yes" : "no"}</div>
      </div>
    );
  }, [hover]);

  const scenarioMin = scenarios[0] ?? 0;
  const scenarioMax = scenarios[scenarios.length - 1] ?? 200;
  const scenarioStep = scenarios.length > 1
    ? Math.max(1, Math.min(...scenarios.slice(1).map((v, i) => v - scenarios[i])))
    : 10;

  return (
    <div className="app">
      <div className="map">
        <FloodMap
          viewState={view}
          onViewStateChange={setView}
          cells={cells}
          events={events}
          showRealFloods={showFloods}
          buildings={buildings}
          showBuildings={showBuildings}
          theme={theme}
          pin={addressHit ? { lat: addressHit.lat, lon: addressHit.lon, prob: addressHit.flood_prob } : null}
          onHover={onHover}
        />
      </div>

      <aside className="panel">
        <h1>Underwriting Flood Risk</h1>
        <p className="subtitle">Scenario-based flood risk and property exposure</p>

        <AddressCard
          aoi={aoi}
          scenarioMm={snappedScenario}
          onResult={onAddressResult}
          result={addressHit}
        />

        <div className="row">
          <label htmlFor="aoi">AOI</label>
          <select id="aoi" value={aoi} onChange={(e) => setAoi(e.target.value)}>
            {aois.map((a) => (
              <option key={a.aoi_name} value={a.aoi_name}>{a.aoi_name}</option>
            ))}
          </select>
        </div>

        <div className="row">
          <label>Basemap</label>
          <div className="segmented">
            {(Object.keys(THEMES) as MapTheme[]).map((k) => (
              <button
                key={k}
                type="button"
                className={theme === k ? "seg-on" : ""}
                onClick={() => setTheme(k)}
              >
                {THEMES[k].label}
              </button>
            ))}
          </div>
        </div>

        <div className="row">
          <label htmlFor="rain">24h rainfall</label>
          <input
            id="rain"
            type="range"
            min={scenarioMin}
            max={scenarioMax}
            step={scenarioStep}
            value={scenarioMm}
            onChange={(e) => setScenarioMm(parseInt(e.target.value, 10))}
          />
          <span className="value">{snappedScenario} mm</span>
        </div>

        <div className="row">
          <label htmlFor="thr">Min flood probability</label>
          <input
            id="thr"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
          <span className="value">{threshold.toFixed(2)}</span>
        </div>

        <label className="toggle">
          <input
            type="checkbox"
            checked={showFloods}
            onChange={(e) => setShowFloods(e.target.checked)}
          />
          Overlay 2017/2019 historical floods
          {overlayLoading && <span className="toggle-loading">loading…</span>}
        </label>

        <label className="toggle">
          <input
            type="checkbox"
            checked={showBuildings}
            onChange={(e) => setShowBuildings(e.target.checked)}
          />
          Building exposure (underwriting)
          {buildingsLoading && <span className="toggle-loading">loading…</span>}
        </label>

        {showBuildings && (
          <div className="row row-sub">
            <label htmlFor="bld-thr">Building risk floor</label>
            <input
              id="bld-thr"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={buildingsThreshold}
              onChange={(e) => setBuildingsThreshold(parseFloat(e.target.value))}
            />
            <span className="value">{buildingsThreshold.toFixed(2)}</span>
          </div>
        )}

        <div className="metrics">
          <div className="kpi">
            <span>Scenario</span>
            <span className="v">{snappedScenario} mm / 24h</span>
          </div>
          <div className="kpi">
            <span>Cells shown</span><span className="v">{cells.length.toLocaleString()}</span>
          </div>
          {metrics && (
            <>
              <div className="kpi">
                <span>Mean probability</span>
                <span className="v">{(metrics.mean_prob * 100).toFixed(1)}%</span>
              </div>
              <div className="kpi">
                <span>High-risk rate @ 0.5</span>
                <span className="v">{(metrics.high_risk_rate * 100).toFixed(1)}%</span>
              </div>
              <div className="kpi">
                <span>Real flood cells</span>
                <span className="v">{metrics.real_flood_cells.toLocaleString()}</span>
              </div>
              {metrics.precision_vs_real !== null && (
                <div className="kpi">
                  <span>Precision vs 2017/2019</span>
                  <span className="v">{(metrics.precision_vs_real * 100).toFixed(1)}%</span>
                </div>
              )}
              {metrics.recall_vs_real !== null && (
                <div className="kpi">
                  <span>Recall vs 2017/2019</span>
                  <span className="v">{(metrics.recall_vs_real * 100).toFixed(1)}%</span>
                </div>
              )}
              <div className="kpi kpi-section">Exposure</div>
              <div className="kpi">
                <span>Expected buildings at risk</span>
                <span className="v">
                  {Math.round(metrics.expected_buildings_at_risk).toLocaleString()}
                </span>
              </div>
              <div className="kpi">
                <span>of which residential</span>
                <span className="v">
                  {Math.round(metrics.expected_residential_at_risk).toLocaleString()}
                </span>
              </div>
              <div className="kpi">
                <span>High-risk cells with buildings</span>
                <span className="v">
                  {metrics.high_risk_cells_with_buildings.toLocaleString()}
                </span>
              </div>
            </>
          )}
          {showBuildings && (
            <>
              <div className="kpi kpi-section">Underwriting (visible)</div>
              <div className="kpi">
                <span>Buildings rendered</span>
                <span className="v">{buildings.length.toLocaleString()}</span>
              </div>
              <div className="kpi">
                <span>Aggregate expected loss</span>
                <span className="v">{fmtUsd(buildingsTotalEl)}</span>
              </div>
              <div className="kpi kpi-foot">
                Footprints: {footprintSource === "real" ? "real (silver_building_footprints)" : "synthesized from centroids"}
              </div>
            </>
          )}
        </div>
      </aside>

      <Chat aoi={aoi} scenarioMm={snappedScenario} />

      <div className="legend">
        <div>Flood probability - {snappedScenario} mm / 24 h scenario</div>
        <div className="bar" />
        <div className="bar-labels"><span>0</span><span>0.5</span><span>1</span></div>
        {showFloods && events.length > 0 && (
          <div className="legend-years">
            <span className="swatch swatch-2017" /> 2017 flood
            <span className="swatch swatch-2019" /> 2019 flood
          </div>
        )}
        {showBuildings && (
          <div className="legend-tiers">
            <span className="swatch swatch-tier tier-low" /> low
            <span className="swatch swatch-tier tier-moderate" /> moderate
            <span className="swatch swatch-tier tier-high" /> high
            <span className="swatch swatch-tier tier-severe" /> severe
          </div>
        )}
      </div>

      {loading && (
        <div
          className={`loader${cells.length === 0 ? " loader-initial" : ""}`}
          data-testid="prediction-loading"
          role="status"
          aria-live="polite"
        >
          <div className="loader-spinner" aria-hidden="true" />
          <span>{cells.length === 0 ? "Loading flood risk data…" : "Updating predictions…"}</span>
        </div>
      )}
      {overlayLoading && !loading && (
        <div className="spinner">Loading historical flood overlay…</div>
      )}
      {tooltip}
    </div>
  );
}
