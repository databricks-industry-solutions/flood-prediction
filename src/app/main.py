"""FastAPI backend for the flood prediction Databricks App.

Serves three JSON endpoints (`/api/aoi`, `/api/predictions`, `/api/flood_events`,
`/api/metrics`) backed by a Databricks SQL warehouse, plus the compiled React SPA
from `./client/dist` at `/`.

Authentication on Databricks Apps is provided by the app resource's service
principal - we pick up the PAT/host from the `DATABRICKS_*` env vars that the
platform injects.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import h3
import requests
from databricks import sql as dbsql
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("flood-app")

CATALOG = os.environ.get("DATABRICKS_CATALOG", "flood_demo")
SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "montreal")
DEFAULT_AOI = os.environ.get("DEFAULT_AOI", "greater_montreal")
MAX_CELLS = int(os.environ.get("MAX_CELLS_PER_REQUEST", "50000"))

# Underwriting assumptions for the /api/buildings_at_risk layer. These are
# intentionally simple "demo-grade" defaults that an underwriter would replace
# with their own per-policy data. Tune via env vars at deploy time.
#   * Building replacement cost (BRC): residential = single-family-home average,
#     other = small-commercial average for a Canadian metro market.
#   * Loss severity (LS): % of BRC paid out at this flood depth scenario.
#     25% is a common FEMA-derived flat assumption for inland flood at ~0.5 m
#     depth and is what we'd use for portfolio-level expected-loss math when
#     we don't have a depth-damage curve per building.
RESIDENTIAL_BRC_USD = float(os.environ.get("RESIDENTIAL_BRC_USD", "300000"))
COMMERCIAL_BRC_USD = float(os.environ.get("COMMERCIAL_BRC_USD", "1500000"))
FLOOD_LOSS_SEVERITY = float(os.environ.get("FLOOD_LOSS_SEVERITY", "0.25"))
MAX_BUILDINGS = int(os.environ.get("MAX_BUILDINGS_PER_REQUEST", "8000"))

# Genie Space backing the underwriter chat panel. Provisioned out-of-band by
# `scripts/genie_bootstrap.py` and wired into the app at deploy time via the
# DATABRICKS_GENIE_SPACE_ID env var. When unset the /api/chat/* routes return
# 503 so the frontend can hide the chat panel cleanly.
GENIE_SPACE_ID = os.environ.get("DATABRICKS_GENIE_SPACE_ID", "").strip()
GENIE_POLL_TIMEOUT_S = float(os.environ.get("GENIE_POLL_TIMEOUT_S", "60"))
GENIE_MAX_RESULT_ROWS = int(os.environ.get("GENIE_MAX_RESULT_ROWS", "200"))

NS = f"`{CATALOG}`.`{SCHEMA}`"

# Databricks Apps injects DATABRICKS_HOST, DATABRICKS_HTTP_PATH (warehouse) and
# DATABRICKS_TOKEN / or oauth automatically when a SQL Warehouse resource is attached.
HOST = os.environ.get("DATABRICKS_HOST") or os.environ.get("DATABRICKS_WORKSPACE_HOSTNAME")
HTTP_PATH = os.environ.get("DATABRICKS_HTTP_PATH") or os.environ.get("DATABRICKS_WAREHOUSE_HTTP_PATH")
TOKEN = os.environ.get("DATABRICKS_TOKEN")
CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")

if not (HOST and HTTP_PATH):
    log.warning("DATABRICKS_HOST / DATABRICKS_HTTP_PATH not set - API calls will fail")


PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE")


def _connect():
    """Open a SQL warehouse connection using whichever auth mode is available.

    Priority:
    1. ``DATABRICKS_TOKEN`` (PAT) — used by Databricks Apps in production.
    2. ``DATABRICKS_CLIENT_ID`` + ``DATABRICKS_CLIENT_SECRET`` (SP OAuth).
    3. ``DATABRICKS_CONFIG_PROFILE`` — local dev via the CLI's stored OAuth
       (U2M) credentials.
    """
    kwargs: dict[str, Any] = {"server_hostname": HOST, "http_path": HTTP_PATH}
    if TOKEN:
        kwargs["access_token"] = TOKEN
    elif CLIENT_ID and CLIENT_SECRET:
        from databricks.sdk.core import Config, oauth_service_principal

        cfg = Config(host=f"https://{HOST}", client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        kwargs["credentials_provider"] = lambda: oauth_service_principal(cfg)
    elif PROFILE:
        from databricks.sdk.core import Config

        cfg = Config(profile=PROFILE)

        def _provider():
            def _header_factory():
                headers = cfg.authenticate()
                # Config.authenticate() returns a dict like {"Authorization": "Bearer ..."}
                return headers

            return _header_factory

        kwargs["credentials_provider"] = _provider
    else:
        raise RuntimeError(
            "No Databricks auth available — set DATABRICKS_TOKEN, "
            "DATABRICKS_CLIENT_ID/SECRET, or DATABRICKS_CONFIG_PROFILE."
        )
    return dbsql.connect(**kwargs)


def _fetch(query: str, params: tuple | None = None) -> list[dict[str, Any]]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, params or ())
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _h3_hex(v: Any) -> str:
    """Normalize an H3 cell id (stored as BIGINT in Delta) to the hex string
    format deck.gl's H3HexagonLayer and h3-js expect."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    # Python int / Decimal -> 15-char hex (H3 is 64-bit; leading zeroes matter)
    return f"{int(v):x}"


# Pydantic schemas ---------------------------------------------------------------


class AOI(BaseModel):
    aoi_name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class PredictionCell(BaseModel):
    h3: str
    flood_prob: float
    min_elev: float | None = None
    slope_deg: float | None = None
    dist_to_water_m: float | None = None
    annual_precip_mm: float | None = None
    max24h_precip_mm: float | None = None
    building_count: int | None = None
    residential_count: int | None = None
    expected_buildings_at_risk: float | None = None
    label_real: int = 0
    geometry: dict[str, Any] = Field(description="GeoJSON Polygon for the H3 cell")


class PredictionsResponse(BaseModel):
    aoi_name: str
    scenario_24h_mm: int
    threshold: float
    count: int
    cells: list[PredictionCell]


class ScenariosResponse(BaseModel):
    aoi_name: str
    scenarios_24h_mm: list[int]


class FloodEvent(BaseModel):
    year: str
    geometry: dict[str, Any]


class FloodEventsResponse(BaseModel):
    aoi_name: str
    count: int
    events: list[FloodEvent]


class Metrics(BaseModel):
    aoi_name: str
    scenario_24h_mm: int
    total_cells: int
    mean_prob: float
    high_risk_rate: float
    real_flood_cells: int
    precision_vs_real: float | None = None
    recall_vs_real: float | None = None
    expected_buildings_at_risk: float = 0.0
    expected_residential_at_risk: float = 0.0
    high_risk_cells_with_buildings: int = 0


class BuildingExposure(BaseModel):
    osm_id: str
    building_type: str | None = None
    residential: int = 0
    h3: str
    flood_prob: float
    risk_tier: str  # one of: low, moderate, high, severe
    brc_usd: float = Field(description="Building replacement cost (USD)")
    loss_severity: float = Field(
        description="Damage ratio applied at this scenario depth (0..1)"
    )
    expected_loss_usd: float = Field(
        description="flood_prob * brc_usd * loss_severity"
    )
    geometry: dict[str, Any] = Field(
        description="GeoJSON Polygon - real footprint when available, "
        "otherwise a small square synthesized around the centroid"
    )


class BuildingsAtRiskResponse(BaseModel):
    aoi_name: str
    scenario_24h_mm: int
    threshold: float
    count: int
    total_expected_loss_usd: float
    footprint_source: str  # "real" | "synthesized"
    buildings: list[BuildingExposure]


class AddressLookup(BaseModel):
    query: str
    resolved_name: str
    lat: float
    lon: float
    h3: str
    scenario_24h_mm: int
    flood_prob: float
    min_elev: float | None = None
    slope_deg: float | None = None
    dist_to_water_m: float | None = None
    annual_precip_mm: float | None = None
    max24h_precip_mm: float | None = None
    building_count: int | None = None
    residential_count: int | None = None
    expected_buildings_at_risk: float | None = None
    sweep: list[tuple[int, float]] = Field(
        default_factory=list,
        description="[(scenario_24h_mm, flood_prob), ...] across all available scenarios",
    )


# App --------------------------------------------------------------------------

app = FastAPI(title="Flood Prediction API", version="0.1.0")
# GeoJSON compresses ~5-10x. Skip tiny payloads where the overhead doesn't pay off.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "catalog": CATALOG, "schema": SCHEMA, "default_aoi": DEFAULT_AOI}


@app.get("/api/aoi", response_model=list[AOI])
def list_aoi() -> list[AOI]:
    rows = _fetch(f"SELECT aoi_name, min_lon, min_lat, max_lon, max_lat FROM {NS}.gold_aoi")
    return [AOI(**r) for r in rows]


@app.get("/api/scenarios", response_model=ScenariosResponse)
def list_scenarios(aoi: str = Query(default=DEFAULT_AOI)) -> ScenariosResponse:
    rows = _fetch(
        f"SELECT DISTINCT scenario_24h_mm FROM {NS}.gold_scenarios"
        " WHERE aoi_name = ? ORDER BY scenario_24h_mm",
        (aoi,),
    )
    return ScenariosResponse(
        aoi_name=aoi,
        scenarios_24h_mm=[int(r["scenario_24h_mm"]) for r in rows],
    )


def _nearest_scenario(aoi: str, mm: int) -> int:
    """Snap arbitrary slider values to one of the pre-scored scenarios."""
    rows = _fetch(
        f"SELECT DISTINCT scenario_24h_mm FROM {NS}.gold_scenarios WHERE aoi_name = ?",
        (aoi,),
    )
    available = sorted(int(r["scenario_24h_mm"]) for r in rows)
    if not available:
        return mm
    return min(available, key=lambda s: abs(s - mm))


@app.get("/api/predictions", response_model=PredictionsResponse)
def predictions(
    aoi: str = Query(default=DEFAULT_AOI),
    scenario_mm: int = Query(default=60, ge=0, le=500,
                              description="24-hour rainfall scenario in mm"),
    threshold: float = Query(default=0.0, ge=0.0, le=1.0,
                              description="Only return cells with flood_prob >= threshold"),
    min_lon: float | None = None,
    min_lat: float | None = None,
    max_lon: float | None = None,
    max_lat: float | None = None,
    limit: int = Query(default=MAX_CELLS, le=MAX_CELLS),
) -> PredictionsResponse:
    scenario = _nearest_scenario(aoi, scenario_mm)

    where = ["aoi_name = ?", "scenario_24h_mm = ?", "flood_prob >= ?"]
    params: list[Any] = [aoi, scenario, threshold]

    if None not in (min_lon, min_lat, max_lon, max_lat):
        # Use Spatial SQL to bbox-filter by the H3 centroid - keeps payloads small.
        where.append(
            "ST_Intersects(ST_GeomFromGeoJSON(h3_centerasgeojson(h3)),"
            " ST_GeomFromText(?, 4326))"
        )
        params.append(
            "POLYGON(({a} {b}, {c} {b}, {c} {d}, {a} {d}, {a} {b}))".format(
                a=min_lon, b=min_lat, c=max_lon, d=max_lat
            )
        )

    where_sql = " AND ".join(where)
    q = f"""
        SELECT h3, flood_prob, min_elev, slope_deg, dist_to_water_m,
               annual_precip_mm, max24h_precip_mm,
               building_count, residential_count, expected_buildings_at_risk,
               label_real, geometry_geojson
        FROM {NS}.gold_h3_flood_predictions
        WHERE {where_sql}
        ORDER BY flood_prob DESC
        LIMIT {int(limit)}
    """
    rows = _fetch(q, tuple(params))
    cells = [
        PredictionCell(
            h3=_h3_hex(r["h3"]),
            flood_prob=float(r["flood_prob"] or 0.0),
            min_elev=float(r["min_elev"]) if r["min_elev"] is not None else None,
            slope_deg=float(r["slope_deg"]) if r["slope_deg"] is not None else None,
            dist_to_water_m=float(r["dist_to_water_m"]) if r["dist_to_water_m"] is not None else None,
            annual_precip_mm=float(r["annual_precip_mm"]) if r["annual_precip_mm"] is not None else None,
            max24h_precip_mm=float(r["max24h_precip_mm"]) if r["max24h_precip_mm"] is not None else None,
            building_count=int(r["building_count"]) if r.get("building_count") is not None else None,
            residential_count=int(r["residential_count"]) if r.get("residential_count") is not None else None,
            expected_buildings_at_risk=(
                float(r["expected_buildings_at_risk"])
                if r.get("expected_buildings_at_risk") is not None else None
            ),
            label_real=int(r["label_real"] or 0),
            geometry=json.loads(r["geometry_geojson"]) if r["geometry_geojson"] else {},
        )
        for r in rows
    ]
    return PredictionsResponse(
        aoi_name=aoi, scenario_24h_mm=scenario,
        threshold=threshold, count=len(cells), cells=cells,
    )


@app.get("/api/flood_event_years", response_model=list[str])
def flood_event_years(aoi: str = Query(default=DEFAULT_AOI)) -> list[str]:
    """Distinct years available for the AOI - lets the client batch the
    overlay download one year at a time and render each batch as it arrives."""
    rows = _fetch(
        f"SELECT DISTINCT year FROM {NS}.gold_flood_events"
        " WHERE aoi_name = ? AND year IS NOT NULL ORDER BY year",
        (aoi,),
    )
    return [str(r["year"]) for r in rows]


@app.get("/api/flood_events", response_model=FloodEventsResponse)
def flood_events(aoi: str = Query(default=DEFAULT_AOI), year: str | None = None) -> FloodEventsResponse:
    params: list[Any] = [aoi]
    extra = ""
    if year:
        extra = " AND year = ?"
        params.append(year)
    rows = _fetch(
        f"SELECT year, geometry_geojson FROM {NS}.gold_flood_events WHERE aoi_name = ?{extra}",
        tuple(params),
    )
    events = [FloodEvent(year=r["year"] or "", geometry=json.loads(r["geometry_geojson"])) for r in rows]
    return FloodEventsResponse(aoi_name=aoi, count=len(events), events=events)


@app.get("/api/metrics", response_model=Metrics)
def metrics(
    aoi: str = Query(default=DEFAULT_AOI),
    scenario_mm: int = Query(default=60, ge=0, le=500),
    threshold: float = 0.5,
) -> Metrics:
    scenario = _nearest_scenario(aoi, scenario_mm)
    row = _fetch(
        f"""
        SELECT COUNT(*)                                                         AS total,
               AVG(flood_prob)                                                  AS mean_prob,
               AVG(CASE WHEN flood_prob >= ? THEN 1.0 ELSE 0.0 END)             AS high_rate,
               SUM(label_real)                                                  AS real_pos,
               SUM(CASE WHEN flood_prob >= ? AND label_real = 1 THEN 1 ELSE 0 END) AS tp,
               SUM(CASE WHEN flood_prob >= ? AND label_real = 0 THEN 1 ELSE 0 END) AS fp,
               SUM(CASE WHEN flood_prob <  ? AND label_real = 1 THEN 1 ELSE 0 END) AS fn,
               SUM(expected_buildings_at_risk)                                    AS exp_bld,
               SUM(flood_prob * COALESCE(residential_count, 0))                   AS exp_res,
               SUM(CASE WHEN flood_prob >= ? AND COALESCE(building_count,0) > 0
                        THEN 1 ELSE 0 END)                                        AS hi_bld_cells
        FROM {NS}.gold_h3_flood_predictions
        WHERE aoi_name = ? AND scenario_24h_mm = ?
        """,
        (threshold, threshold, threshold, threshold, threshold, aoi, scenario),
    )[0]
    tp, fp, fn = (int(row[k] or 0) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return Metrics(
        aoi_name=aoi,
        scenario_24h_mm=scenario,
        total_cells=int(row["total"] or 0),
        mean_prob=float(row["mean_prob"] or 0.0),
        high_risk_rate=float(row["high_rate"] or 0.0),
        real_flood_cells=int(row["real_pos"] or 0),
        precision_vs_real=precision,
        recall_vs_real=recall,
        expected_buildings_at_risk=float(row["exp_bld"] or 0.0),
        expected_residential_at_risk=float(row["exp_res"] or 0.0),
        high_risk_cells_with_buildings=int(row["hi_bld_cells"] or 0),
    )


# Buildings-at-risk -----------------------------------------------------------


@lru_cache(maxsize=4)
def _has_footprints_table() -> bool:
    """Check once per process whether the optional silver footprints table
    exists. Lets us auto-upgrade from synthesized squares to real polygons
    when the pipeline gets re-run with the polygon ingest cell enabled."""
    try:
        _fetch(f"SELECT 1 FROM {NS}.silver_building_footprints LIMIT 1")
        return True
    except Exception as e:  # noqa: BLE001 - table-not-found is the common case
        log.info("silver_building_footprints not available (%s); using centroid fallback", e)
        return False


def _risk_tier(prob: float) -> str:
    if prob >= 0.6:
        return "severe"
    if prob >= 0.3:
        return "high"
    if prob >= 0.1:
        return "moderate"
    return "low"


def _synth_square(lon: float, lat: float, half_meters: float = 6.0) -> dict[str, Any]:
    """Build a small GeoJSON Polygon (~12 m square) around a centroid.

    Used as a visual stand-in when only the centroid is available in
    bronze_buildings. ~12 m matches typical Montreal residential footprints
    and reads as an individual building at city-block zoom levels.

    Latitude degree is ~111,320 m; longitude degree shrinks by cos(lat).
    """
    import math

    dlat = half_meters / 111_320.0
    dlon = half_meters / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - dlon, lat - dlat],
            [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat],
            [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat],
        ]],
    }


@app.get("/api/buildings_at_risk", response_model=BuildingsAtRiskResponse)
def buildings_at_risk(
    aoi: str = Query(default=DEFAULT_AOI),
    scenario_mm: int = Query(default=60, ge=0, le=500),
    threshold: float = Query(
        default=0.3, ge=0.0, le=1.0,
        description="Only return buildings whose enclosing H3 cell has flood_prob >= threshold",
    ),
    min_lon: float | None = None,
    min_lat: float | None = None,
    max_lon: float | None = None,
    max_lat: float | None = None,
    limit: int = Query(default=MAX_BUILDINGS, le=MAX_BUILDINGS),
) -> BuildingsAtRiskResponse:
    """Return building-level flood exposure for the underwriter overlay.

    Each row is the building joined to its H3 cell's flood probability for
    the requested scenario, plus a simple loss model:

        expected_loss = flood_prob * BRC * loss_severity

    where BRC is residential vs commercial replacement cost and loss
    severity is a flat 25% (FEMA-style demo assumption). Real underwriting
    would substitute per-policy insured values and a depth-damage curve.
    """
    scenario = _nearest_scenario(aoi, scenario_mm)
    use_real = _has_footprints_table()

    bbox_clause = ""
    bbox_params: list[Any] = []
    if None not in (min_lon, min_lat, max_lon, max_lat):
        bbox_clause = " AND b.lon BETWEEN ? AND ? AND b.lat BETWEEN ? AND ?"
        bbox_params = [min_lon, max_lon, min_lat, max_lat]

    if use_real:
        # Real polygons live in silver_building_footprints alongside the
        # centroid (lon/lat) so the same bbox filter still works.
        q = f"""
            SELECT b.osm_id, b.building AS building_type, b.residential,
                   b.lon, b.lat, b.geometry_geojson,
                   p.h3, p.flood_prob
            FROM {NS}.silver_building_footprints b
            JOIN {NS}.gold_h3_flood_predictions p
              ON p.aoi_name = b.aoi_name AND p.h3 = b.h3
            WHERE b.aoi_name = ? AND p.scenario_24h_mm = ?
              AND p.flood_prob >= ?{bbox_clause}
            ORDER BY p.flood_prob DESC
            LIMIT {int(limit)}
        """
    else:
        # Fallback: synthesize tiny squares around each centroid in
        # bronze_buildings, join to predictions via the centroid's H3 cell.
        # The h3_longlat_ash3 SQL function ships with Databricks Runtime
        # (h3-spark). Resolution 9 matches the pipeline's feature grid.
        q = f"""
            WITH bld AS (
              SELECT b.osm_id, b.building AS building_type, b.residential,
                     b.lon, b.lat,
                     h3_longlatash3(b.lon, b.lat, 9) AS h3
              FROM {NS}.bronze_buildings b
              WHERE b.aoi_name = ?{bbox_clause}
            )
            SELECT bld.osm_id, bld.building_type, bld.residential,
                   bld.lon, bld.lat,
                   CAST(NULL AS STRING) AS geometry_geojson,
                   bld.h3, p.flood_prob
            FROM bld
            JOIN {NS}.gold_h3_flood_predictions p
              ON p.aoi_name = ? AND p.h3 = bld.h3
            WHERE p.scenario_24h_mm = ? AND p.flood_prob >= ?
            ORDER BY p.flood_prob DESC
            LIMIT {int(limit)}
        """

    if use_real:
        params: tuple = (aoi, scenario, threshold, *bbox_params)
    else:
        params = (aoi, *bbox_params, aoi, scenario, threshold)

    rows = _fetch(q, params)
    total_el = 0.0
    buildings: list[BuildingExposure] = []
    for r in rows:
        prob = float(r["flood_prob"] or 0.0)
        is_res = bool(int(r.get("residential") or 0))
        brc = RESIDENTIAL_BRC_USD if is_res else COMMERCIAL_BRC_USD
        el = prob * brc * FLOOD_LOSS_SEVERITY
        total_el += el

        if r.get("geometry_geojson"):
            geom = json.loads(r["geometry_geojson"])
        else:
            geom = _synth_square(float(r["lon"]), float(r["lat"]))

        buildings.append(BuildingExposure(
            osm_id=str(r.get("osm_id") or ""),
            building_type=r.get("building_type") or None,
            residential=1 if is_res else 0,
            h3=_h3_hex(r["h3"]),
            flood_prob=prob,
            risk_tier=_risk_tier(prob),
            brc_usd=brc,
            loss_severity=FLOOD_LOSS_SEVERITY,
            expected_loss_usd=el,
            geometry=geom,
        ))

    return BuildingsAtRiskResponse(
        aoi_name=aoi,
        scenario_24h_mm=scenario,
        threshold=threshold,
        count=len(buildings),
        total_expected_loss_usd=total_el,
        footprint_source="real" if use_real else "synthesized",
        buildings=buildings,
    )


# Address lookup --------------------------------------------------------------

NOMINATIM_URL = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
NOMINATIM_USER_AGENT = os.environ.get(
    "NOMINATIM_USER_AGENT",
    "flood-prediction-demo/0.1 (github.com/databricks-industry-solutions/flood-prediction)",
)


@lru_cache(maxsize=512)
def _geocode(query: str, bbox: tuple[float, float, float, float] | None) -> dict[str, Any] | None:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 0,
    }
    if bbox is not None:
        # Nominatim `viewbox` is minlon,maxlat,maxlon,minlat (top-left + bottom-right)
        params["viewbox"] = f"{bbox[0]},{bbox[3]},{bbox[2]},{bbox[1]}"
        params["bounded"] = 1
    r = requests.get(
        NOMINATIM_URL, params=params,
        headers={"User-Agent": NOMINATIM_USER_AGENT, "Accept-Language": "en,fr"},
        timeout=15,
    )
    r.raise_for_status()
    hits = r.json() or []
    return hits[0] if hits else None


def _aoi_bbox(aoi: str) -> tuple[float, float, float, float] | None:
    rows = _fetch(
        f"SELECT min_lon, min_lat, max_lon, max_lat FROM {NS}.gold_aoi WHERE aoi_name = ?",
        (aoi,),
    )
    if not rows:
        return None
    r = rows[0]
    return (float(r["min_lon"]), float(r["min_lat"]),
            float(r["max_lon"]), float(r["max_lat"]))


@app.get("/api/lookup", response_model=AddressLookup)
def lookup(
    q: str = Query(..., min_length=3, description="Address or place name to search"),
    aoi: str = Query(default=DEFAULT_AOI),
    scenario_mm: int = Query(default=100, ge=0, le=500),
) -> AddressLookup:
    bbox = _aoi_bbox(aoi)
    hit = _geocode(q, bbox)
    if hit is None:
        # Retry once without the bbox constraint so out-of-AOI addresses still
        # resolve (the map won't show a prediction for them, but the caller
        # can display a "not in AOI" message).
        hit = _geocode(q, None)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"No geocoding result for '{q}'")

    lat, lon = float(hit["lat"]), float(hit["lon"])
    cell_str = h3.latlng_to_cell(lat, lon, 9)
    # h3-py v4 returns the canonical hex string already.
    cell_str = cell_str if isinstance(cell_str, str) else f"{int(cell_str):x}"
    cell_int = int(cell_str, 16)

    scenario = _nearest_scenario(aoi, scenario_mm)
    cur = _fetch(
        f"""
        SELECT flood_prob, min_elev, slope_deg, dist_to_water_m,
               annual_precip_mm, max24h_precip_mm,
               building_count, residential_count, expected_buildings_at_risk
        FROM {NS}.gold_h3_flood_predictions
        WHERE aoi_name = ? AND h3 = ? AND scenario_24h_mm = ?
        """,
        (aoi, cell_int, scenario),
    )
    cur_row = cur[0] if cur else {}

    sweep_rows = _fetch(
        f"""
        SELECT scenario_24h_mm, flood_prob
        FROM {NS}.gold_h3_flood_predictions
        WHERE aoi_name = ? AND h3 = ?
        ORDER BY scenario_24h_mm
        """,
        (aoi, cell_int),
    )
    sweep = [(int(r["scenario_24h_mm"]), float(r["flood_prob"])) for r in sweep_rows]

    return AddressLookup(
        query=q,
        resolved_name=str(hit.get("display_name") or q),
        lat=lat, lon=lon,
        h3=cell_str,
        scenario_24h_mm=scenario,
        flood_prob=float(cur_row.get("flood_prob") or 0.0),
        min_elev=float(cur_row["min_elev"]) if cur_row.get("min_elev") is not None else None,
        slope_deg=float(cur_row["slope_deg"]) if cur_row.get("slope_deg") is not None else None,
        dist_to_water_m=(
            float(cur_row["dist_to_water_m"])
            if cur_row.get("dist_to_water_m") is not None else None
        ),
        annual_precip_mm=(
            float(cur_row["annual_precip_mm"])
            if cur_row.get("annual_precip_mm") is not None else None
        ),
        max24h_precip_mm=(
            float(cur_row["max24h_precip_mm"])
            if cur_row.get("max24h_precip_mm") is not None else None
        ),
        building_count=(
            int(cur_row["building_count"])
            if cur_row.get("building_count") is not None else None
        ),
        residential_count=(
            int(cur_row["residential_count"])
            if cur_row.get("residential_count") is not None else None
        ),
        expected_buildings_at_risk=(
            float(cur_row["expected_buildings_at_risk"])
            if cur_row.get("expected_buildings_at_risk") is not None else None
        ),
        sweep=sweep,
    )


# Underwriter chat (Genie Conversation API) -----------------------------------


class ChatStartRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    aoi: str | None = None
    scenario_mm: int | None = None


class ChatMessageRequest(BaseModel):
    conversation_id: str
    content: str = Field(min_length=1, max_length=2000)
    aoi: str | None = None
    scenario_mm: int | None = None


class ChatMessage(BaseModel):
    conversation_id: str
    message_id: str
    status: str  # PENDING | COMPLETED | FAILED
    text: str | None = None
    sql: str | None = None
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    truncated: bool = False
    error: str | None = None


def _genie_headers() -> dict[str, str]:
    """Build an Authorization header for the Genie REST API using the same
    auth-source priority as `_connect()`. Genie sits on the workspace API
    surface so PAT / SP-OAuth / CLI-profile all work."""
    if TOKEN:
        return {"Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json"}
    if CLIENT_ID and CLIENT_SECRET:
        from databricks.sdk.core import Config, oauth_service_principal

        cfg = Config(host=f"https://{HOST}", client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        provider = oauth_service_principal(cfg)
        return {**provider(), "Content-Type": "application/json"}
    if PROFILE:
        from databricks.sdk.core import Config

        cfg = Config(profile=PROFILE)
        return {**cfg.authenticate(), "Content-Type": "application/json"}
    raise RuntimeError(
        "No Databricks auth available for Genie - set DATABRICKS_TOKEN, "
        "DATABRICKS_CLIENT_ID/SECRET, or DATABRICKS_CONFIG_PROFILE."
    )


def _genie_url(path: str) -> str:
    if not HOST:
        raise HTTPException(status_code=503, detail="DATABRICKS_HOST not configured")
    return f"https://{HOST}{path}"


def _require_genie() -> str:
    if not GENIE_SPACE_ID:
        raise HTTPException(
            status_code=503,
            detail="Underwriter chat is disabled: DATABRICKS_GENIE_SPACE_ID is not set. "
                   "Run scripts/genie_bootstrap.py and redeploy.",
        )
    return GENIE_SPACE_ID


def _genie_user_prompt(content: str, aoi: str | None, scenario_mm: int | None) -> str:
    """Prefix the user's question with the current map context. This is the
    single biggest accuracy lever for Genie: when the user types 'total EL
    at this scenario' we silently substitute the slider values so the
    generated SQL is concrete."""
    aoi_ctx = aoi or DEFAULT_AOI
    bits = [f"Current map context: AOI={aoi_ctx}"]
    if scenario_mm is not None:
        bits.append(f"scenario={int(scenario_mm)} mm/24h")
    return f"{'; '.join(bits)}.\n\nQuestion: {content}"


def _normalize_genie_message(conv_id: str, msg: dict[str, Any]) -> ChatMessage:
    """Coerce the (still-evolving) Genie REST message shape into our typed
    ChatMessage. Status is one of: PENDING (Genie still working), COMPLETED
    (assistant turn ready), FAILED."""
    # Normalize Genie's many intermediate states (IN_PROGRESS,
    # EXECUTING_QUERY, FETCHING_METADATA, FILTERING_CONTEXT, ASKING_AI,
    # PENDING_WAREHOUSE, SUBMITTED, ...) to a single PENDING so the frontend
    # only has to special-case the two terminal states. Allowlisting failed
    # in practice because Genie keeps adding new intermediate statuses; we
    # invert the logic - anything not explicitly terminal is "still working".
    raw_status = (msg.get("status") or "").upper()
    if raw_status == "COMPLETED":
        status = "COMPLETED"
    elif raw_status in {"FAILED", "QUERY_RESULT_EXPIRED", "CANCELLED"}:
        status = "FAILED"
    else:
        status = "PENDING"

    sql: str | None = None
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    truncated = False

    # Genie returns attachments[]: each can be a `text` (assistant message)
    # or `query` (generated SQL + result rows). We surface the first of each.
    for att in msg.get("attachments", []) or []:
        if "query" in att and sql is None:
            q = att["query"]
            sql = q.get("query")
            result = q.get("query_result") or q.get("result") or {}
            if isinstance(result, dict):
                schema = result.get("schema") or result.get("manifest", {}).get("schema") or {}
                cols = schema.get("columns") or schema.get("column_names") or []
                if cols and isinstance(cols[0], dict):
                    columns = [c.get("name") for c in cols if c.get("name")]
                else:
                    columns = list(cols) if cols else None
                data_rows = (result.get("data_array")
                             or result.get("rows")
                             or result.get("data", {}).get("data_array") if isinstance(result.get("data"), dict) else None)
                if isinstance(data_rows, list):
                    if len(data_rows) > GENIE_MAX_RESULT_ROWS:
                        truncated = True
                        data_rows = data_rows[:GENIE_MAX_RESULT_ROWS]
                    rows = data_rows
                    row_count = result.get("row_count") or len(data_rows)

    text: str | None = None
    for att in msg.get("attachments", []) or []:
        if "text" in att and isinstance(att["text"], dict):
            text = att["text"].get("content") or text

    error = None
    if status == "FAILED":
        error = msg.get("error", {}).get("error_message") if isinstance(msg.get("error"), dict) else str(msg.get("error") or "Genie failed")

    return ChatMessage(
        conversation_id=conv_id,
        message_id=msg.get("message_id") or msg.get("id") or "",
        status=status,
        text=text,
        sql=sql,
        columns=columns,
        rows=rows,
        row_count=row_count,
        truncated=truncated,
        error=error,
    )


@app.get("/api/chat/health")
def chat_health() -> dict[str, Any]:
    return {"enabled": bool(GENIE_SPACE_ID), "space_id": GENIE_SPACE_ID or None}


@app.post("/api/chat/start", response_model=ChatMessage)
def chat_start(req: ChatStartRequest) -> ChatMessage:
    space_id = _require_genie()
    body = {"content": _genie_user_prompt(req.content, req.aoi, req.scenario_mm)}
    r = requests.post(
        _genie_url(f"/api/2.0/genie/spaces/{space_id}/start-conversation"),
        headers=_genie_headers(),
        json=body,
        timeout=15,
    )
    if r.status_code >= 400:
        log.warning("Genie start-conversation %s: %s", r.status_code, r.text[:300])
        raise HTTPException(status_code=502, detail=f"Genie {r.status_code}: {r.text[:200]}")
    payload = r.json() or {}
    conv_id = payload.get("conversation_id") or payload.get("conversation", {}).get("id")
    msg = payload.get("message") or {}
    if not conv_id:
        raise HTTPException(status_code=502, detail="Genie did not return a conversation_id")
    return _normalize_genie_message(conv_id, msg)


@app.post("/api/chat/message", response_model=ChatMessage)
def chat_message(req: ChatMessageRequest) -> ChatMessage:
    space_id = _require_genie()
    body = {"content": _genie_user_prompt(req.content, req.aoi, req.scenario_mm)}
    r = requests.post(
        _genie_url(
            f"/api/2.0/genie/spaces/{space_id}/conversations/{req.conversation_id}/messages"
        ),
        headers=_genie_headers(),
        json=body,
        timeout=15,
    )
    if r.status_code >= 400:
        log.warning("Genie send-message %s: %s", r.status_code, r.text[:300])
        raise HTTPException(status_code=502, detail=f"Genie {r.status_code}: {r.text[:200]}")
    return _normalize_genie_message(req.conversation_id, r.json() or {})


@app.get("/api/chat/message/{conversation_id}/{message_id}", response_model=ChatMessage)
def chat_poll(conversation_id: str, message_id: str) -> ChatMessage:
    """Poll a Genie message until it reaches COMPLETED (or FAILED). The
    frontend long-polls this every ~1.2s while status == PENDING."""
    space_id = _require_genie()
    r = requests.get(
        _genie_url(
            f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}"
        ),
        headers=_genie_headers(),
        timeout=15,
    )
    if r.status_code >= 400:
        log.warning("Genie poll %s: %s", r.status_code, r.text[:300])
        raise HTTPException(status_code=502, detail=f"Genie {r.status_code}: {r.text[:200]}")
    msg = _normalize_genie_message(conversation_id, r.json() or {})

    # If Genie returned a query attachment but no rows (newer API splits result
    # fetch onto its own sub-resource), fetch it explicitly.
    if msg.status == "COMPLETED" and msg.sql is not None and msg.rows is None:
        raw = r.json() or {}
        for att in raw.get("attachments", []) or []:
            att_id = att.get("attachment_id") or att.get("id")
            if "query" in att and att_id:
                qr = requests.get(
                    _genie_url(
                        f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}"
                        f"/messages/{message_id}/attachments/{att_id}/query-result"
                    ),
                    headers=_genie_headers(),
                    timeout=30,
                )
                if qr.status_code < 400:
                    stmt = qr.json().get("statement_response", {}).get("result", {})
                    cols = (qr.json().get("statement_response", {})
                            .get("manifest", {}).get("schema", {}).get("columns", []))
                    msg.columns = [c.get("name") for c in cols] or msg.columns
                    data = stmt.get("data_array") or []
                    if len(data) > GENIE_MAX_RESULT_ROWS:
                        msg.truncated = True
                        data = data[:GENIE_MAX_RESULT_ROWS]
                    msg.rows = data
                    msg.row_count = stmt.get("row_count") or len(data)
                    break
    return msg


# Static SPA -------------------------------------------------------------------

CLIENT_DIR = Path(__file__).parent / "client" / "dist"
if CLIENT_DIR.exists():
    app.mount("/assets", StaticFiles(directory=CLIENT_DIR / "assets"), name="assets")

    @app.get("/")
    @app.get("/{_path:path}")
    def spa(_path: str = ""):
        index = CLIENT_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="SPA build missing")
        return FileResponse(index)
else:
    @app.get("/")
    def no_spa():
        return {"message": "SPA not built yet. Run `bun install && bun run build` in src/app/client/."}
