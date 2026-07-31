# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingestion (parameterized by AOI)
# MAGIC
# MAGIC Downloads the three geospatial inputs for the flood demo into a Unity Catalog
# MAGIC Volume, then registers them as **bronze Delta tables** via the GeoBrix readers:
# MAGIC
# MAGIC 1. **Elevation** - NASA SRTM 1-arc-second DEM (~30 m, global, free, no auth).
# MAGIC    Canada's HRDEM is higher resolution but requires per-tile FTP lookups; SRTM
# MAGIC    is good enough for a demo and trivial to grab from AWS Open Data.
# MAGIC 2. **Hydrography** - OpenStreetMap water polygons and rivers, pulled from the
# MAGIC    Overpass API for the AOI bbox and written as GeoJSON.
# MAGIC 3. **Historical flood polygons** - Quebec open-data "Limites des zones inondées"
# MAGIC    dataset for the 2017 and 2019 Saint-Laurent / Ottawa River floods
# MAGIC    (`donneesquebec.ca`).
# MAGIC
# MAGIC All downloads are cached on the Volume so re-runs are fast. The AOI bbox comes
# MAGIC from job parameters so the same pipeline retargets any city.

# COMMAND ----------

dbutils.widgets.text("catalog", "flood_demo")
dbutils.widgets.text("schema", "montreal")
dbutils.widgets.text("aoi_name", "greater_montreal")
dbutils.widgets.text("aoi_bbox_min_lon", "-74.05")
dbutils.widgets.text("aoi_bbox_min_lat", "45.30")
dbutils.widgets.text("aoi_bbox_max_lon", "-73.30")
dbutils.widgets.text("aoi_bbox_max_lat", "45.80")
# Workspace-file raw storage root. Default to the current user's home so the
# notebook works both inside the DAB job and when run interactively.
dbutils.widgets.text("raw_root", "")

# Flood-polygon source dispatch. See databricks.yml for the full descriptions.
dbutils.widgets.dropdown("flood_source", "melcc", ["melcc", "cems", "none"])
dbutils.widgets.text("cems_urls", "")
dbutils.widgets.text("cems_year_default", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
aoi_name = dbutils.widgets.get("aoi_name")
min_lon = float(dbutils.widgets.get("aoi_bbox_min_lon"))
min_lat = float(dbutils.widgets.get("aoi_bbox_min_lat"))
max_lon = float(dbutils.widgets.get("aoi_bbox_max_lon"))
max_lat = float(dbutils.widgets.get("aoi_bbox_max_lat"))
raw_root_param = dbutils.widgets.get("raw_root")
flood_source = (dbutils.widgets.get("flood_source") or "melcc").strip().lower()
cems_urls_raw = dbutils.widgets.get("cems_urls") or ""
cems_year_default = (dbutils.widgets.get("cems_year_default") or "").strip()

import os

if raw_root_param:
    raw_root = raw_root_param.rstrip("/")
else:
    user = (spark.sql("SELECT current_user()").collect()[0][0])
    raw_root = f"/Workspace/Users/{user}/flood-demo/raw"
volume_root = f"{raw_root}/{aoi_name}"
os.makedirs(volume_root, exist_ok=True)

# Multi-AOI co-residence: every bronze table is partitioned by aoi_name, and a
# `mode("overwrite")` write must only replace the current AOI's partition
# instead of wiping the entire table. Without that scoping, running the pipeline
# for AOI #2 would erase AOI #1's bronze rows and the App's AOI dropdown would
# shrink back to a single entry.
#
# Serverless rejects spark.sql.sources.partitionOverwriteMode=dynamic
# (CONFIG_NOT_AVAILABLE), so we scope the overwrite with Delta's replaceWhere,
# which behaves the same on every compute type.


def write_aoi_partition(df, table, extra_partitions=(), options=None):
    """Overwrite only the active AOI's partition of `table`.

    `replaceWhere` is skipped on the very first write because the table does
    not exist yet and there is nothing to replace.
    """
    partitions = ("aoi_name",) + tuple(extra_partitions)
    writer = df.write.format("delta").mode("overwrite").partitionBy(*partitions)
    for key, value in (options or {}).items():
        writer = writer.option(key, value)
    if spark.catalog.tableExists(table):
        writer = writer.option("replaceWhere", f"aoi_name = '{aoi_name}'")
    writer.saveAsTable(table)

print(f"AOI {aoi_name}: ({min_lon},{min_lat}) -> ({max_lon},{max_lat})")
print("Raw data root:", volume_root)

# Note: catalog + schema are provisioned by the DABs bundle. We do NOT issue
# `CREATE CATALOG IF NOT EXISTS` here because some metastores with broken
# default-storage bindings reject even idempotent DDL. Assume the target
# schema already exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. DEM (SRTM 1-arc-second tiles)
# MAGIC
# MAGIC We only pull the raw `.hgt` tiles here (one 1° x 1° tile per AOI grid cell)
# MAGIC from AWS Open Data and write them to workspace files with their canonical
# MAGIC names (e.g. `N45W074.hgt`). The GeoBrix GDAL reader in notebook 02 is what
# MAGIC actually reads them, merges them, and clips them to the AOI - all in
# MAGIC Spark, no rasterio/GDAL-from-Python involved at this step.

# COMMAND ----------

import math
import os
import gzip
from pathlib import Path
from urllib.request import urlopen, Request


def _aoi_srtm_tiles(min_lon, min_lat, max_lon, max_lat):
    tiles = []
    for lat in range(int(math.floor(min_lat)), int(math.ceil(max_lat))):
        for lon in range(int(math.floor(min_lon)), int(math.ceil(max_lon))):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return tiles


def _download_srtm_hgt(tile_id: str) -> bytes:
    lat_band = tile_id[:3]
    url = f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{lat_band}/{tile_id}.hgt.gz"
    req = Request(url, headers={"User-Agent": "flood-demo/1.0"})
    with urlopen(req, timeout=120) as r:
        return r.read()


dem_dir = f"{volume_root}/dem"
os.makedirs(dem_dir, exist_ok=True)

tile_ids = _aoi_srtm_tiles(min_lon, min_lat, max_lon, max_lat)
print(f"SRTM tiles for AOI: {tile_ids}")

dem_tile_paths = []
for tid in tile_ids:
    out_path = f"{dem_dir}/{tid}.hgt"
    if not os.path.exists(out_path):
        print(f"  downloading {tid} -> {out_path}")
        raw = gzip.decompress(_download_srtm_hgt(tid))
        with open(out_path, "wb") as f:
            f.write(raw)
    dem_tile_paths.append(out_path)

# `dem_path` is the tile **directory**. GeoBrix's GDAL reader resolves it via
# Hadoop `listFiles` and yields one raster row per file (SRTMHGT driver picks
# up lat/lon from the filename). The directory contains nothing but `.hgt`
# tiles, so no extra filtering is needed downstream.
dem_path = dem_dir
print(f"DEM dir: {dem_path} ({len(dem_tile_paths)} tiles)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Hydrography (OpenStreetMap water via Overpass)

# COMMAND ----------

import json
import time
from urllib.parse import quote


def fetch_overpass(query: str) -> dict:
    url = "https://overpass-api.de/api/interpreter"
    req = Request(url, data=f"data={quote(query)}".encode(), method="POST",
                  headers={"User-Agent": "flood-demo/1.0"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"Overpass attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    raise RuntimeError("Overpass API failed after 3 retries")


hydro_path = f"{volume_root}/hydro/{aoi_name}_water.geojson"
if not os.path.exists(hydro_path):
    os.makedirs(os.path.dirname(hydro_path), exist_ok=True)
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    # Water polygons + major rivers (waterway=river) as lines buffered client side.
    query = f"""
    [out:json][timeout:180];
    (
      way["natural"="water"]({bbox});
      relation["natural"="water"]({bbox});
      way["waterway"="river"]({bbox});
      way["waterway"="stream"]({bbox});
    );
    out geom;
    """
    raw = fetch_overpass(query)
    # Convert Overpass elements -> GeoJSON FeatureCollection.
    features = []
    for el in raw.get("elements", []):
        geom = el.get("geometry")
        if not geom:
            continue
        coords = [[p["lon"], p["lat"]] for p in geom]
        if el["type"] == "way":
            tags = el.get("tags", {})
            if tags.get("natural") == "water" and coords and coords[0] == coords[-1]:
                shape = {"type": "Polygon", "coordinates": [coords]}
            else:
                shape = {"type": "LineString", "coordinates": coords}
            features.append({
                "type": "Feature",
                "properties": {"osm_id": el["id"], **tags},
                "geometry": shape,
            })
    fc = {"type": "FeatureCollection", "features": features}
    with open(hydro_path, "w") as f:
        json.dump(fc, f)
    print(f"Wrote {len(features)} hydro features -> {hydro_path}")
else:
    print("Hydro already cached ->", hydro_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. OSM buildings (for exposure computation)
# MAGIC
# MAGIC We pull every `building=*` footprint in the AOI via Overpass and store the
# MAGIC centroid + residential flag. Silver tessellates these to H3 to give each
# MAGIC cell a `building_count`, which downstream lets us compute
# MAGIC `expected_buildings_at_risk = flood_prob * building_count` per prediction.

# COMMAND ----------

buildings_path = f"{volume_root}/buildings/{aoi_name}_buildings.geojson"
if not os.path.exists(buildings_path):
    os.makedirs(os.path.dirname(buildings_path), exist_ok=True)
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    # `out center tags` gives us the centroid + tag dict without shipping every
    # polygon vertex - at Montreal AOI scale this is ~300k rows vs ~30 MB, fine.
    query = f"""
    [out:json][timeout:180];
    (
      way["building"]({bbox});
      relation["building"]({bbox});
    );
    out center tags;
    """
    raw = fetch_overpass(query)

    RESIDENTIAL_VALUES = {
        "yes", "residential", "house", "apartments", "detached",
        "semidetached_house", "terrace", "bungalow", "dormitory",
    }
    features = []
    for el in raw.get("elements", []):
        tags = el.get("tags", {}) or {}
        btype = tags.get("building", "yes")
        c = el.get("center") or {}
        lon, lat = c.get("lon"), c.get("lat")
        if lon is None or lat is None:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "osm_id": f"{el['type']}/{el['id']}",
                "building": btype,
                "residential": 1 if btype in RESIDENTIAL_VALUES else 0,
            },
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
    fc = {"type": "FeatureCollection", "features": features}
    with open(buildings_path, "w") as f:
        json.dump(fc, f)
    print(f"Wrote {len(features)} building centroids -> {buildings_path}")
else:
    print("Buildings already cached ->", buildings_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Historical flood polygons (pluggable source)
# MAGIC
# MAGIC Three sources, dispatched on the `flood_source` widget:
# MAGIC
# MAGIC * **`melcc`** (default) — Quebec MDDELCC ArcGIS Feature Service backing
# MAGIC   the Données Québec "Territoire inondé en 2017 et 2019" dataset.
# MAGIC   Quebec-only; raises if 0 features intersect the AOI bbox.
# MAGIC * **`cems`** — Copernicus Emergency Management Service Rapid Mapping.
# MAGIC   Global. Provide a comma-separated list of GeoJSON URLs in
# MAGIC   `cems_urls` (browse [emergency.copernicus.eu/mapping](https://emergency.copernicus.eu/mapping/list-of-activations-rapid)
# MAGIC   for the activation you want and grab the GeoJSON download links).
# MAGIC   Also auto-picks up any `*.geojson` or `*.json` files dropped manually
# MAGIC   into `<volume_root>/floods/cems_input/`.
# MAGIC * **`none`** — write an empty `bronze_flood_events`. The App's validation
# MAGIC   overlay disappears; the model still trains on synthetic labels.
# MAGIC
# MAGIC Whichever source is used, the output is normalized to one row per
# MAGIC polygon with `(aoi_name, year, geom_geojson)` so downstream notebooks
# MAGIC don't care where the data came from.

# COMMAND ----------

from datetime import datetime as _dt
from urllib.parse import urlencode, urlparse
import re
import time

# Cache file is per-source so swapping `flood_source` doesn't cross-contaminate.
flood_path = f"{volume_root}/floods/{aoi_name}_historical_{flood_source}.geojson"

# Allow the user to bust the cache without manually deleting the volume file.
dbutils.widgets.text("force_refresh_floods", "false")
_force_refresh = dbutils.widgets.get("force_refresh_floods").strip().lower() in (
    "1", "true", "yes")


def _cache_is_usable(path: str, allow_empty: bool = False) -> bool:
    """A cache parses as a FeatureCollection. Empty is OK iff allow_empty=True."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            fc = json.load(f)
    except Exception:
        return False
    if not (isinstance(fc, dict) and fc.get("type") == "FeatureCollection"):
        return False
    return allow_empty or bool(fc.get("features"))


# ─────────────────────────────────────────────────────────────────────────────
# MELCC source — Quebec MDDELCC ArcGIS Feature Service
# ─────────────────────────────────────────────────────────────────────────────
FLOOD_MAPSERVER = (
    "https://www.servicesgeo.enviroweb.gouv.qc.ca/donnees/rest/services/"
    "Public/Territoire_inonde_en_2017_et_2019/MapServer/1/query"
)
FLOOD_MAX_RECORDS = 1000  # = MapServer maxRecordCount, do not raise


def _fetch_melcc_features(min_lon, min_lat, max_lon, max_lat,
                          page_size=FLOOD_MAX_RECORDS,
                          max_attempts=4, base_backoff_s=2.0):
    """Page through the MDDELCC ArcGIS Feature Service for the AOI bbox.

    Each feature gets a synthetic `year` derived from the `Date_debut` decree
    timestamp — this is the decree's effective date, not the flood occurrence
    date, so for the current combined 2017+2019 ZIS layer every feature ends
    up tagged `'2019'`. Raises on transport / HTTP failure after retries; the
    caller decides whether 0 features is fatal.
    """
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": "Date_debut,Nm_decret,No_decret",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": "4326",
            "resultRecordCount": page_size,
            "resultOffset": offset,
        }
        url = f"{FLOOD_MAPSERVER}?{urlencode(params)}"
        last_err: Exception | None = None
        payload = None
        for attempt in range(1, max_attempts + 1):
            try:
                req = Request(url, headers={"User-Agent": "flood-demo/1.0"})
                with urlopen(req, timeout=180) as r:
                    payload = json.loads(r.read())
                break
            except Exception as e:
                last_err = e
                if attempt < max_attempts:
                    sleep_s = base_backoff_s * (2 ** (attempt - 1))
                    print(f"  flood fetch offset={offset} attempt {attempt} "
                          f"failed ({type(e).__name__}: {e}); retrying in {sleep_s:.1f}s")
                    time.sleep(sleep_s)
        if payload is None:
            raise RuntimeError(
                f"MDDELCC ArcGIS fetch failed after {max_attempts} attempts at "
                f"offset={offset}: {last_err!r}"
            ) from last_err

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(
                f"ArcGIS service returned error at offset={offset}: {payload['error']}"
            )

        batch = payload.get("features", [])
        for feat in batch:
            props = feat.get("properties") or {}
            ts = props.get("Date_debut")
            if ts:
                try:
                    props["year"] = _dt.utcfromtimestamp(ts / 1000).strftime("%Y")
                except Exception:
                    props["year"] = ""
            else:
                props["year"] = ""
            feat["properties"] = props
            features.append(feat)

        if not payload.get("exceededTransferLimit") and len(batch) < page_size:
            break
        if not batch:
            break
        offset += page_size
    return features


# ─────────────────────────────────────────────────────────────────────────────
# CEMS source — Copernicus Emergency Management Service Rapid Mapping
# ─────────────────────────────────────────────────────────────────────────────
# We accept GeoJSON / JSON URLs (most CEMS Rapid Mapping products are offered
# as GeoJSON in addition to Shapefile). Shapefile-only zips would require a
# heavy GDAL/fiona dependency and are intentionally not supported — pick the
# GeoJSON download from the CEMS portal instead.
_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")


def _year_from_string(s: str) -> str:
    if not s:
        return ""
    m = _YEAR_RE.search(s)
    return m.group(1) if m else ""


def _year_from_feature_props(props: dict) -> str:
    """Best-effort year extraction from CEMS feature properties.

    CEMS schemas vary by activation/provider. Common candidates we've seen:
    `event_date`, `obs_date`, `acq_date`, `src_date`, `event_year`. Fall
    through them in order; first parseable year wins.
    """
    if not isinstance(props, dict):
        return ""
    for key in ("event_year", "year", "event_date", "obs_date", "acq_date",
                "src_date", "src_acq_da", "Date_debut"):
        v = props.get(key)
        if v in (None, ""):
            continue
        # Numeric epoch (ms) — same convention as MELCC ArcGIS.
        if isinstance(v, (int, float)) and v > 10**11:
            try:
                return _dt.utcfromtimestamp(v / 1000).strftime("%Y")
            except Exception:
                pass
        s = str(v)
        y = _year_from_string(s)
        if y:
            return y
    return ""


def _bbox_hits(geom: dict, mn_lon, mn_lat, mx_lon, mx_lat) -> bool:
    """Cheap bbox-vs-bbox prune. CEMS products span large AoIs that may extend
    well beyond our AOI; clip in-Python to keep bronze focused."""
    if not geom:
        return False
    coords = geom.get("coordinates")
    if coords is None:
        return False

    def _walk(c):
        if isinstance(c, (int, float)):
            return
        if c and isinstance(c[0], (int, float)):
            yield c[0], c[1]
            return
        for sub in c:
            yield from _walk(sub)

    pts = list(_walk(coords))
    if not pts:
        return False
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return (max(lons) >= mn_lon and min(lons) <= mx_lon
            and max(lats) >= mn_lat and min(lats) <= mx_lat)


def _read_geojson_bytes(raw: bytes, source_label: str) -> list[dict]:
    """Parse a GeoJSON byte blob into a list of features.

    Accepts both top-level FeatureCollection and a bare list of features.
    Returns [] (with a warning) on parse failure rather than raising — so a
    single bad URL doesn't sink the whole run.
    """
    try:
        doc = json.loads(raw)
    except Exception as e:
        print(f"  WARN: {source_label} is not valid JSON ({e}); skipping.")
        return []
    if isinstance(doc, dict) and doc.get("type") == "FeatureCollection":
        return doc.get("features") or []
    if isinstance(doc, list):
        return [f for f in doc if isinstance(f, dict) and f.get("type") == "Feature"]
    if isinstance(doc, dict) and doc.get("type") == "Feature":
        return [doc]
    print(f"  WARN: {source_label} is JSON but not a FeatureCollection / Feature; "
          f"skipping (top-level type: {doc.get('type') if isinstance(doc, dict) else type(doc).__name__}).")
    return []


def _download(url: str, max_attempts: int = 3) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "flood-demo/1.0"})
            with urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed for {url}: {last_err!r}") from last_err


def _fetch_cems_features(urls: list[str], local_dir: str,
                         year_default: str,
                         mn_lon, mn_lat, mx_lon, mx_lat) -> list[dict]:
    """Aggregate CEMS Rapid Mapping features from URLs and a local drop folder.

    For each input file, we extract a year from (in priority order) the
    feature properties, the source filename, the configured `year_default`,
    or `""`. Geometries are bbox-clipped to the AOI to drop AoI overshoot.
    """
    sources: list[tuple[str, bytes]] = []  # (label, raw bytes)

    # Remote URLs
    for u in urls:
        u = u.strip()
        if not u:
            continue
        path = urlparse(u).path.lower()
        if not (path.endswith(".geojson") or path.endswith(".json")):
            print(f"  WARN: {u} doesn't look like GeoJSON (extension '{path.rsplit('.',1)[-1]}'); "
                  f"attempting to parse anyway. For shapefile zips, download "
                  f"the GeoJSON product from the CEMS portal instead.")
        print(f"  fetching {u}")
        try:
            raw = _download(u)
            sources.append((os.path.basename(path) or u, raw))
        except Exception as e:
            print(f"  WARN: failed to download {u}: {e}; skipping.")

    # Local drop folder
    if os.path.isdir(local_dir):
        for fn in sorted(os.listdir(local_dir)):
            if not (fn.lower().endswith(".geojson") or fn.lower().endswith(".json")):
                continue
            fp = os.path.join(local_dir, fn)
            print(f"  reading local {fp}")
            try:
                with open(fp, "rb") as f:
                    sources.append((fn, f.read()))
            except Exception as e:
                print(f"  WARN: failed to read {fp}: {e}; skipping.")

    if not sources:
        return []

    out: list[dict] = []
    kept = dropped_oob = 0
    for label, raw in sources:
        feats = _read_geojson_bytes(raw, label)
        file_year = _year_from_string(label)
        for feat in feats:
            geom = feat.get("geometry") or {}
            if not _bbox_hits(geom, mn_lon, mn_lat, mx_lon, mx_lat):
                dropped_oob += 1
                continue
            props = feat.get("properties") or {}
            year = (_year_from_feature_props(props) or file_year
                    or year_default or "")
            props["year"] = year
            props.setdefault("cems_source", label)
            feat["properties"] = props
            out.append(feat)
            kept += 1
    print(f"  CEMS: kept {kept} features, dropped {dropped_oob} outside AOI bbox "
          f"(across {len(sources)} sources)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
print(f"flood_source = {flood_source!r}")

if _force_refresh or not _cache_is_usable(flood_path, allow_empty=(flood_source == "none")):
    if _force_refresh and os.path.exists(flood_path):
        print("force_refresh_floods=true; ignoring existing cache at", flood_path)
    elif os.path.exists(flood_path):
        print("Existing flood cache is empty/malformed, re-fetching:", flood_path)
    os.makedirs(os.path.dirname(flood_path), exist_ok=True)

    if flood_source == "melcc":
        feats = _fetch_melcc_features(min_lon, min_lat, max_lon, max_lat)
        if not feats:
            raise RuntimeError(
                "MDDELCC ArcGIS returned 0 flood polygons for AOI "
                f"({min_lon},{min_lat})->({max_lon},{max_lat}). Verify the bbox "
                "intersects the 2017/2019 ZIS extents, or switch flood_source "
                "to 'cems' (global) or 'none' (skip). Refusing to write an empty cache."
            )
        years = sorted({f["properties"].get("year") or "?" for f in feats})
        print(f"Pulled {len(feats)} MELCC flood features (years: {years}; "
              "note: derived from decree Date_debut, not flood date)")

    elif flood_source == "cems":
        cems_local = f"{volume_root}/floods/cems_input"
        urls = [u for u in cems_urls_raw.split(",") if u.strip()]
        if not urls and not os.path.isdir(cems_local):
            raise RuntimeError(
                "flood_source='cems' but neither cems_urls nor a local drop "
                f"directory exists. Either set --var=cems_urls=https://... or "
                f"upload GeoJSON files to {cems_local}/. Browse "
                "https://emergency.copernicus.eu/mapping/list-of-activations-rapid "
                "for activation downloads."
            )
        feats = _fetch_cems_features(
            urls, cems_local, cems_year_default,
            min_lon, min_lat, max_lon, max_lat,
        )
        if not feats:
            raise RuntimeError(
                "flood_source='cems' produced 0 features after AOI clipping. "
                "Verify the activation AoI overlaps your bbox and that the URLs "
                "point to GeoJSON (not Shapefile) products."
            )
        years = sorted({f["properties"].get("year") or "?" for f in feats})
        print(f"Loaded {len(feats)} CEMS flood features (years: {years})")

    elif flood_source == "none":
        feats = []
        print("flood_source='none' — writing an empty FeatureCollection. "
              "The App's validation overlay will be hidden.")

    else:
        raise ValueError(
            f"Unknown flood_source={flood_source!r}. Use 'melcc', 'cems', or 'none'."
        )

    fc = {"type": "FeatureCollection", "features": feats}
    with open(flood_path, "w") as f:
        json.dump(fc, f)
else:
    print("Flood polygons already cached ->", flood_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register bronze Delta tables
# MAGIC
# MAGIC We store raw file paths + geometry/WKB so downstream notebooks can read them
# MAGIC with GeoBrix / Databricks Spatial SQL.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

bronze_ns = f"{catalog}.{schema}"

# Raw DEM -> bronze manifest (we don't expand tiles here; GeoBrix does that in silver)
dem_df = spark.createDataFrame(
    [(aoi_name, dem_path, min_lon, min_lat, max_lon, max_lat)],
    schema=StructType([
        StructField("aoi_name", StringType(), False),
        StructField("dem_path", StringType(), False),
        StructField("min_lon", DoubleType(), False),
        StructField("min_lat", DoubleType(), False),
        StructField("max_lon", DoubleType(), False),
        StructField("max_lat", DoubleType(), False),
    ]),
)
write_aoi_partition(dem_df.withColumn("ingested_at", F.current_timestamp()),
                    f"{bronze_ns}.bronze_dem_manifest")

# Hydrography -> bronze (one row per feature, geometry as GeoJSON string)
with open(hydro_path) as f:
    hydro_fc = json.load(f)
hydro_rows = [
    (aoi_name, feat["properties"].get("osm_id"),
     feat["properties"].get("natural"),
     feat["properties"].get("waterway"),
     json.dumps(feat["geometry"]))
    for feat in hydro_fc["features"]
]
hydro_df = spark.createDataFrame(
    hydro_rows,
    schema=StructType([
        StructField("aoi_name", StringType(), False),
        StructField("osm_id", StringType(), True),
        StructField("natural", StringType(), True),
        StructField("waterway", StringType(), True),
        StructField("geom_geojson", StringType(), False),
    ]),
)
write_aoi_partition(hydro_df.withColumn("ingested_at", F.current_timestamp()),
                    f"{bronze_ns}.bronze_hydrography")

# Historical flood polygons -> bronze
# `year` is synthesized in `_fetch_flood_features` from the decree start date
# and is the only year-like field present in the MDDELCC layer; older
# Données Québec exports of "Limites des zones inondées" used `annee`, kept as
# a fallback so the cache remains forward/backward-compatible if the upstream
# source is swapped.
with open(flood_path) as f:
    flood_fc = json.load(f)


def _coords_to_2d(coords):
    # MDDELCC ArcGIS emits XYZM positions like [lon, lat, 0, null]; the null M
    # value breaks ST_GeomFromGeoJSON. Trim every position to [lon, lat].
    if (coords and isinstance(coords, list)
            and coords and not isinstance(coords[0], list)):
        return coords[:2]
    return [_coords_to_2d(c) for c in coords]


def _geom_to_2d(geom):
    if not geom or geom.get("type") == "GeometryCollection":
        return geom
    return {**geom, "coordinates": _coords_to_2d(geom["coordinates"])}


flood_rows = [
    (aoi_name, str(feat.get("properties", {}).get("year")
                   or feat.get("properties", {}).get("annee") or ""),
     json.dumps(_geom_to_2d(feat["geometry"])))
    for feat in flood_fc["features"]
]
flood_df = spark.createDataFrame(
    flood_rows,
    schema=StructType([
        StructField("aoi_name", StringType(), False),
        StructField("year", StringType(), True),
        StructField("geom_geojson", StringType(), False),
    ]),
) if flood_rows else spark.createDataFrame(
    [], schema=StructType([
        StructField("aoi_name", StringType(), False),
        StructField("year", StringType(), True),
        StructField("geom_geojson", StringType(), False),
    ]))

write_aoi_partition(flood_df.withColumn("ingested_at", F.current_timestamp()),
                    f"{bronze_ns}.bronze_flood_events")

# Buildings -> bronze (point centroid + residential flag + OSM-derived tags)
with open(buildings_path) as f:
    bld_fc = json.load(f)
bld_rows = [
    (aoi_name,
     feat["properties"].get("osm_id", ""),
     feat["properties"].get("building", ""),
     int(feat["properties"].get("residential", 0) or 0),
     float(feat["geometry"]["coordinates"][0]),
     float(feat["geometry"]["coordinates"][1]))
    for feat in bld_fc["features"]
]
from pyspark.sql.types import IntegerType
bld_schema = StructType([
    StructField("aoi_name",     StringType(),  False),
    StructField("osm_id",       StringType(),  True),
    StructField("building",     StringType(),  True),
    StructField("residential",  IntegerType(), False),
    StructField("lon",          DoubleType(),  False),
    StructField("lat",          DoubleType(),  False),
])
bld_df = spark.createDataFrame(bld_rows, schema=bld_schema) \
    if bld_rows else spark.createDataFrame([], schema=bld_schema)
write_aoi_partition(bld_df.withColumn("ingested_at", F.current_timestamp()),
                    f"{bronze_ns}.bronze_buildings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Precipitation climatology (Open-Meteo ERA5 archive)
# MAGIC
# MAGIC We grid-sample the AOI at 0.1 degree spacing and pull daily precipitation
# MAGIC totals from Open-Meteo's free [ERA5 historical archive](https://open-meteo.com/en/docs/historical-weather-api)
# MAGIC (no auth required). For each grid point we compute:
# MAGIC
# MAGIC * `annual_precip_mm` - mean annual precipitation
# MAGIC * `max24h_precip_mm` - 99th percentile of daily precipitation
# MAGIC * `max5d_precip_mm`  - 99th percentile of 5-day rolling precip
# MAGIC
# MAGIC These feed the silver step which nearest-neighbors them onto H3 cells.

# COMMAND ----------

from datetime import date, timedelta
import numpy as np

# Keep the pull short to stay under Open-Meteo's free-tier rate limits
# (demo-grade climatology - roughly 4 years is plenty to characterize P99 events).
PRECIP_END = (date.today() - timedelta(days=7)).isoformat()
PRECIP_START = (date.today() - timedelta(days=365 * 4)).isoformat()
PRECIP_GRID_STEP = 0.2  # degrees - coarser grid to reduce API call count

precip_cache = f"{volume_root}/precip/{aoi_name}_climatology.json"


def _precip_points():
    pts = []
    lat = min_lat
    while lat <= max_lat + 1e-9:
        lon = min_lon
        while lon <= max_lon + 1e-9:
            pts.append((round(lon, 4), round(lat, 4)))
            lon += PRECIP_GRID_STEP
        lat += PRECIP_GRID_STEP
    return pts


def _fetch_open_meteo_daily(lat: float, lon: float) -> list[float]:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={PRECIP_START}&end_date={PRECIP_END}"
        "&daily=precipitation_sum&timezone=UTC"
    )
    req = Request(url, headers={"User-Agent": "flood-demo/1.0"})
    # Short timeout with retry so we don't hang the whole notebook if one point
    # is rate-limited. 3 retries with exponential backoff.
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=30) as r:
                raw = json.loads(r.read())
            return [0.0 if v is None else float(v)
                    for v in raw.get("daily", {}).get("precipitation_sum", [])]
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo failed for ({lat},{lon}): {last_err}")


def _climatology(daily_mm: list[float]) -> dict:
    a = np.asarray(daily_mm, dtype=float)
    if a.size == 0:
        return {"annual_precip_mm": 0.0, "max24h_precip_mm": 0.0, "max5d_precip_mm": 0.0}
    years = a.size / 365.25
    roll5 = np.convolve(a, np.ones(5), mode="valid")
    return {
        "annual_precip_mm": float(a.sum() / max(years, 1e-6)),
        "max24h_precip_mm": float(np.percentile(a, 99)),
        "max5d_precip_mm":  float(np.percentile(roll5, 99)) if roll5.size else 0.0,
    }


if not os.path.exists(precip_cache):
    os.makedirs(os.path.dirname(precip_cache), exist_ok=True)
    points = _precip_points()
    n = len(points)
    print(f"Pulling precip climatology for {n} grid points {PRECIP_START}..{PRECIP_END}")
    records = []
    for i, (lon, lat) in enumerate(points):
        try:
            daily = _fetch_open_meteo_daily(lat, lon)
            clim = _climatology(daily)
            records.append({"lon": lon, "lat": lat, **clim})
            if (i + 1) % 5 == 0 or i == n - 1:
                print(f"  [{i+1}/{n}] ({lon:.2f},{lat:.2f}) "
                      f"annual={clim['annual_precip_mm']:.0f}mm "
                      f"p99_24h={clim['max24h_precip_mm']:.0f}mm")
        except Exception as e:
            print(f"  point {i} ({lon},{lat}) FAILED: {e}; using zeros")
            records.append({"lon": lon, "lat": lat,
                            "annual_precip_mm": 0.0, "max24h_precip_mm": 0.0, "max5d_precip_mm": 0.0})
        time.sleep(0.3)
    with open(precip_cache, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} precip points -> {precip_cache}")
else:
    print("Precip climatology already cached ->", precip_cache)

# COMMAND ----------

with open(precip_cache) as f:
    precip_records = json.load(f)

precip_rows = [
    (aoi_name, float(r["lon"]), float(r["lat"]),
     float(r["annual_precip_mm"]), float(r["max24h_precip_mm"]), float(r["max5d_precip_mm"]))
    for r in precip_records
]
precip_df = spark.createDataFrame(
    precip_rows,
    schema=StructType([
        StructField("aoi_name", StringType(), False),
        StructField("lon", DoubleType(), False),
        StructField("lat", DoubleType(), False),
        StructField("annual_precip_mm", DoubleType(), False),
        StructField("max24h_precip_mm", DoubleType(), False),
        StructField("max5d_precip_mm", DoubleType(), False),
    ]),
)
write_aoi_partition(precip_df.withColumn("ingested_at", F.current_timestamp()),
                    f"{bronze_ns}.bronze_precip_grid")

print("Bronze tables written:")
for t in ("bronze_dem_manifest", "bronze_hydrography", "bronze_flood_events",
          "bronze_precip_grid", "bronze_buildings"):
    n = spark.table(f"{bronze_ns}.{t}").count()
    print(f"  {bronze_ns}.{t}: {n} rows")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "aoi_name": aoi_name,
    "dem_path": dem_path,
    "hydro_path": hydro_path,
    "flood_path": flood_path,
    "precip_cache": precip_cache,
    "buildings_path": buildings_path,
}))
