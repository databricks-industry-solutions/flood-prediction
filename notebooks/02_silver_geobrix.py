# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver with GeoBrix + Databricks Spatial SQL
# MAGIC
# MAGIC Reads the raw DEM and hydrography, then produces three silver tables keyed
# MAGIC on H3 cell id:
# MAGIC
# MAGIC | table                    | columns                                    |
# MAGIC |--------------------------|--------------------------------------------|
# MAGIC | `silver_h3_elev`         | h3, avg_elev, min_elev                     |
# MAGIC | `silver_h3_slope`        | h3, avg_slope_deg                          |
# MAGIC | `silver_h3_dist_water`   | h3, dist_to_water_m                        |
# MAGIC
# MAGIC GeoBrix RasterX handles the DEM pipeline (clip, retile, slope, H3 tessellation
# MAGIC via `gbx_rst_h3_rastertogridavg`). Distance-to-water uses the built-in
# MAGIC `ST_Distance` / `ST_Intersects` available in DBR 17.1+.
# MAGIC
# MAGIC Requires a cluster with GeoBrix installed (see the `pipeline.yml` task libs).

# COMMAND ----------

dbutils.widgets.text("catalog", "flood_demo")
dbutils.widgets.text("schema", "montreal")
dbutils.widgets.text("volume", "raw")
dbutils.widgets.text("aoi_name", "greater_montreal")
dbutils.widgets.text("aoi_bbox_wkt",
                     "POLYGON((-74.05 45.30, -73.30 45.30, -73.30 45.80, -74.05 45.80, -74.05 45.30))")
dbutils.widgets.text("h3_resolution", "9")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
aoi_name = dbutils.widgets.get("aoi_name")
aoi_bbox_wkt = dbutils.widgets.get("aoi_bbox_wkt")
h3_res = int(dbutils.widgets.get("h3_resolution"))

ns = f"{catalog}.{schema}"
print(f"Namespace: {ns} | AOI: {aoi_name} | H3 res: {h3_res}")

# Multi-AOI co-residence: every silver/gold table is partitioned by aoi_name and
# we use dynamic partition overwrite so a `mode("overwrite")` write only
# replaces the active AOI's partition. See 01_ingest.py for context.
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register GeoBrix RasterX functions

# COMMAND ----------

from databricks.labs.gbx.rasterx import functions as rx
rx.register(spark)

from pyspark.sql import functions as F

# Dump key function signatures so the job log tells us exactly which arg shapes
# the installed GeoBrix build expects (useful across GeoBrix versions).
for fn in ("gbx_rst_clip", "gbx_rst_isempty", "gbx_rst_h3_rastertogridavg",
           "gbx_rst_h3_rastertogridmin"):
    try:
        spark.sql(f"DESCRIBE FUNCTION EXTENDED {fn}").show(truncate=False)
    except Exception as e:
        print(f"  (describe {fn} failed: {e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and clip DEM

# COMMAND ----------

manifest = spark.table(f"{ns}.bronze_dem_manifest").where(F.col("aoi_name") == aoi_name).first()
dem_path = manifest["dem_path"]
print("DEM path:", dem_path)

# Read every SRTM .hgt tile as a separate Spark row; GDAL's SRTMHGT driver
# picks up tile lat/lon from the filename. One row per tile lets us parallelize
# the clip and tessellation across the AOI.
dem_df = spark.read.format("gdal").load(dem_path)
print("Raw tiles loaded:", dem_df.count())

# Clip each tile to the AOI polygon using the Python API - avoids SQL-vs-Scala
# signature mismatches. `rx.rst_clip(tile_col, geom_col_or_wkb_col, cutline=True)`
# accepts either a WKB BINARY column or a GEOMETRY column depending on build;
# we feed it WKB bytes since they work across GeoBrix versions.
aoi_wkb_bytes = bytes(
    spark.sql(f"SELECT ST_AsBinary(ST_GeomFromWKT('{aoi_bbox_wkt}')) AS wkb").first()["wkb"]
)
dem_clip_df = (dem_df
    .withColumn("tile", rx.rst_clip(F.col("tile"), F.lit(aoi_wkb_bytes), F.lit(True)))
    .where(~rx.rst_isempty(F.col("tile"))))
dem_clip_df.createOrReplaceTempView("v_dem_retile")  # kept name for downstream compat
print("Clipped non-empty DEM tiles:", dem_clip_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. H3 tessellation - elevation -> silver
# MAGIC
# MAGIC GeoBrix read + clipped the tiles above. For the pixel-to-H3 step we use
# MAGIC rasterio + the `h3` Python library on the driver - it's 2 small SRTM tiles
# MAGIC (~3 M valid pixels total for Greater Montreal) so this is fast and robust
# MAGIC across GeoBrix versions.

# COMMAND ----------

import os
import numpy as np
import rasterio
import h3
from shapely.geometry import box
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

tile_paths = sorted(
    os.path.join(dem_path, fn) for fn in os.listdir(dem_path) if fn.endswith(".hgt")
)
print(f"Processing {len(tile_paths)} SRTM tiles from {dem_path}")

aoi_bounds = (
    float(manifest["min_lon"]), float(manifest["min_lat"]),
    float(manifest["max_lon"]), float(manifest["max_lat"]),
)

# h3 -> (sum, min, count) accumulator
h3_accumulator: dict[int, tuple[float, float, int]] = {}

STRIDE = 3  # subsample SRTM (~30m) by 3x3 - still ~11 samples per H3 res-9 cell

for tp in tile_paths:
    with rasterio.open(tp) as src:
        arr = src.read(1)[::STRIDE, ::STRIDE]
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else -32768
    rows, cols = arr.shape
    col_idx, row_idx = np.meshgrid(np.arange(cols) * STRIDE,
                                   np.arange(rows) * STRIDE)
    xs = transform.c + (col_idx + 0.5) * transform.a
    ys = transform.f + (row_idx + 0.5) * transform.e
    mn_lon, mn_lat, mx_lon, mx_lat = aoi_bounds
    mask = (arr != nodata) & np.isfinite(arr) \
         & (xs >= mn_lon) & (xs <= mx_lon) & (ys >= mn_lat) & (ys <= mx_lat)
    xs_f, ys_f, zs_f = xs[mask], ys[mask], arr[mask].astype(float)
    print(f"  {os.path.basename(tp)}: {xs_f.size:,} pixels inside AOI")

    for lon, lat, z in zip(xs_f, ys_f, zs_f):
        cell = h3.latlng_to_cell(float(lat), float(lon), h3_res)
        cid = int(cell, 16) if isinstance(cell, str) else int(cell)
        if cid in h3_accumulator:
            s, mn, c = h3_accumulator[cid]
            h3_accumulator[cid] = (s + z, z if z < mn else mn, c + 1)
        else:
            h3_accumulator[cid] = (z, z, 1)

print(f"Total unique H3 cells at res {h3_res}: {len(h3_accumulator):,}")

elev_rows = [(aoi_name, int(cid), float(s / c), float(mn))
             for cid, (s, mn, c) in h3_accumulator.items()]
elev_schema = StructType([
    StructField("aoi_name", StringType(), False),
    StructField("h3",       LongType(),   False),
    StructField("avg_elev", DoubleType(), False),
    StructField("min_elev", DoubleType(), False),
])
(spark.createDataFrame(elev_rows, elev_schema)
      .write.mode("overwrite")
      .partitionBy("aoi_name")
      .saveAsTable(f"{ns}.silver_h3_elev"))
print("silver_h3_elev:", spark.table(f"{ns}.silver_h3_elev")
      .where(F.col("aoi_name") == aoi_name).count(), "(this AOI)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Slope (approximated from elevation neighbourhood per H3 cell)
# MAGIC
# MAGIC Rather than running a raster-level slope convolution (GeoBrix signatures for
# MAGIC `rst_mapalgebra` / `rst_derivedband` vary across builds), we approximate a
# MAGIC per-cell slope directly in SQL using `h3_maxdistanceparent` + neighbours:
# MAGIC slope_deg ~ atan(|elev_diff_neighbour| / edge_length).

# COMMAND ----------

# Use H3 built-ins available in DBR 17.1+ to find neighbours of each cell,
# then compute max elevation difference as a proxy for slope. We compute the
# rows for the current AOI only and write them through the DataFrame API so
# dynamic partition overwrite applies — `CREATE OR REPLACE TABLE` would wipe
# every other AOI on the way through.
slope_df = spark.sql(f"""
  WITH cells AS (
    SELECT aoi_name, h3, min_elev
    FROM {ns}.silver_h3_elev
    WHERE aoi_name = '{aoi_name}'
  ),
  neighbours AS (
    SELECT c.aoi_name, c.h3, c.min_elev,
           explode(h3_kring(c.h3, 1)) AS nh
    FROM cells c
  ),
  pairs AS (
    SELECT n.aoi_name, n.h3, n.min_elev AS e1, c2.min_elev AS e2
    FROM neighbours n
    JOIN cells c2 ON n.nh = c2.h3 AND n.h3 != c2.h3
  )
  SELECT aoi_name, h3,
         CAST(
           DEGREES(ATAN( MAX(ABS(e1 - e2)) / 174.0 ))
           AS DOUBLE
         ) AS avg_slope_deg
  FROM pairs
  GROUP BY aoi_name, h3
""")
(slope_df.write.mode("overwrite")
        .partitionBy("aoi_name")
        .saveAsTable(f"{ns}.silver_h3_slope"))
print("silver_h3_slope:", spark.table(f"{ns}.silver_h3_slope")
      .where(F.col("aoi_name") == aoi_name).count(), "(this AOI)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distance to water via Spatial SQL
# MAGIC
# MAGIC DBR 17.1+ exposes `ST_GeomFromGeoJSON`, `ST_Distance`, `ST_Transform`. We
# MAGIC cast H3 cells to centroids via the `h3_centerasgeojson` built-in, then
# MAGIC compute nearest-water distance in metres using EPSG:3347 (Statistics Canada
# MAGIC Lambert) which is approximately equal-area over Quebec.

# COMMAND ----------

# Databricks Spatial SQL `ST_Transform` behaves inconsistently when the input
# geometry lacks an SRID (h3_centerasgeojson returns SRID 0), producing SRID
# mismatches downstream. We sidestep that by computing distance-to-water on
# the driver with shapely - AOI is small (~40km x 55km), ~N H3 cells x M water
# features is tractable, and we get metre accuracy via a simple local metric.

import json as _json
import math as _math
from shapely.geometry import shape as _shape
from shapely.strtree import STRtree

water_rows = spark.table(f"{ns}.bronze_hydrography") \
                  .where(F.col("aoi_name") == aoi_name) \
                  .select("geom_geojson").collect()
water_geoms = [_shape(_json.loads(r["geom_geojson"])) for r in water_rows]
water_tree = STRtree(water_geoms) if water_geoms else None
print(f"Built STRtree on {len(water_geoms):,} water features")

# Local-flat-earth metre conversion around the AOI centroid (good to ~0.5%).
_cx = (aoi_bounds[0] + aoi_bounds[2]) / 2
_cy = (aoi_bounds[1] + aoi_bounds[3]) / 2
_m_per_deg_lat = 111_320.0
_m_per_deg_lon = 111_320.0 * _math.cos(_math.radians(_cy))

h3_rows = spark.table(f"{ns}.silver_h3_elev") \
               .where(F.col("aoi_name") == aoi_name) \
               .select("h3").collect()

from shapely.geometry import Point

def _cell_to_str(cid):
    return cid if isinstance(cid, str) else hex(int(cid))[2:]

# Degree distance scaled by geometric mean of lat/lon metre factors. This is a
# reasonable approximation when the geometry type is polygon OR line OR point
# and the AOI is small (~50km).
_deg_to_m = _math.sqrt(_m_per_deg_lat * _m_per_deg_lon)

dist_rows = []
for r in h3_rows:
    cid = r["h3"]
    lat, lon = h3.cell_to_latlng(_cell_to_str(cid))
    if water_tree is None:
        dist_rows.append((aoi_name, int(cid), 1e6))
        continue
    pt = Point(lon, lat)
    result = water_tree.nearest(pt)
    nearest = water_geoms[result] if isinstance(result, (int, np.integer)) else result
    d_deg = float(pt.distance(nearest))  # works for Point/LineString/Polygon
    d_m = d_deg * _deg_to_m
    dist_rows.append((aoi_name, int(cid), float(d_m)))

dist_schema = StructType([
    StructField("aoi_name",         StringType(), False),
    StructField("h3",               LongType(),   False),
    StructField("dist_to_water_m",  DoubleType(), False),
])
(spark.createDataFrame(dist_rows, dist_schema)
      .write.mode("overwrite")
      .partitionBy("aoi_name")
      .saveAsTable(f"{ns}.silver_h3_dist_water"))
print("silver_h3_dist_water:", spark.table(f"{ns}.silver_h3_dist_water")
      .where(F.col("aoi_name") == aoi_name).count(), "(this AOI)")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Assign precipitation climatology to each H3 cell
# MAGIC
# MAGIC Open-Meteo gives us precip at ~0.1 degree grid points; we attach each H3 cell
# MAGIC to its nearest grid point using `ST_Distance` on 4326 centroids (sub-degree
# MAGIC differences, so geodesic distance is close enough for nearest-neighbour).

# COMMAND ----------

# Assign each H3 cell to its nearest precip grid point in Python (O(H x P) with
# P ~ 9, trivial). Uses an STRtree for robustness as we scale to bigger AOIs.
precip_pts = spark.table(f"{ns}.bronze_precip_grid") \
                  .where(F.col("aoi_name") == aoi_name).collect()
precip_geoms = [Point(float(r["lon"]), float(r["lat"])) for r in precip_pts]
precip_tree = STRtree(precip_geoms) if precip_geoms else None

precip_rows = []
for r in h3_rows:
    cid = r["h3"]
    lat, lon = h3.cell_to_latlng(_cell_to_str(cid))
    if precip_tree is None:
        precip_rows.append((aoi_name, int(cid), 0.0, 0.0, 0.0))
        continue
    pt = Point(lon, lat)
    result = precip_tree.nearest(pt)
    idx = int(result) if isinstance(result, (int, np.integer)) else precip_geoms.index(result)
    src = precip_pts[idx]
    precip_rows.append((aoi_name, int(cid),
                        float(src["annual_precip_mm"]),
                        float(src["max24h_precip_mm"]),
                        float(src["max5d_precip_mm"])))

precip_schema = StructType([
    StructField("aoi_name",          StringType(), False),
    StructField("h3",                LongType(),   False),
    StructField("annual_precip_mm",  DoubleType(), False),
    StructField("max24h_precip_mm",  DoubleType(), False),
    StructField("max5d_precip_mm",   DoubleType(), False),
])
(spark.createDataFrame(precip_rows, precip_schema)
      .write.mode("overwrite")
      .partitionBy("aoi_name")
      .saveAsTable(f"{ns}.silver_h3_precip"))
print("silver_h3_precip:", spark.table(f"{ns}.silver_h3_precip")
      .where(F.col("aoi_name") == aoi_name).count(), "(this AOI)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5b. Buildings per H3 cell (exposure layer)
# MAGIC
# MAGIC Convert every OSM building centroid into an H3 cell at the feature
# MAGIC resolution and aggregate. We also track a residential subcount, which
# MAGIC is the subset of building tags most clearly tied to people-at-risk.

# COMMAND ----------

from pyspark.sql.types import IntegerType

bld_pts = (spark.table(f"{ns}.bronze_buildings")
                .where(F.col("aoi_name") == aoi_name)
                .select("lat", "lon", "residential").collect())
print(f"Tessellating {len(bld_pts):,} building centroids to H3 res {h3_res}")

bld_acc: dict[int, tuple[int, int]] = {}   # cell -> (count_all, count_residential)
for r in bld_pts:
    cell_str = h3.latlng_to_cell(float(r["lat"]), float(r["lon"]), h3_res)
    cid = int(cell_str, 16) if isinstance(cell_str, str) else int(cell_str)
    a, b = bld_acc.get(cid, (0, 0))
    bld_acc[cid] = (a + 1, b + (1 if int(r["residential"] or 0) else 0))

bld_rows = [(aoi_name, int(cid), int(a), int(b)) for cid, (a, b) in bld_acc.items()]
bld_schema = StructType([
    StructField("aoi_name",          StringType(),  False),
    StructField("h3",                LongType(),    False),
    StructField("building_count",    IntegerType(), False),
    StructField("residential_count", IntegerType(), False),
])
(spark.createDataFrame(bld_rows, bld_schema)
      .write.mode("overwrite")
      .partitionBy("aoi_name")
      .saveAsTable(f"{ns}.silver_h3_buildings"))
print("silver_h3_buildings:", spark.table(f"{ns}.silver_h3_buildings")
      .where(F.col("aoi_name") == aoi_name).count(), "(this AOI)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Exclude cells that are actually waterways
# MAGIC
# MAGIC Cells whose centroid sits on top of a water polygon/line have
# MAGIC `dist_to_water_m = 0` (shapely returns 0 for points inside/on the geom).
# MAGIC These cells are river/lake surfaces, not flood-prone land, so we drop
# MAGIC them from every silver table. `< 5 m` is a conservative threshold
# MAGIC that handles both OSM geometry slop and H3 cell-centre quantization.

# COMMAND ----------

WATER_DIST_THRESHOLD_M = 5.0

water_cells_sql = f"""
  SELECT h3 FROM {ns}.silver_h3_dist_water
  WHERE aoi_name = '{aoi_name}' AND dist_to_water_m < {WATER_DIST_THRESHOLD_M}
"""
water_cells_df = spark.sql(water_cells_sql)
n_dropped = water_cells_df.count()
print(f"Dropping {n_dropped} in-water cells (dist_to_water_m < {WATER_DIST_THRESHOLD_M} m)")

water_cells_df.createOrReplaceTempView("_water_cells_to_drop")
for tbl in ("silver_h3_elev", "silver_h3_slope",
            "silver_h3_precip", "silver_h3_dist_water",
            "silver_h3_buildings"):
    spark.sql(f"""
      MERGE INTO {ns}.{tbl} t
      USING _water_cells_to_drop s
      ON t.h3 = s.h3 AND t.aoi_name = '{aoi_name}'
      WHEN MATCHED THEN DELETE
    """)
    n_aoi = spark.table(f"{ns}.{tbl}").where(F.col("aoi_name") == aoi_name).count()
    print(f"  {tbl}: {n_aoi} rows (this AOI)")

# COMMAND ----------

display(spark.sql(f"""
  SELECT e.h3, e.avg_elev, e.min_elev, s.avg_slope_deg,
         d.dist_to_water_m, p.annual_precip_mm, p.max24h_precip_mm,
         COALESCE(b.building_count, 0)    AS building_count,
         COALESCE(b.residential_count, 0) AS residential_count
  FROM {ns}.silver_h3_elev e
  LEFT JOIN {ns}.silver_h3_slope      s USING (aoi_name, h3)
  LEFT JOIN {ns}.silver_h3_dist_water d USING (aoi_name, h3)
  LEFT JOIN {ns}.silver_h3_precip     p USING (aoi_name, h3)
  LEFT JOIN {ns}.silver_h3_buildings  b USING (aoi_name, h3)
  WHERE e.aoi_name = '{aoi_name}'
  ORDER BY b.building_count DESC NULLS LAST
  LIMIT 20
"""))
