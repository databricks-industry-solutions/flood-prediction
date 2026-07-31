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
# MAGIC GeoBrix RasterX handles the DEM pipeline (split, clip, H3 tessellation via
# MAGIC `gbx_rst_h3_rastertogridavg`). Distance-to-water uses the built-in
# MAGIC `ST_Distance` / `ST_Intersects` available in DBR 17.1+.
# MAGIC
# MAGIC Runs on **serverless** using the GeoBrix *lightweight* tier - pure Python
# MAGIC (rasterio / h3 backed), no JAR and no GDAL init script.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install GeoBrix
# MAGIC
# MAGIC GeoBrix is not on PyPI, so we install the published release wheel with its
# MAGIC `light` extra. This is a notebook-scoped install rather than a job
# MAGIC `environments:` dependency on purpose - notebook-scoped libraries
# MAGIC propagate to the Python workers that run the `gbx_rst_*` table functions,
# MAGIC and they work on workspaces where the serverless environment builder
# MAGIC cannot reach external package sources.

# COMMAND ----------

# MAGIC %pip install --quiet "geobrix[light] @ https://github.com/databrickslabs/geobrix/releases/download/v0.4.3/geobrix-0.4.3-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()

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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-AOI partition writes
# MAGIC
# MAGIC Every silver/gold table is partitioned by `aoi_name` so multiple AOIs
# MAGIC co-exist in one catalog. Serverless rejects
# MAGIC `spark.sql.sources.partitionOverwriteMode=dynamic`
# MAGIC (`CONFIG_NOT_AVAILABLE`), so we scope each overwrite with Delta's
# MAGIC `replaceWhere` instead - same effect, and it works on every compute type.

# COMMAND ----------

from pyspark.sql import functions as F


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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register GeoBrix RasterX (lightweight tier)
# MAGIC
# MAGIC `pyrx` is the pure-Python implementation - same `gbx_rst_*` SQL names as
# MAGIC the JVM tier, no JAR or GDAL install, and the only tier that runs on
# MAGIC serverless. The readers are not auto-registered, so we register
# MAGIC `raster_gbx` explicitly.

# COMMAND ----------

from databricks.labs.gbx.pyrx import functions as rx
from databricks.labs.gbx.ds.register import register as gbx_register

rx.register(spark)
gbx_register(spark, only=["raster_gbx"])

# Fail loudly here rather than deep inside a LATERAL query if the installed
# build is missing something we depend on.
_required = ("gbx_rst_clip", "gbx_rst_isempty", "gbx_rst_h3_rastertogridavg",
             "gbx_rst_h3_rastertogridmin", "gbx_rst_h3_rastertogridcount")
_available = {r.function.split(".")[-1]
              for r in spark.sql("SHOW FUNCTIONS LIKE 'gbx_rst_*'").collect()}
_missing = [f for f in _required if f not in _available]
if _missing:
    raise RuntimeError(
        f"GeoBrix build is missing required functions: {_missing}. "
        f"Found {len(_available)} gbx_rst_* functions."
    )
print(f"GeoBrix lightweight ready ({len(_available)} gbx_rst_* functions)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and clip DEM

# COMMAND ----------

manifest = spark.table(f"{ns}.bronze_dem_manifest").where(F.col("aoi_name") == aoi_name).first()
dem_path = manifest["dem_path"]
print("DEM path:", dem_path)

# `raster_gbx` is the lightweight (rasterio-backed) reader. It resolves the SRTM
# `.hgt` tiles directly - rasterio's SRTMHGT driver derives CRS and bounds from
# the filename, so no GeoTIFF conversion step is needed.
#
# sizeInMB is NOT optional: the default (-1) emits one whole-image tile per file,
# and tessellating a full 3601x3601 SRTM tile in a single Python worker exhausts
# its memory. Splitting also gives us real parallelism across the AOI.
DEM_TILE_SPLIT_MB = "4"

dem_df = (spark.read.format("raster_gbx")
          .option("sizeInMB", DEM_TILE_SPLIT_MB)
          .load(dem_path))
dem_df.createOrReplaceTempView("v_dem_raw")
print(f"DEM split into {dem_df.count()} tiles at sizeInMB={DEM_TILE_SPLIT_MB}")

# Clip each split tile to the AOI polygon. Pixels outside the cutline become
# nodata and are ignored by the tessellation aggregates below; tiles that fall
# entirely outside the AOI drop out via gbx_rst_isempty.
#
# The tessellation aggregates (avg / count / min) each need to be evaluated
# against the *same* set of tiles and paired per tile, so we materialize the
# clipped tiles with a stable id first. `monotonically_increasing_id()` is only
# stable once persisted - recomputing the view would reassign the ids and
# silently corrupt the joins below.
scratch_tiles = f"{ns}._scratch_dem_tiles"

clipped_df = spark.sql(f"""
    SELECT gbx_rst_clip(tile,
                        ST_AsBinary(ST_GeomFromWKT('{aoi_bbox_wkt}')),
                        true) AS tile
    FROM v_dem_raw
""").where(~rx.rst_isempty(F.col("tile")))

(clipped_df
    .withColumn("aoi_name", F.lit(aoi_name))
    .withColumn("tile_id", F.monotonically_increasing_id())
    .write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(scratch_tiles))

spark.table(scratch_tiles).createOrReplaceTempView("v_dem_tiles")
print("Clipped non-empty DEM tiles:", spark.table(scratch_tiles).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. H3 tessellation - elevation -> silver
# MAGIC
# MAGIC `gbx_rst_h3_rastertogrid*` maps every DEM pixel onto an H3 cell and
# MAGIC aggregates, distributed across the split tiles. These are table functions,
# MAGIC so they are invoked as SQL `LATERAL` calls - the Python wrappers raise
# MAGIC `NotImplementedError` in the lightweight tier.
# MAGIC
# MAGIC **Seams.** Because we split the DEM, a cell sitting on a tile boundary is
# MAGIC reported once per tile it touches, each time averaging only that tile's
# MAGIC pixels. Averaging those partial averages would silently over-weight the
# MAGIC tile that contributed fewer pixels, so we pull the per-tile pixel counts
# MAGIC as well and recombine with a pixel-weighted mean. `min` needs no weighting.

# COMMAND ----------

import numpy as np
import h3
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

aoi_bounds = (
    float(manifest["min_lon"]), float(manifest["min_lat"]),
    float(manifest["max_lon"]), float(manifest["max_lat"]),
)

elev_df = spark.sql(f"""
  WITH avg_cells AS (
    SELECT r.tile_id, t.cellID AS h3, t.measure AS avg_elev
    FROM v_dem_tiles r, LATERAL gbx_rst_h3_rastertogridavg(r.tile, {h3_res}) t
  ),
  cnt_cells AS (
    SELECT r.tile_id, t.cellID AS h3, t.measure AS px
    FROM v_dem_tiles r, LATERAL gbx_rst_h3_rastertogridcount(r.tile, {h3_res}) t
  ),
  min_cells AS (
    SELECT r.tile_id, t.cellID AS h3, t.measure AS min_elev
    FROM v_dem_tiles r, LATERAL gbx_rst_h3_rastertogridmin(r.tile, {h3_res}) t
  )
  SELECT a.h3,
         SUM(a.avg_elev * c.px) / SUM(c.px) AS avg_elev,
         MIN(m.min_elev)                    AS min_elev
  FROM avg_cells a
  JOIN cnt_cells c ON a.tile_id = c.tile_id AND a.h3 = c.h3
  JOIN min_cells m ON a.tile_id = m.tile_id AND a.h3 = m.h3
  GROUP BY a.h3
  HAVING SUM(c.px) > 0
""")

# GeoBrix returns cellID as BIGINT on current builds, but older ones hand back
# the hex string form. Downstream tables and the H3 built-ins both want BIGINT.
if dict(elev_df.dtypes)["h3"] == "string":
    elev_df = elev_df.withColumn("h3", F.conv(F.col("h3"), 16, 10).cast("long"))

elev_df = (elev_df
    .withColumn("aoi_name", F.lit(aoi_name))
    .select(
        F.col("aoi_name"),
        F.col("h3").cast("long").alias("h3"),
        F.col("avg_elev").cast("double").alias("avg_elev"),
        F.col("min_elev").cast("double").alias("min_elev"),
    ))

write_aoi_partition(elev_df, f"{ns}.silver_h3_elev")
print("silver_h3_elev:", spark.table(f"{ns}.silver_h3_elev")
      .where(F.col("aoi_name") == aoi_name).count(), "(this AOI)")

# The clipped-tile rasters are only needed for the tessellation above.
spark.sql(f"DROP TABLE IF EXISTS {scratch_tiles}")

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
# rows for the current AOI only and write them through the DataFrame API so the
# replaceWhere scope applies — `CREATE OR REPLACE TABLE` would wipe every other
# AOI on the way through.
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
write_aoi_partition(slope_df, f"{ns}.silver_h3_slope")
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
write_aoi_partition(spark.createDataFrame(dist_rows, dist_schema),
                    f"{ns}.silver_h3_dist_water")
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
write_aoi_partition(spark.createDataFrame(precip_rows, precip_schema),
                    f"{ns}.silver_h3_precip")
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
write_aoi_partition(spark.createDataFrame(bld_rows, bld_schema),
                    f"{ns}.silver_h3_buildings")
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
