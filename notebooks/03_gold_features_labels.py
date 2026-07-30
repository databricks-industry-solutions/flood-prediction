# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Gold features + hybrid labels (with rainfall scenarios)
# MAGIC
# MAGIC Outputs three gold tables:
# MAGIC
# MAGIC * `gold_h3_features(h3, terrain + precip climatology)` - one row per H3 cell.
# MAGIC * `gold_h3_training(h3, features..., scenario_24h_mm, label)` - the
# MAGIC   **scenario-expanded** training set: each cell is replicated for every
# MAGIC   discrete 24-hour rainfall scenario (e.g. 10, 30, 60, 100, 150, 200 mm) with
# MAGIC   a synthetic label that depends on both terrain and rainfall, so the model
# MAGIC   actually learns the rainfall effect.
# MAGIC * `gold_h3_labels(h3, label_real)` - hold-out real 2017/2019 flood label
# MAGIC   used only for validation / overlay.
# MAGIC
# MAGIC Parameter `scenarios_24h_mm` is a CSV string of rainfall scenarios shared with
# MAGIC notebook 04 and the app so they all use the same partitions.

# COMMAND ----------

dbutils.widgets.text("catalog", "flood_demo")
dbutils.widgets.text("schema", "montreal")
dbutils.widgets.text("aoi_name", "greater_montreal")
dbutils.widgets.text("h3_resolution", "9")
dbutils.widgets.text("scenarios_24h_mm", "10,30,60,100,150,200")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
aoi_name = dbutils.widgets.get("aoi_name")
h3_res = int(dbutils.widgets.get("h3_resolution"))
scenarios = [int(s.strip()) for s in dbutils.widgets.get("scenarios_24h_mm").split(",") if s.strip()]
ns = f"{catalog}.{schema}"
print("Scenarios (mm/24h):", scenarios)

# Multi-AOI co-residence: every gold table is partitioned by aoi_name and we
# write only the active AOI's rows. `CREATE OR REPLACE TABLE ... AS SELECT`
# would replace the whole table on each run, so we compute the rows via
# spark.sql() and write through the DataFrame API with dynamic partition
# overwrite — see 01_ingest.py for the full rationale.
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Build feature table

# COMMAND ----------

features_df = spark.sql(f"""
  WITH joined AS (
    SELECT e.aoi_name,
           e.h3,
           e.avg_elev,
           e.min_elev,
           COALESCE(s.avg_slope_deg, 0.0)      AS slope_deg,
           COALESCE(d.dist_to_water_m, 1e6)    AS dist_to_water_m,
           COALESCE(p.annual_precip_mm, 0.0)   AS annual_precip_mm,
           COALESCE(p.max24h_precip_mm, 0.0)   AS max24h_precip_mm,
           COALESCE(p.max5d_precip_mm,  0.0)   AS max5d_precip_mm,
           -- Exposure columns: carried as metadata, NOT used as model features
           -- so we don't leak location into the terrain model.
           COALESCE(b.building_count, 0)        AS building_count,
           COALESCE(b.residential_count, 0)     AS residential_count
    FROM {ns}.silver_h3_elev e
    LEFT JOIN {ns}.silver_h3_slope      s USING (aoi_name, h3)
    LEFT JOIN {ns}.silver_h3_dist_water d USING (aoi_name, h3)
    LEFT JOIN {ns}.silver_h3_precip     p USING (aoi_name, h3)
    LEFT JOIN {ns}.silver_h3_buildings  b USING (aoi_name, h3)
    WHERE e.aoi_name = '{aoi_name}'
  )
  SELECT *,
         -- Topographic Wetness Index proxy
         LN(1.0 / (slope_deg + 0.5)) - LN(dist_to_water_m + 1.0) AS twi
  FROM joined
""")
(features_df.write.mode("overwrite")
            .partitionBy("aoi_name")
            .saveAsTable(f"{ns}.gold_h3_features"))
print("gold_h3_features:",
      spark.table(f"{ns}.gold_h3_features")
           .where(F.col("aoi_name") == aoi_name).count(),
      "(this AOI)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-event labels (held out)

# COMMAND ----------

spark.sql(f"""
  CREATE OR REPLACE TEMP VIEW v_flood_polys AS
  SELECT ST_GeomFromGeoJSON(geom_geojson) AS geom, year
  FROM {ns}.bronze_flood_events
  WHERE aoi_name = '{aoi_name}'
""")

labels_df = spark.sql(f"""
  WITH centroids AS (
    -- Restrict to the current AOI's cells; without this filter the labels
    -- table would tag every other AOI's H3 cells with the current AOI's
    -- flood polygons (which is silently wrong even when the spatial join
    -- returns zero matches because the row carries the wrong aoi_name).
    SELECT h3, ST_GeomFromGeoJSON(h3_centerasgeojson(h3)) AS geom
    FROM {ns}.gold_h3_features
    WHERE aoi_name = '{aoi_name}'
  )
  SELECT '{aoi_name}' AS aoi_name,
         c.h3,
         CASE WHEN COUNT(f.geom) > 0 THEN 1 ELSE 0 END AS label_real
  FROM centroids c
  LEFT JOIN v_flood_polys f ON ST_Intersects(c.geom, f.geom)
  GROUP BY c.h3
""")
(labels_df.write.mode("overwrite")
          .partitionBy("aoi_name")
          .saveAsTable(f"{ns}.gold_h3_labels"))
print("gold_h3_labels positives:", spark.sql(
    f"SELECT SUM(label_real) FROM {ns}.gold_h3_labels "
    f"WHERE aoi_name = '{aoi_name}'").collect()[0][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario-expanded training set with rainfall-dependent synthetic label
# MAGIC
# MAGIC For each H3 cell we create one row per rainfall scenario. The synthetic
# MAGIC label is:
# MAGIC
# MAGIC ```
# MAGIC susceptibility = w1 * low_elev_score
# MAGIC                + w2 * near_water_score
# MAGIC                + w3 * low_slope_score
# MAGIC                + w4 * wet_climatology_score
# MAGIC rain_factor    = min(1.0, scenario_24h_mm / 150)
# MAGIC p_flood        = susceptibility * rain_factor + noise
# MAGIC label          = 1 if p_flood > 0.5 else 0
# MAGIC ```
# MAGIC
# MAGIC All scores are normalized to [0, 1] using AOI quantiles so the model learns
# MAGIC a proper rainfall response function instead of a constant mapping.

# COMMAND ----------

q = (spark.sql(f"""
  SELECT
    percentile_approx(min_elev, 0.20)        AS p20_elev,
    percentile_approx(min_elev, 0.80)        AS p80_elev,
    percentile_approx(dist_to_water_m, 0.80) AS p80_dist,
    percentile_approx(slope_deg, 0.80)       AS p80_slope,
    percentile_approx(annual_precip_mm, 0.20) AS p20_precip,
    percentile_approx(annual_precip_mm, 0.80) AS p80_precip
  FROM {ns}.gold_h3_features
  WHERE aoi_name = '{aoi_name}'
""").collect()[0])
p20_elev, p80_elev = float(q["p20_elev"]), float(q["p80_elev"])
p80_dist = float(q["p80_dist"])
p80_slope = float(q["p80_slope"])
p20_precip, p80_precip = float(q["p20_precip"]), float(q["p80_precip"])
print(f"Quantiles: elev [{p20_elev:.1f}, {p80_elev:.1f}]  dist_p80={p80_dist:.0f}  "
      f"slope_p80={p80_slope:.2f}  precip [{p20_precip:.0f}, {p80_precip:.0f}]")

# COMMAND ----------

scenarios_sql = ", ".join(f"({s})" for s in scenarios)

training_df = spark.sql(f"""
  WITH feats AS (
    SELECT * FROM {ns}.gold_h3_features
    WHERE aoi_name = '{aoi_name}'
  ),
  scen(scenario_24h_mm) AS (VALUES {scenarios_sql}),
  cross AS (
    SELECT f.*, CAST(s.scenario_24h_mm AS DOUBLE) AS scenario_24h_mm
    FROM feats f CROSS JOIN scen s
  ),
  scored AS (
    SELECT *,
      -- Normalize each driver to [0,1] where 1 = high flood susceptibility.
      GREATEST(0.0, LEAST(1.0,
        ({p80_elev} - min_elev) / NULLIF({p80_elev} - {p20_elev}, 0)
      )) AS low_elev_score,
      GREATEST(0.0, LEAST(1.0,
        1.0 - (dist_to_water_m / NULLIF({p80_dist}, 0))
      )) AS near_water_score,
      GREATEST(0.0, LEAST(1.0,
        1.0 - (slope_deg / NULLIF({p80_slope}, 0))
      )) AS low_slope_score,
      GREATEST(0.0, LEAST(1.0,
        (annual_precip_mm - {p20_precip}) / NULLIF({p80_precip} - {p20_precip}, 0)
      )) AS wet_clim_score,
      LEAST(1.0, scenario_24h_mm / 150.0) AS rain_factor
    FROM cross
  ),
  labelled AS (
    SELECT *,
      (0.35 * low_elev_score + 0.30 * near_water_score +
       0.20 * low_slope_score + 0.15 * wet_clim_score) AS susceptibility,
      -- Deterministic pseudo-noise keyed on h3 + scenario so training is reproducible
      (CAST(xxhash64(h3, scenario_24h_mm) AS BIGINT) % 10000) / 10000.0 AS u
    FROM scored
  )
  SELECT aoi_name, h3,
         avg_elev, min_elev, slope_deg, dist_to_water_m, twi,
         annual_precip_mm, max24h_precip_mm, max5d_precip_mm,
         scenario_24h_mm,
         CASE WHEN (susceptibility * rain_factor) + (u - 0.5) * 0.15 > 0.5
              THEN 1 ELSE 0 END AS label_synthetic
  FROM labelled
""")
(training_df.write.mode("overwrite")
            .partitionBy("aoi_name")
            .saveAsTable(f"{ns}.gold_h3_training"))
print("gold_h3_training rows (this AOI):",
      spark.table(f"{ns}.gold_h3_training")
           .where(F.col("aoi_name") == aoi_name).count())

display(spark.sql(f"""
  SELECT scenario_24h_mm,
         COUNT(*) AS rows,
         ROUND(AVG(label_synthetic), 4) AS positive_rate
  FROM {ns}.gold_h3_training
  WHERE aoi_name = '{aoi_name}'
  GROUP BY scenario_24h_mm
  ORDER BY scenario_24h_mm
"""))
