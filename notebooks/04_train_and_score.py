# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Train Random Forest, register model, score per-scenario
# MAGIC
# MAGIC * Trains `RandomForestClassifier` on the **scenario-expanded** `gold_h3_training`
# MAGIC   set so the model learns the flood response to 24-hour rainfall.
# MAGIC * Logs and registers the model in Unity Catalog via MLflow.
# MAGIC * Scores every H3 cell at each discrete rainfall scenario and persists
# MAGIC   `gold_h3_flood_predictions` partitioned by `(aoi_name, scenario_24h_mm)` so
# MAGIC   the app can switch partitions on every slider move.

# COMMAND ----------

dbutils.widgets.text("catalog", "flood_demo")
dbutils.widgets.text("schema", "montreal")
dbutils.widgets.text("volume", "raw")
dbutils.widgets.text("aoi_name", "greater_montreal")
dbutils.widgets.text("model_name", "flood_demo.montreal.flood_rf")
dbutils.widgets.text("scenarios_24h_mm", "10,30,60,100,150,200")
dbutils.widgets.text("h3_resolution", "9")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
aoi_name = dbutils.widgets.get("aoi_name")
model_name = dbutils.widgets.get("model_name")
h3_res = int(dbutils.widgets.get("h3_resolution"))
scenarios = [int(s.strip()) for s in dbutils.widgets.get("scenarios_24h_mm").split(",") if s.strip()]
ns = f"{catalog}.{schema}"
print("Scenarios (mm/24h):", scenarios)

# Multi-AOI co-residence: every gold table is partitioned by aoi_name and we
# only overwrite the active AOI's partitions. Each AOI's training run still
# registers a new UC model version under the same name — the App scores cells
# directly off `gold_h3_flood_predictions`, so model versioning is independent.
#
# Serverless rejects spark.sql.sources.partitionOverwriteMode=dynamic, so the
# scoping is done with Delta's replaceWhere instead.


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

import mlflow
import mlflow.spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

mlflow.set_registry_uri("databricks-uc")

FEATURES = [
    "avg_elev", "min_elev", "slope_deg", "dist_to_water_m", "twi",
    "annual_precip_mm", "max24h_precip_mm", "max5d_precip_mm",
    "scenario_24h_mm",
]

train_base = (spark.table(f"{ns}.gold_h3_training")
                    .where(F.col("aoi_name") == aoi_name)
                    .na.drop(subset=FEATURES))
print("Training rows:", train_base.count())

train, test = train_base.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label_synthetic",
    numTrees=100,
    maxDepth=8,
    seed=42,
)
pipe = Pipeline(stages=[assembler, rf])

with mlflow.start_run(run_name=f"flood_rf_{aoi_name}") as run:
    model = pipe.fit(train)

    pred_test = model.transform(test)
    auc = BinaryClassificationEvaluator(
        labelCol="label_synthetic", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(pred_test)
    mlflow.log_metric("test_auc", auc)
    print(f"AUC on synthetic holdout: {auc:.3f}")

    # Feature importances
    rf_model = model.stages[-1]
    for name, imp in sorted(zip(FEATURES, rf_model.featureImportances.toArray()),
                            key=lambda x: -x[1]):
        mlflow.log_metric(f"importance__{name}", float(imp))
        print(f"  {name}: {imp:.4f}")

    from mlflow.models.signature import infer_signature
    sample_input = train.select(FEATURES).limit(10).toPandas()
    sample_pred = model.transform(train.limit(10)).select("prediction").toPandas()
    signature = infer_signature(sample_input, sample_pred)
    # On serverless (and shared clusters) MLflow refuses to stage a SparkML
    # model through the driver's local filesystem, so it needs a UC Volume to
    # write the intermediate model directory to.
    mlflow.spark.log_model(
        spark_model=model,
        artifact_path="model",
        registered_model_name=model_name,
        input_example=sample_input.head(3),
        signature=signature,
        dfs_tmpdir=f"/Volumes/{catalog}/{schema}/{volume}/_mlflow_tmp",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score every H3 cell at every scenario

# COMMAND ----------

prob_pos = F.udf(lambda v: float(v[1]) if v is not None else None, DoubleType())

features = spark.table(f"{ns}.gold_h3_features").where(F.col("aoi_name") == aoi_name)
labels = spark.table(f"{ns}.gold_h3_labels").select("aoi_name", "h3", "label_real")

scen_df = spark.createDataFrame([(int(s),) for s in scenarios], ["scenario_24h_mm"])

score_input = (features.crossJoin(scen_df)
                        .join(labels, ["aoi_name", "h3"], "left")
                        .withColumn("scenario_24h_mm",
                                    F.col("scenario_24h_mm").cast("double")))

scored = model.transform(score_input)

publish = (scored
  .withColumn("flood_prob", prob_pos(F.col("probability")))
  .select("aoi_name", "h3", "scenario_24h_mm", "avg_elev", "min_elev",
          "slope_deg", "dist_to_water_m", "twi",
          "annual_precip_mm", "max24h_precip_mm", "max5d_precip_mm",
          "building_count", "residential_count",
          "flood_prob",
          F.coalesce(F.col("label_real"), F.lit(0)).cast("int").alias("label_real"))
  .withColumn("scenario_24h_mm", F.col("scenario_24h_mm").cast("int")))

publish.createOrReplaceTempView("v_scored")

final = spark.sql("""
  SELECT aoi_name, scenario_24h_mm, h3,
         h3_boundaryasgeojson(h3) AS geometry_geojson,
         avg_elev, min_elev, slope_deg, dist_to_water_m, twi,
         annual_precip_mm, max24h_precip_mm, max5d_precip_mm,
         building_count, residential_count,
         CAST(flood_prob * building_count    AS DOUBLE) AS expected_buildings_at_risk,
         CAST(flood_prob * residential_count AS DOUBLE) AS expected_residential_at_risk,
         flood_prob, label_real
  FROM v_scored
""")

write_aoi_partition(final, f"{ns}.gold_h3_flood_predictions",
                    extra_partitions=("scenario_24h_mm",),
                    options={"delta.columnMapping.mode": "name"})

# gold_flood_events is the App's overlay source. We rewrite only the current
# AOI's partition; other AOIs' rows survive untouched.
fe_df = spark.sql(f"""
  SELECT aoi_name, year, geom_geojson AS geometry_geojson
  FROM {ns}.bronze_flood_events
  WHERE aoi_name = '{aoi_name}'
""")
write_aoi_partition(fe_df, f"{ns}.gold_flood_events")

# gold_aoi is the App's AOI-dropdown source. One row per AOI; we rewrite our
# own row from this AOI's bronze_dem_manifest entry.
aoi_meta_df = spark.sql(f"""
  SELECT aoi_name,
         MIN(min_lon) AS min_lon, MIN(min_lat) AS min_lat,
         MAX(max_lon) AS max_lon, MAX(max_lat) AS max_lat
  FROM {ns}.bronze_dem_manifest
  WHERE aoi_name = '{aoi_name}'
  GROUP BY aoi_name
""")
write_aoi_partition(aoi_meta_df, f"{ns}.gold_aoi")

# Scenario registry — lets the App enumerate available rainfall partitions per AOI.
scen_rows = [(aoi_name, int(s)) for s in scenarios]
write_aoi_partition(spark.createDataFrame(scen_rows, ["aoi_name", "scenario_24h_mm"]),
                    f"{ns}.gold_scenarios")

_n_partitions_for_aoi = (
    spark.table(f"{ns}.gold_h3_flood_predictions")
         .where(F.col("aoi_name") == aoi_name)
         .select("scenario_24h_mm").distinct().count()
)
print(f"Partitions written for this AOI: {_n_partitions_for_aoi} (one per scenario)")

display(spark.sql(f"""
  SELECT scenario_24h_mm,
         COUNT(*) AS cells,
         ROUND(AVG(flood_prob), 3) AS mean_prob,
         ROUND(AVG(CASE WHEN flood_prob > 0.5 THEN 1.0 ELSE 0.0 END), 3) AS high_risk_rate,
         SUM(label_real) AS real_flood_cells
  FROM {ns}.gold_h3_flood_predictions
  WHERE aoi_name = '{aoi_name}'
  GROUP BY scenario_24h_mm
  ORDER BY scenario_24h_mm
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Underwriting tables (chatbot surface)
# MAGIC
# MAGIC Two tables that turn the cell-level predictions into a building-level
# MAGIC view that the Genie Space chatbot can talk to:
# MAGIC
# MAGIC * `gold_underwriting_assumptions` - two-row reference table holding the
# MAGIC   demo-grade BRC + loss-severity constants per building class. Created
# MAGIC   with `IF NOT EXISTS` so an SA can edit the numbers in-place via SQL
# MAGIC   without the pipeline blowing them away on next deploy.
# MAGIC * `gold_building_exposure` - one row per (aoi, scenario, building),
# MAGIC   joining `bronze_buildings` centroids to `gold_h3_flood_predictions`
# MAGIC   by H3 and to the assumptions table by class. Includes pre-computed
# MAGIC   `risk_tier`, `expected_loss_usd`, and `inside_historical_flood`
# MAGIC   (the cell's 2017/2019 label) so Genie answers stay single-SELECT.

# COMMAND ----------

# Reference table for replacement-cost + loss-severity assumptions. Demo-grade
# defaults; an SA can `UPDATE gold_underwriting_assumptions SET brc_usd = ...`
# in the workspace and the next pipeline run will preserve the edit because of
# IF NOT EXISTS.
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {ns}.gold_underwriting_assumptions (
    aoi_name       STRING,
    building_class STRING,
    brc_usd        DOUBLE,
    loss_severity  DOUBLE,
    notes          STRING
  ) USING DELTA
""")
_asm_count = spark.table(f"{ns}.gold_underwriting_assumptions").count()
if _asm_count == 0:
    spark.createDataFrame(
        [
            ("*", "residential", 300000.0, 0.25,
             "Demo default. Replace with the carrier's policy book or "
             "CoreLogic/Verisk replacement-cost estimator."),
            ("*", "commercial",  1500000.0, 0.25,
             "Demo default. Same source caveat as residential. Loss "
             "severity is a flat FEMA-style assumption; production would "
             "use a depth-damage curve."),
        ],
        ["aoi_name", "building_class", "brc_usd", "loss_severity", "notes"],
    ).write.mode("append").saveAsTable(f"{ns}.gold_underwriting_assumptions")
print("gold_underwriting_assumptions rows:",
      spark.table(f"{ns}.gold_underwriting_assumptions").count())

# COMMAND ----------

# Building-level exposure. One row per (aoi, scenario, building). We compute
# rows for the current AOI only and write through the DataFrame API so dynamic
# partition overwrite leaves other AOIs untouched. Joining the assumptions
# table (rather than hardcoding the constants) lets the SA edit BRC/severity
# without changing pipeline code.
exposure_df = spark.sql(f"""
  WITH bld AS (
    SELECT
      b.aoi_name,
      b.osm_id,
      COALESCE(NULLIF(b.building, ''), 'unknown') AS building_type,
      b.residential,
      b.lon, b.lat,
      h3_longlatash3(b.lon, b.lat, {h3_res}) AS h3
    FROM {ns}.bronze_buildings b
    WHERE b.aoi_name = '{aoi_name}'
  ),
  asm AS (
    SELECT building_class, brc_usd, loss_severity
    FROM {ns}.gold_underwriting_assumptions
    WHERE aoi_name = '*'
  )
  SELECT
    p.aoi_name,
    p.scenario_24h_mm,
    bld.osm_id,
    bld.building_type,
    bld.residential,
    bld.h3,
    bld.lon,
    bld.lat,
    p.flood_prob,
    CASE
      WHEN p.flood_prob >= 0.6 THEN 'severe'
      WHEN p.flood_prob >= 0.3 THEN 'high'
      WHEN p.flood_prob >= 0.1 THEN 'moderate'
      ELSE 'low'
    END AS risk_tier,
    asm.brc_usd,
    asm.loss_severity,
    CAST(p.flood_prob * asm.brc_usd * asm.loss_severity AS DOUBLE) AS expected_loss_usd,
    p.min_elev,
    p.slope_deg,
    p.dist_to_water_m,
    p.label_real AS inside_historical_flood
  FROM bld
  JOIN {ns}.gold_h3_flood_predictions p
    ON p.aoi_name = bld.aoi_name AND p.h3 = bld.h3
  JOIN asm
    ON asm.building_class =
       CASE WHEN bld.residential = 1 THEN 'residential' ELSE 'commercial' END
""")

write_aoi_partition(exposure_df, f"{ns}.gold_building_exposure",
                    extra_partitions=("scenario_24h_mm",),
                    options={"delta.columnMapping.mode": "name"})

_exposure_rows = (
    spark.table(f"{ns}.gold_building_exposure")
         .where(F.col("aoi_name") == aoi_name)
         .count()
)
print(f"gold_building_exposure rows for {aoi_name}: {_exposure_rows:,}")

# Genie-facing column comments. Genie's auto-generated table summary uses these
# verbatim, so spelling them out here is the single highest-leverage thing we
# can do to improve NL->SQL accuracy without touching the space instructions.
for col, comment in [
    ("aoi_name",                "Area of interest slug, e.g. 'greater_montreal' or 'manhattan'. Partition key."),
    ("scenario_24h_mm",         "24-hour rainfall scenario in millimetres. Partition key. Discrete values: 10, 30, 60, 100, 150, 200."),
    ("osm_id",                  "OpenStreetMap feature id of the building, e.g. 'way/123456'. Primary id within an (aoi, scenario)."),
    ("building_type",           "OSM building tag value, e.g. 'detached', 'apartments', 'commercial', 'industrial', 'yes', 'unknown'."),
    ("residential",             "1 if the OSM building tag is in the residential allowlist (house, apartments, detached, etc.), 0 otherwise."),
    ("h3",                      "H3 cell id at resolution 9 (~174 m edge) covering the building centroid. BIGINT."),
    ("lon",                     "Building centroid longitude in EPSG:4326."),
    ("lat",                     "Building centroid latitude in EPSG:4326."),
    ("flood_prob",              "Model-predicted flood probability for this building's H3 cell at this rainfall scenario. 0..1."),
    ("risk_tier",               "Discrete risk band: 'low' (<0.1), 'moderate' (0.1-0.3), 'high' (0.3-0.6), 'severe' (>=0.6)."),
    ("brc_usd",                 "Building replacement cost in USD. Demo constants from gold_underwriting_assumptions."),
    ("loss_severity",           "Fraction of BRC paid out at this flood depth. Flat 0.25 demo assumption."),
    ("expected_loss_usd",       "flood_prob * brc_usd * loss_severity. The underwriter's primary exposure metric."),
    ("min_elev",                "Cell-level minimum elevation in metres. Joined from gold_h3_flood_predictions for explainability."),
    ("slope_deg",               "Cell-level slope in degrees."),
    ("dist_to_water_m",         "Cell centroid distance to nearest OSM water feature, in metres."),
    ("inside_historical_flood", "1 if this building's H3 cell intersects a 2017 or 2019 historical flood polygon, 0 otherwise."),
]:
    spark.sql(
        f"ALTER TABLE {ns}.gold_building_exposure ALTER COLUMN {col} "
        f"COMMENT '{comment.replace(chr(39), chr(39) * 2)}'"
    )

spark.sql(
    f"COMMENT ON TABLE {ns}.gold_building_exposure IS "
    f"'One row per (aoi, scenario, building). Joins OSM building centroids "
    f"to the H3 flood-prediction grid and to underwriting assumptions to "
    f"produce a single-SELECT-friendly view of building-level expected loss. "
    f"This is the Genie Space `flood_underwriter` data surface.'"
)

# COMMAND ----------

display(spark.sql(f"""
  SELECT scenario_24h_mm, risk_tier,
         COUNT(*)                                AS buildings,
         SUM(CASE WHEN residential = 1 THEN 1 ELSE 0 END) AS residential,
         ROUND(SUM(expected_loss_usd), 0)        AS total_expected_loss_usd
  FROM {ns}.gold_building_exposure
  WHERE aoi_name = '{aoi_name}'
  GROUP BY scenario_24h_mm, risk_tier
  ORDER BY scenario_24h_mm, risk_tier
"""))
