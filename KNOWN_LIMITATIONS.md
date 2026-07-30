# Known Limitations

This is a reference architecture / solution accelerator, not a production flood
model. The list below captures the limitations we are aware of and have
explicitly chosen to live with for the demo. Each item points at the file(s)
where the trade-off is taken so a future contributor knows where to start if
they want to lift it.

## Data and modelling

### H3 cells over open water are not always filtered out

**Symptom.** Over wide rivers (e.g. the Hudson when the AOI is retargeted at
NYC, or the Saint-Laurent in the default Greater Montreal AOI) you can see
H3 hexagons rendered on top of the river surface in the app, even though
flooding semantics don't apply to a cell that is already 100% water.

**Root cause.** Two compounding issues in the silver pipeline:

1. **OSM multipolygon relations are dropped at ingest.**
   `notebooks/01_ingest.py` queries Overpass for both
   `way["natural"="water"]` and `relation["natural"="water"]`, but the
   feature-conversion loop only handles `el["type"] == "way"`. Wide rivers
   and large lakes are commonly modelled in OSM as **multipolygon
   relations**, so their polygon geometry never makes it into
   `bronze_hydrography`. Only `waterway=river` and `waterway=stream` lines
   (the river *centerlines*) survive, as `LineString` features.

2. **The "drop in-water cells" filter uses centroid-to-feature distance.**
   `notebooks/02_silver_geobrix.py` (section 6) drops cells where
   `dist_to_water_m < 5 m`, computed from the cell *centroid* to the nearest
   bronze water feature. With only river centerlines available, an H3 cell
   sitting in the middle of a 1 km wide river — but offset 200 m from the
   centerline — has `dist_to_water_m ≈ 200 m` and passes the filter, even
   though the entire hexagon is over water.

**Why we are leaving it.** Fixing this end-to-end requires (a) reassembling
multipolygon relations from Overpass `out geom;` member arrays in the ingest
step, and (b) replacing the centroid-distance mask with a hexagon-vs-water
*area-overlap* mask (build the H3 boundary polygon via
`h3.cell_to_boundary`, intersect with the unioned water layer, drop cells
where `intersection_area / hex_area > 0.5`). Both are tractable but add
non-trivial complexity to a pipeline whose primary purpose is to demonstrate
the GeoBrix + Spatial SQL + H3 + MLflow + Apps story end-to-end. The visual
artefact does not affect the reported precision/recall metrics, since the
real flood polygons used for validation also do not cover open river
surfaces.

**Workarounds for users who care.**

- Tighten the AOI bbox so it doesn't span large open-water bodies.
- Lower `WATER_DIST_THRESHOLD_M` in `02_silver_geobrix.py` from `5.0` to
  something larger (e.g. `100.0`). This is crude — it will also remove
  legitimately flood-prone riparian cells — but it visibly improves the
  app for screenshots.
- Implement the proper fix described above; the work is contained to
  `01_ingest.py` and the silver "drop in-water cells" section.

### Building footprints are synthesized from centroids

**Symptom.** The `/api/buildings_at_risk` endpoint and the "Building exposure
(underwriting)" map layer in the app render every building as a small ~12 m
square around its OSM centroid, not as the building's real footprint
polygon. At city-block zoom this reads as individual buildings, but it is not
geometrically accurate, and very large structures (industrial, multi-block
mall) are drawn the same size as a single-family home.

**Root cause.** `notebooks/01_ingest.py` queries Overpass with
`out center tags`, which ships only the centroid + tags for each building
way/relation (~30 MB for Greater Montreal, ~300k features). Pulling the full
polygon geometry with `out geom` is closer to ~300+ MB and triggers Overpass
rate-limit / timeout for the default AOI on the free tier. The app therefore
joins `bronze_buildings` (point) to `gold_h3_flood_predictions` by H3 cell
and synthesizes a square around each centroid at request time.

**Why we are leaving it.** Building-level footprints don't change the
exposure math (`expected_loss = flood_prob * BRC * loss_severity`) — that's
already accurate because it keys off the cell-level `flood_prob` and a
per-building `residential` flag from OSM tags. The polygon shape is purely
a visual upgrade.

**Path to real polygons.** The app endpoint already auto-upgrades when a
`silver_building_footprints` table exists in the schema. To produce it:

1. In `notebooks/01_ingest.py`, replace `out center tags` with
   `out geom tags` for the buildings Overpass query, page the response by
   sub-bbox if Overpass times out, and write a new `bronze_building_footprints`
   table with `(aoi_name, osm_id, building, residential, lon, lat,
   geometry_geojson)`.
2. In `notebooks/02_silver_geobrix.py`, add a step that reads
   `bronze_building_footprints`, computes the H3 cell of each centroid with
   `h3.latlng_to_cell(lat, lon, h3_res)`, and writes
   `silver_building_footprints` with the same columns plus `h3`.
3. Redeploy the app. The `_has_footprints_table()` lru_cache picks the new
   table up on next process start and switches `footprint_source` from
   `"synthesized"` to `"real"` in the response.

### Underwriter chatbot can't explain how the model works

**Symptom.** Ask the chat panel "how is `flood_prob` computed?" or "why is
this building in the severe tier?" and it will either refuse or hallucinate.

**Root cause.** The chatbot is a **pure Genie Space** — natural language in,
SQL out, no document retrieval. It only knows what's queryable from the four
gold tables. Model internals (the RandomForest, the feature engineering, the
hybrid label) are not in those tables; they live in `BLOG.md` and the
notebooks.

**Why we're leaving it.** Genie is purpose-built for the underwriter
question class (aggregates, slicing, top-N, scenario stress). Mixing
doc retrieval in requires a Knowledge Assistant alongside Genie, orchestrated
either by a Multi-Agent Supervisor or a custom Agent Framework agent —
roughly 4-5x the build effort for a question class the underwriter persona
doesn't actually need.

**Upgrade path.** Add a Knowledge Assistant pointed at `README.md`,
`BLOG.md`, and `KNOWN_LIMITATIONS.md`; wire both into an Agent Bricks
Multi-Agent Supervisor; have the supervisor route data questions to Genie
and explanation questions to the KA. See [resources/genie/flood_underwriter.json](resources/genie/flood_underwriter.json)
"general_instructions" — the space already tells users to consult the README
for these questions.

### Underwriter chatbot can't drive the map

**Symptom.** Ask "zoom to this address" or "set the rainfall slider to
100 mm" and the chatbot will (correctly) tell you it can't do that.

**Root cause.** Genie's contract is SQL only — no tool calls, no UI
mutations. The map controls already live in the side panel and the address
search box already covers "is my address at risk" with single-building
precision.

**Upgrade path.** If map-action questions become important, wrap the
existing `/api/lookup` and the FloodMap setters as tools on a Mosaic AI
Agent Framework agent, and replace `/api/chat/*` with that agent's
endpoint. The Genie space becomes one of that agent's tools (via the
`ask_genie` action) so SQL questions still work.

### Underwriter cost numbers are flat constants, not policy data

**Symptom.** The chatbot's expected-loss math (and the buildings-at-risk
overlay's tooltip $$) uses the same numbers for every residential building
and the same numbers for every commercial building, regardless of
square-footage, occupancy class, construction type, year built, or insured
value. Real underwriters never accept a portfolio-level number computed this
way.

**Root cause.** `gold_underwriting_assumptions` ships two rows
(residential = $300K BRC, commercial = $1.5M BRC, 25% flat loss severity)
as demo placeholders. There's no policy book, no replacement-cost estimator,
and no depth-damage curve in the pipeline. This is the same caveat as the
buildings-at-risk overlay constants, just exposed to a chatbot now.

**Upgrade path.** Replace `gold_underwriting_assumptions` with a real
source, in roughly this order of fidelity:
1. The carrier's policy book joined on `osm_id ↔ policy_id` via a
   property-matching service. Per-policy `insured_value`, `coverage_a..d`,
   `deductible`, `policy_form` -> exact `expected_loss_usd`.
2. CoreLogic / Verisk / Cotality replacement-cost API keyed by address +
   sqft + construction type. Cached as a Delta table joined to
   `gold_building_exposure` by `osm_id`.
3. Public assessor rolls (Quebec rôle d'évaluation foncière, NYC PLUTO).
   Cheaper, less accurate; good for sandbox demos.
4. Replace the flat 25% severity with FEMA HAZUS depth-damage curves keyed
   by occupancy class and a hydraulic-model depth-of-flood per cell — but
   that also requires a depth model the current pipeline doesn't have.

Until that swap happens, `expected_loss_usd` should be read as a *demo
signal*, not a portfolio reserve number. The chatbot's space instructions
already say "treat the dollar values as demo assumptions, not policy data".

### Hybrid labels mix synthetic and real

The model trains on a rainfall-aware **synthetic susceptibility label**, with
the real 2017/2019 flood polygons held out for validation only. This is a
deliberate demo pattern (see `BLOG.md` → "Hybrid labels: honest about what is
synthetic"), not a bug, but it means the absolute probability values from
the model are not calibrated against real flood frequencies. Production use
would substitute insurance claims, expanded historical events, or radar QPE
for the synthetic signal.

### SRTM 30 m DEM, not HRDEM 1 m

Elevation comes from the global SRTM 1-arc-second tiles on AWS Open Data,
which is roughly 30 m horizontal resolution. Canada's HRDEM is 1 m where it
exists, which would dramatically sharpen slope and low-point detection in
urban areas. The pipeline is tile-agnostic — swapping the DEM source only
changes `notebooks/01_ingest.py`.

### Precipitation is climatology, not forecast

The rainfall-scenario slider blends the model with **historical climatology**
(Open-Meteo ERA5 P99 24h / 5d aggregated to a 0.2° grid), not a live
forecast. There is no streaming ingestion of HRDPS / GFS / ECMWF forecasts.
For real-time alerting you would point silver step 5 at a live source.

### Slope is approximated, not derived from a raster kernel

Per-cell slope is computed as
`atan(max(|elev_diff_to_neighbour|) / 174 m)` over the H3 res-9 1-ring
(`02_silver_geobrix.py`, section 3). This is robust across GeoBrix versions
but coarser than a true raster slope kernel. For most of Greater Montreal
(low relief) the difference is negligible.

### Distance-to-water is computed on the driver

The `silver_h3_dist_water` step pulls all bronze water geometries to the
driver, builds an STRtree, and iterates H3 cells in Python. This is fast
enough for a Montreal-sized AOI (~50 km × 55 km, low tens of thousands of
cells) but is not how you would do this at province / country scale —
you'd push it down to Spatial SQL with `ST_Distance` after a proper SRID
fix-up, or use `gbx_st_*` functions if your GeoBrix build exposes them
consistently.

## Operational

### Free-tier API rate limits

Both Overpass (OSM) and Open-Meteo ERA5 are free, unauthenticated services
with rate limits. The ingest notebook caches everything to the AOI volume
on first run, so re-runs are fast, but the *first* run for a new AOI can
take several minutes and will fail loudly if the upstream service is
temporarily unavailable. Expand the retry budget in `fetch_overpass` and
`_fetch_open_meteo_daily` if you hit transient 429s.

### MELCC flood polygons are Quebec-only

`flood_source="melcc"` (the default) only returns features for AOIs that
intersect the Quebec MDDELCC ZIS extents. Retargeting the demo at a city
outside Quebec requires switching to `flood_source="cems"` (Copernicus EMS
Rapid Mapping, global, but you must supply activation URLs) or
`flood_source="none"` (no real-flood overlay; the model still trains on
synthetic labels). See the dispatch block in `01_ingest.py` for details.

### GeoBrix version drift

`02_silver_geobrix.py` deliberately calls `DESCRIBE FUNCTION EXTENDED` on
the GeoBrix RasterX functions it relies on at the top of the notebook so
the job log captures the exact signatures of the installed build.
Function signatures (especially `rst_clip`, `rst_mapalgebra`,
`rst_derivedband`) have moved across GeoBrix versions; if a future build
breaks the pipeline, the job log makes the diagnosis a one-line read.
