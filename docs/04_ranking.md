# Part C: ranking, contracts, and merge checks

## Inspected branch snapshots

Read independently of `milly/part`; no teammate branch was merged or modified.

| Snapshot | Observed interface |
| --- | --- |
| `main` at `b89a5f5ad20f1eae14a0039f3393b6375e7c9a20` | `01_clean_facilities.ipynb` writes `workspace.datathon.facilities_clean`. Uses `unique_id`, `name`, `city_clean`, `state_clean`, `*_list` evidence arrays, `location_confidence` = high/medium/low, quality flags and `quality_notes`. |
| `service` at `6b4104bef87409e4fc000cc69a7e0e7905f7134b` | `03_service_evidence.ipynb` writes `workspace.datathon.facility_service_evidence`. `config/services.yaml` currently defines `hemodialysis`, `ct_scan`, `mri`. |
| `ann/data` at `ff123e8f45f31009f55195502b29de892228a1e3` | Legacy `ann_facilities_clean_v1` with `facility_id`, `name_clean`, `*_items`, coordinate/duplicate flags. No location-confidence output. |

The original ranking commit used the legacy contract. The updated implementation
uses the current clean table by default, while retaining an explicit legacy
schema adapter. It never imports or reruns the cleaner or evidence notebook.

## Merge readiness: upstream blocker remains

The current evidence notebook prefers `facilities_clean`, then requires raw
`procedure`, `equipment`, `capability` columns. The current cleaner's
`OUTPUT_COLUMNS` removes those raw columns and outputs `procedure_list`,
`equipment_list`, `capability_list` instead. Consequently B's cell 5 raises
`ValueError: Expected procedure, equipment and capability in the facilities input.`
when it reads the new clean table. The configured fallback is not attempted
once the preferred table exists.

This is an upstream producer mismatch, not a ranking exception to suppress.
No changes were made to B's file or branch. B can adapt its local DataFrame
immediately after `clean = spark.table(source)`, before checking/scoring fields:

```python
# Suggested change for the service owner, not applied to their branch here.
for field in ["procedure", "equipment", "capability", "specialties"]:
    if field not in clean.columns and f"{field}_list" in clean.columns:
        clean = clean.withColumn(field, F.col(f"{field}_list"))
```

The scorer already accepts lists via its JSON struct serialization. This aliases
columns in memory and need not rewrite the clean table. After fixing the input,
B must regenerate evidence from the same cleaning snapshot used by ranking.
Old evidence may quote text discarded by the new cleaner and will be excluded
when the literal source text no longer matches.

**Do not call the whole pipeline ready until the producer fix, real-data checks
and App integration are complete.** Local Git mergeability does not prove that.

## Confirmed B output contract

| Column | Observed type / meaning |
| --- | --- |
| `unique_id` | String, facility ID; ranking trims surrounding whitespace on both sides. |
| `service` | Currently `hemodialysis`, `ct_scan`, `mri`. No broad clinical aliases inferred. |
| `evidence_level` | strong / moderate / weak. |
| `evidence_score` | Double rule weight: strong 0.85/0.90, moderate 0.55, weak 0.20. Not probabilities. |
| `quotes` | `array<struct<field:string,sentence:string>>`; ranking returns `{field,text}` preserving the sentence. `{field,text}` and JSON-encoded arrays are also accepted. |
| `n_direct`, `n_supporting` | Integer counts of distinct evidence snippets. |

B omits no-hit and vetoed facility/service pairs, and groups source records by
ID. Ranking requires unique facility IDs and facility/service keys rather than
silently amplifying or resolving duplicate joins. Specialty-only evidence is
weak. B currently does not define emergency C-section, NICU or burn care;
those services cannot be claimed as live supported features.

## Running

Open `04_ranking.ipynb` in the repository Git folder. `ranking.py` and `examples/`
must be alongside it. Default mode uses the synthetic **current-schema** fixture
`examples/04_ranking_current_sample.json`; the older fixture remains for legacy
adapter regression tests. No fixture is a real facility or data export.

```python
from ranking import configure, load_from_spark, recommend
configure(load_from_spark(spark))
result = recommend("hemodialysis", 25.5941, 85.1376, k=5)
```

Enable `RUN_LIVE = True` only after the upstream producer fix and schema checks.
Reload with `configure(load_from_spark(spark))` after table refreshes. For a
reviewed legacy table, explicitly pass its table name as `facility_table`;
there is no silent table fallback. The reader never mutates tables. It collects
selected columns into an in-memory snapshot with a 200,000-row limit per table
and fails rather than truncating. No LLM or dependency installation is required
by the ranking core.

## Decision rules and limitations

`score = evidence_score * location_weight * exp(-distance_km / decay_km)`

- Current cleaner location labels map to high=0.90, medium=0.60, low=0.30.
  These are explicit prototype weights, not calibrated probabilities. Unknown
  labels receive 0.30 and a warning. Duplicate identity warnings cap the weight
  at 0.40. Unknown quality flags reduce the weight.
- Exclude `flag_row_corrupted=True` or unknown corruption status in current
  rows. Exclude unusable coordinates and missing names. The current cleaner
  flags >150 km city outliers as low confidence; usable low-confidence points
  remain visible with a low badge and their quality notes.
- Current location checks are same-city medians and PIN-prefix/state heuristics,
  not independent city boundaries or PIN geocoding. The India rectangle is only
  a gross coordinate screen. Preserve `quality_notes` for the App.
- Legacy input has no trust label; its provisional proxy is 0.60, reduced to
  0.40 for duplicate or unknown flags. It is labelled `unverified`.
- Routine services use nearby radius 60 km and decay scale 75 km. Reserved
  urgent policies use 30/25 km, but B does not yet produce those services.
  These parameters are prototype choices, not clinical access standards.
- Distance is great-circle distance, not road distance or travel time. Display
  location resolver precision/ambiguity separately when using city/PIN centroids.
- Strong labels require a direct hit and more than specialty-only quotes.
  Unsupported strong labels are downgraded, with original level and score exposed.
- Quotes must occur literally in available original fields, legacy `*_items`,
  or current `*_list` fields. Mismatches exclude the pair; absent original fields
  produce a warning and low confidence. No generated replacement citation.
- Overall confidence is at most moderate because identity and service availability
  remain independently unverified; weak evidence or low location weight is low.
- Nearby ranking uses descending unrounded score, then distance, then ID.
  Never use phone numbers as identity or trust signals. Verify contacts independently.

## App response

`recommendations` contains up to k nearby records in score order.
`nearest_strong_option` is the closest usable strong textual-evidence record by
straight-line distance, separately from k. It can overlap the main list; deduplicate
cards by ID. `nearby_weak_options` and `nearby_lower_evidence_options` explicitly
expose fallback groups, including the k=1 case. Empty groups remain empty.

Every candidate includes location, score factors, location label, overall
confidence, English explanation/selection reason, literal quotes, source URLs
and warnings. Keep `status`, `message`, `policy` and exclusion counts. For
`no_nearby_strong_evidence`, display a separate nearest-strong section and nearby
lower-evidence section. Dataset absence is not a demonstrated real-world care gap.

## Verification completed locally

`python3 -m unittest discover -s tests -v`: **27 tests passed**. Updated default
notebook code cells execute without Spark, with live mode disabled. Coverage
includes current and legacy schema adapters, location label weights, corruption
exclusion, preserved B sentence quotes, source grounding, k=1 fallback, urgent
policy mechanics, duplicate keys, stable ranking and empty/error states.

Also executed the actual current cleaner's pure pandas function and B's actual
pure scorer/config on synthetic raw records, then passed their outputs through
ranking: two eligible candidates and one bad-coordinate exclusion, as expected.
This exercises the record adapters; it bypasses B's broken Spark input selection
and does not establish that the notebook-to-table pipeline runs.

## Mandatory Databricks checks — actual results PENDING

No Databricks table access or App session is available in this local repository.
Do not treat local software tests as real-data or clinical validation.

1. Confirm tables/schema and evidence regeneration after B's input fix:

```sql
SHOW TABLES IN workspace.datathon;
DESCRIBE TABLE workspace.datathon.facilities_clean;
DESCRIBE TABLE workspace.datathon.facility_service_evidence;
SELECT location_confidence, flag_row_corrupted, COUNT(*) AS n
FROM workspace.datathon.facilities_clean GROUP BY 1, 2;
SELECT service, evidence_level, COUNT(*) AS n,
       MIN(evidence_score) AS min_score, MAX(evidence_score) AS max_score
FROM workspace.datathon.facility_service_evidence GROUP BY 1, 2 ORDER BY 1, 2;
```

Expected: confirmed current schemas, all three canonical service IDs, valid
location labels/flags. Record actual counts; the supplied profile's 10,088 rows
is not a substitute for querying the current source snapshot.

2. Validate usable IDs and evidence key uniqueness:

```sql
SELECT TRIM(unique_id) AS id, COUNT(*) AS n
FROM workspace.datathon.facilities_clean
WHERE unique_id IS NOT NULL AND TRIM(unique_id) <> ''
GROUP BY 1 HAVING COUNT(*) > 1;
SELECT TRIM(unique_id) AS id, service, COUNT(*) AS n
FROM workspace.datathon.facility_service_evidence
GROUP BY 1, 2 HAVING COUNT(*) > 1;
SELECT e.unique_id, e.service
FROM workspace.datathon.facility_service_evidence e
LEFT ANTI JOIN workspace.datathon.facilities_clean f
ON TRIM(e.unique_id) = TRIM(f.unique_id);
```

Expected: no duplicate usable IDs/keys, unmatched evidence ideally zero. If not,
investigate instead of guessing a duplicate winner. Ranking normalizes service
spaces/hyphens to underscores, so check normalized keys too if vocabulary changes.

3. Enable live mode and run the 10 queries in `examples/04_ranking_queries.json`.
They now cover only B's confirmed hemodialysis/CT/MRI services across city/rural
points. Rural coordinates are approximate test points, not resolved PINs.
Capture status, IDs, distance, confidence, factors, literal citations, exclusions
and elapsed time. No particular real hospital or outcome has been asserted in
advance. Unknown services or no usable candidates fail the live notebook checks.

4. Check product-of-factors equality, descending nearby score order, radius
membership, no corrupted-row recommendations, nearest-strong distance against
all eligible strong records, and separate fallback groups. Manually review one
strong, one weak/specialty-only, one duplicate-warning, one coordinate rejection
and one far-from-city case against full source text and location. Add queries
if the 10 points do not cover these cases.

5. Measure Free Edition load/query latency, confirm B's independent 50-record
review results, and test D's rendering of explanations, warnings and empty/fallback
states. Attach the actual live JSON report before calling the pipeline ready.
