# Part C: ranking and explanations

## Repository evidence inspected

Read independently of `milly/part`; no teammate branch was merged or copied.

| Branch snapshot | File | Confirmed interface |
| --- | --- | --- |
| `ann/data` at `ff123e8f45f31009f55195502b29de892228a1e3` | `01_clean_facilities.ipynb`, cells 3–8 | Writes `workspace.datathon.ann_facilities_clean_v1`; retains raw columns and adds cleaned columns and boolean flags. |
| `service` at `b40757d44d24deb8e51996756b2e385baddce1f2` | Entire tracked tree | Contains `00_config.ipynb` and the same cleaning notebook. No service-evidence producer or services configuration exists in this snapshot. |
| `milly/part` initial `9f77ed884cb9c910b409b5b1557ff8a75d9a4128` | `00_config.ipynb` | `OUT = workspace.datathon`; does not include the cleaning notebook. |

ANN's confirmed columns are `facility_id`, `name_clean`, `city_clean`,
`state_clean`, `postcode_clean`, `latitude`, `longitude`, `procedure_items`,
`equipment_items`, `capability_items`, and raw fields including `unique_id`,
`procedure`, `equipment`, `capability`, `description`, `specialties`, `source_urls`.
The cleaner adds coordinate, missing-field, parsing, and duplicate flags.
It does **not** implement numeric coordinate confidence, PIN cross-validation,
city-coordinate consistency, an India polygon check, or a location resolver.
Its write uses CREATE TABLE without IF NOT EXISTS; do not rerun it blindly.

## Service integration is pending

The target `workspace.datathon.facility_service_evidence` and the following
schema come from the team's agreement, **not** an observed producer. Before live
use, teammate B must confirm the table name, service vocabulary and types:

| Column | Provisional type / meaning |
| --- | --- |
| `unique_id` | String; matches cleaned `facility_id` after surrounding whitespace is trimmed. |
| `service` | String; one exact capability per row. Spaces/hyphens normalize to underscores, but no clinical aliases are inferred. |
| `evidence_level` | `strong`, `moderate`, `weak`. |
| `evidence_score` | Finite number in (0, 1] for ranked candidates. Zero-score/no-claim records are excluded. |
| `quotes` | Array of `{field, text}` or `{field, sentence}`; a JSON string encoding this array is also accepted. |
| `n_direct`, `n_supporting` | Nonnegative integer counts. |

Quote fields accepted: `description`, `procedure`, `equipment`, `capability`,
`specialties`. Quote text must be a literal substring of the original field
when available. Missing original fields produce a warning and low confidence;
contradictory quote text excludes the row. Unstructured quote strings are
excluded with a diagnostic, not silently treated as verified citations.

One row per facility/service is required. Duplicate IDs or evidence keys fail
with an actionable error. Do not join the raw PIN directory directly: its grain
is post office and it can multiply facilities. Phone numbers are not used for
identity, ranking, or contact recommendations.

## Running

Open `04_ranking.ipynb` in the repository folder in Databricks. The default is
the synthetic demo, with no table access. Live mode is opt-in in the notebook;
it requires R's table and B's confirmed table to exist, independently of whether
their code has been merged. The notebook never writes or replaces input tables.

```python
from ranking import configure, load_from_spark, recommend
configure(load_from_spark(spark))
result = recommend("dialysis", 25.5941, 85.1376, k=5)
```

Call `configure(load_from_spark(spark))` again after upstream tables change.
The prototype collects selected columns into a small in-memory snapshot and
fails above 200,000 rows per table rather than silently truncating. Measure
latency/memory in Free Edition before the demo. No LLM, API call or dependency
installation is required by the ranking core.

## Decision rules

`score = evidence_score * location_weight * exp(-distance_km / decay_km)`

- For routine services: nearby radius 60 km, decay scale 75 km.
- For emergency C-section, NICU, trauma, burn care, emergency and ICU: radius
  30 km, decay scale 25 km. These are configurable prototype choices, not
  clinically validated access standards.
- Distance is great-circle distance, not road distance or travel time. A query
  at a city/PIN centroid inherits that location's uncertainty; the App should
  display the resolver's precision and ambiguity separately.
- Missing, invalid or outside-ANN-rectangle facility coordinates are excluded.
  ANN's rectangle is latitude 6–38 / longitude 68–98 and cannot prove a facility
  lies in India or the claimed city. Valid-looking wrong-city coordinates can
  still pass. Do not describe them as verified.
- ANN v1 provides no numeric location confidence. The explicit temporary
  `location_weight` is 0.60 after those screens; possible duplicate name/city,
  duplicate cluster, or unknown quality flags reduce it to 0.40. These are
  heuristic weights. They combine limited identity/location quality signals
  and must be replaced once R supplies a reviewed coordinate confidence output.
- A strong label without a direct hit or with specialty-only citations is
  downgraded to weak. B's numeric score is preserved and exposed; the downgrade
  is visible. Thus a very close weak claim can rank above a distant strong one,
  but it is always labelled weak. Confirm B's count semantics and thresholds.
- Overall confidence is at most `moderate` with the current location proxy,
  and `low` for weak evidence, duplicate warnings or unavailable quote sources.
  It is a rule-based badge, not a probability of actual service availability.
- Order is descending unrounded score, then distance, then ID for stable ties.

No score establishes that a facility exists independently, that high-risk
delivery is supported, or that staff/equipment are currently available.
Emergency C-section is not inferred from maternity/obstetrics alone. Likewise
NICU, ICU, cardiac surgery, and cardiology must have separate service evidence.

## App response contract

- `recommendations`: up to k nearby candidates in score order.
- `nearest_strong_option`: nearest eligible strong-evidence record by distance,
  independent of k and score. May be far away or null. Can overlap the main
  list; the App should deduplicate cards by ID.
- `nearby_weak_options`: up to k nearby weak-evidence records.
- `nearby_lower_evidence_options`: nearby moderate/weak alternatives.
- `status`, `message`, `policy`, exclusion counts and candidate counts must be
  retained. An unknown service, failed data screens, or lack of dataset evidence
  must not be labelled a demonstrated real-world absence of care.
- Each candidate includes coordinates, score components, evidence level,
  overall confidence, literal quotes, source URLs, English explanation and
  warnings. Round only for display.

When `status == no_nearby_strong_evidence`, render two labelled sections:
**Nearest strong textual evidence (outside nearby radius)** and
**Nearby lower textual evidence — verification required**. A farther strong
option must not be described as an emergency routing instruction. If no nearby
options or no usable strong option exists, explicitly show that empty state.

## Local verification

```sh
python3 -m unittest discover -s tests -v
```

The fixture in `examples/04_ranking_sample.json` is entirely synthetic and is
not a data export. For query `dialysis, 25.5941, 85.1376, k=1`, expected:
`no_nearby_strong_evidence`, `sample-weak` nearby, `sample-strong` separately,
and one excluded bad-coordinate record. Tests cover both groups with k=1,
nearest-vs-highest-score behavior, evidence/location/distance factors, urgent
decay, duplicate join failures, quote grounding, unknown services and empty
states. This is software verification, not medical or source-data validation.

Local results: 22 unittest cases passed, and all default notebook code cells
executed successfully without Spark. The sample returned a nearby weak option
at 0.699 km (score 0.208052) and the separate strong option at 89.829 km
(score 0.163016), with one bad-coordinate exclusion. Live cells remained off.

## Mandatory Databricks checks — actual results PENDING

Only repository code and synthetic records were available locally. None of
the following results have been claimed as observed on Databricks.

1. Confirm table existence and exact schema before enabling live ranking:

```sql
SHOW TABLES IN workspace.datathon;
DESCRIBE TABLE workspace.datathon.ann_facilities_clean_v1;
-- Run only after B confirms the table exists:
DESCRIBE TABLE workspace.datathon.facility_service_evidence;
```

Expected: ANN fields match the inspected notebook. B's schema must either match
the provisional contract or receive an explicit adapter; do not guess aliases.

2. Check facility cardinality and missing IDs. Record the counts, not just PASS:

```sql
SELECT COUNT(*) AS rows,
       COUNT(DISTINCT facility_id) AS distinct_ids,
       SUM(CASE WHEN facility_id IS NULL OR TRIM(facility_id) = ''
                THEN 1 ELSE 0 END) AS missing_ids
FROM workspace.datathon.ann_facilities_clean_v1;

SELECT facility_id, COUNT(*) AS n
FROM workspace.datathon.ann_facilities_clean_v1
WHERE facility_id IS NOT NULL AND TRIM(facility_id) <> ''
GROUP BY facility_id HAVING COUNT(*) > 1;
```

Expected: no duplicate usable IDs. Missing IDs are excluded. Compare total rows
to the source snapshot; the supplied profile reports 10,088, but the current
Databricks count must be measured.

3. Check B's service vocabulary, ranges and uniqueness:

```sql
SELECT service, evidence_level, COUNT(*) AS n,
       MIN(evidence_score) AS min_score, MAX(evidence_score) AS max_score
FROM workspace.datathon.facility_service_evidence
GROUP BY service, evidence_level ORDER BY service, evidence_level;

SELECT TRIM(CAST(unique_id AS STRING)) AS id,
       REGEXP_REPLACE(LOWER(TRIM(service)), '[ -]+', '_') AS normalized_service,
       COUNT(*) AS n
FROM workspace.datathon.facility_service_evidence
GROUP BY 1, 2 HAVING COUNT(*) > 1;

SELECT e.unique_id, e.service
FROM workspace.datathon.facility_service_evidence e
LEFT ANTI JOIN workspace.datathon.ann_facilities_clean_v1 f
ON TRIM(CAST(e.unique_id AS STRING)) = TRIM(f.facility_id);
```

Expected: canonical services distinguish the requested capabilities; duplicate
keys zero; unmatched IDs ideally zero, otherwise investigate and record the
excluded count. Scores finite in [0,1]; quoted positive evidence and integer
counts conform to the contract. Python performs further NaN/count checks.

4. Run the live notebook's 10 queries from `examples/04_ranking_queries.json`.
The rural points are approximate test coordinates, not resolved PINs. Record
for each query: status, returned IDs, distances, nearest strong ID/distance,
score components, quotes, exclusions and elapsed time. No particular hospital
name or status can be specified in advance without querying the real tables.
Expected invariants: exact product of factors; stable score ordering; every
nearby result within its configured radius; nearest strong really nearest;
no duplicate IDs; literal source quotes; separate fallback groups when needed.

5. Manually audit at least: one nearby strong case, one fallback case, one
specialty-only weak case, one duplicate warning, one rejected-coordinate case,
and one apparent long-distance/location contradiction. Inspect full source
records and compare coordinates with city/PIN. If one of these cases is absent
from the 10 queries, find an additional real case instead of claiming coverage.

6. Share B's 50-record human evaluation results with the demo team. Ranking
tests do not estimate evidence extraction accuracy. Record Free Edition load
and query latency. The notebook prints a full JSON report; attach actual
Databricks results to the submission after running it.
