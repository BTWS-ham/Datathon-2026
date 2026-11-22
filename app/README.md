# CareFinder app (Databricks App, Streamlit)

Non-technical users pick a **service** and type a **place** (city/district, PIN code or coordinates).
The app returns ranked facilities. Each card shows the facility's own words as evidence, the evidence level,
a confidence badge, data-quality warnings, sources and the exact score calculation.
Facilities with only a specialty tag (weak evidence) are folded into a separate section.

## How it fits the pipeline
| Step | Producer | Used by the app as |
|---|---|---|
| Cleaning | `01_clean_facilities` → `facilities_clean` | facility records, quality flags, location confidence |
| Service evidence | `03_service_evidence` → `facility_service_evidence` | evidence level, score, quotes |
| Ranking | `ranking.py` (`04_ranking`) | **all scores, exclusions and warnings** (the app does not re-implement ranking) |
| Locations (optional) | `05_app_locations` → `app_locations` | PIN codes / districts without facilities |

## Files
| File | What it does |
|---|---|
| `app.py` | Streamlit UI: Find care / How we decide / Data quality |
| `search.py` | Builds `ranking.Ranker` from the tables, folds weak evidence, adds display-only fields |
| `locations.py` | Resolves city, district, PIN code or coordinates to a query point |
| `llm_review.py` | AI second opinion on shown results (verbatim quote required) and the rule-vs-AI policy |
| `adjust.py` | Re-ranks the shown results with road distances and after the AI review, with the same formula |
| `routing.py` | Driving distances from the search point via OSRM (OpenStreetMap), with straight-line fallback |
| `data.py` | Reads the tables through a SQL warehouse; `APP_SAMPLE_DATA=1` uses synthetic data |
| `sample_data.py` | Synthetic records for local development and tests (not real facilities) |
| `tests/test_app.py` | Headless UI and integration tests |

`app.yaml` and `requirements.txt` live at the **repository root**, because the app imports `ranking.py` from there.

## Deploy on Databricks Free Edition
1. Run `01_clean_facilities`, `03_service_evidence` and (optionally) `05_app_locations`.
2. **Compute → Apps → Create app → Custom**, e.g. `carefinder`.
3. Add resources: **SQL warehouse** (Serverless Starter Warehouse), permission *Can use*, key **`sql-warehouse`**; and **Serving endpoint** `databricks-meta-llama-3-3-70b-instruct`, permission *Can query* (for the AI review).
4. Grant the app's service principal read access (SQL at the end of `05_app_locations`).
5. Deploy with the source path set to the **repository root** of the Git folder (where `app.yaml` is).

## Run locally
```sh
# from the repository root
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
APP_SAMPLE_DATA=1 APP_LLM=off streamlit run app/app.py   # synthetic data, no AI
APP_SAMPLE_DATA=1 streamlit run app/app.py                # synthetic data + real AI review (needs Databricks auth)
python app/tests/test_app.py                              # tests (use a fake LLM)
```

## AI review and re-ranking
For each result shown (plus a few extra candidates and the folded weak ones), a language model reads the facility's
own text and returns a rating, one verbatim quote and a short reason. A quote not found in the listing discards the
review. When the AI and the rules agree, the card shows one line. When they disagree, the card shows both ratings
with their quotes and a suggestion (`llm_review.reconcile`):
- AI downgrade → accepted (it reads context such as "being set up" or "inpatients only");
- AI upgrade → accepted only if its quote says the service is available/offered, not just a machine or department;
- no AI quote → rule rating kept.

The list is then re-ranked (`adjust.py`) with the **same formula** as `ranking.py`, replacing only the evidence score by
the recommended rating's rule weight (strong 0.85, moderate 0.55, weak 0.20); *none* moves the facility to a
"Removed after the AI review" section. Every change (old rank → new rank) is listed above the results and on the card.
`APP_LLM=off` disables all of this and shows the pure `ranking.py` order. Reviews are cached per facility and service.
Pilot on 50 human-labelled pairs: LLM 44/50, rules 42/50 (labels were AI-assisted; the upgrade rule was derived from
the same sample).

## Road distances
Candidates are found within the straight-line radius of `ranking.py`. The app then asks OSRM (OpenStreetMap road
network, no live traffic) for the driving distance from the search point to each candidate in one request, and uses it
in the same distance factor `exp(-distance / decay)`. Cards show the road distance only; any change in order versus
straight-line ranking is listed (up / down / out of the list). If OSRM is unreachable, or has no route for a facility,
the road distance is **estimated** as 1.4 x straight-line distance (`routing.ROAD_FACTOR`), labelled "estimated", and the
straight-line distance is shown next to it, so every facility is still ranked on a road-distance basis.
- The public server `router.project-osrm.org` is for light use; set `OSRM_URL` to a self-hosted OSRM for real use.
- Only coordinates are sent (search point and facility locations).
- The deployed app needs outbound internet access; otherwise it falls back to estimated road distances.
- `APP_ROUTING=off` disables it; tests use `APP_ROUTING=stub`.
