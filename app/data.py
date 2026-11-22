"""Data access for the CareFinder app.

On Databricks Apps: reads the team tables through a SQL warehouse as the app's service
principal (DATABRICKS_WAREHOUSE_ID is injected from the app's SQL warehouse resource).
Locally: set APP_SAMPLE_DATA=1 to use a small, clearly synthetic dataset.
Column names match what ranking.Ranker expects for the current facilities_clean schema.
"""

from __future__ import annotations

import os

import pandas as pd

CATALOG_SCHEMA = os.getenv("APP_SCHEMA", "workspace.datathon")
FACILITIES_TABLE = f"{CATALOG_SCHEMA}.facilities_clean"
EVIDENCE_TABLE = f"{CATALOG_SCHEMA}.facility_service_evidence"
LOCATIONS_TABLE = f"{CATALOG_SCHEMA}.app_locations"

FACILITIES_SQL = f"""
SELECT CAST(unique_id AS STRING) AS unique_id, name, facility_type, operator_type,
       city_clean, state_clean, pin_code, latitude, longitude, location_confidence,
       officialWebsite, officialPhone, phone_shared_by_n, source_urls, description,
       to_json(procedure_list)   AS procedure_list,
       to_json(equipment_list)   AS equipment_list,
       to_json(capability_list)  AS capability_list,
       to_json(specialties_list) AS specialties_list,
       flag_row_corrupted, flag_coords_missing, flag_coords_outside_india, flag_coords_far_from_city,
       flag_pin_state_mismatch, flag_possible_duplicate, flag_phone_shared, flag_capability_bloat,
       quality_notes
FROM {FACILITIES_TABLE}
"""
EVIDENCE_SQL = f"""
SELECT CAST(unique_id AS STRING) AS unique_id, service, evidence_level, evidence_score,
       to_json(quotes) AS quotes, n_direct, n_supporting
FROM {EVIDENCE_TABLE}
"""
LOCATIONS_SQL = f"SELECT kind, name, state, lat, lon, n_points, source FROM {LOCATIONS_TABLE}"


def using_sample() -> bool:
    return os.getenv("APP_SAMPLE_DATA") == "1"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Return (facilities, evidence, locations-or-None, notes)."""
    if using_sample():
        from sample_data import build_sample
        fac, evi, loc = build_sample()
        return fac, evi, loc, ["Using SYNTHETIC sample data (APP_SAMPLE_DATA=1)."]

    from databricks import sql
    from databricks.sdk.core import Config

    cfg = Config()
    warehouse = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not set. Add a SQL warehouse resource "
                           "with key 'sql-warehouse' to the app (see app/README.md).")
    notes = []
    with sql.connect(server_hostname=cfg.host,
                     http_path=f"/sql/1.0/warehouses/{warehouse}",
                     credentials_provider=lambda: cfg.authenticate) as conn:
        with conn.cursor() as cur:
            def query(q: str) -> pd.DataFrame:
                cur.execute(q)
                return cur.fetchall_arrow().to_pandas()

            fac = query(FACILITIES_SQL)
            evi = query(EVIDENCE_SQL)
            try:
                loc = query(LOCATIONS_SQL)
            except Exception as e:  # optional table
                loc = None
                notes.append(f"{LOCATIONS_TABLE} not available ({type(e).__name__}); "
                             "PIN codes resolve from facility locations only.")
    return fac, evi, loc, notes
