"""Create missing ranking inputs using reviewed, pinned teammate logic.

Existing tables are never overwritten. This integration bridge changes neither
the main nor service branch. Source evidence lists are aliased in memory to the
field names expected by B's scorer.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from integration.upstream_cleaner import OUTPUT_COLUMNS, clean_facilities, is_missing
from integration.upstream_service import compile_rules, score_record

SOURCE_TABLE = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities"
CLEANER_COMMIT = "b89a5f5ad20f1eae14a0039f3393b6375e7c9a20"
SERVICE_COMMIT = "6b4104bef87409e4fc000cc69a7e0e7905f7134b"


def facility_records(frame):
    """Normalize pandas/numpy scalars with an explicit Spark output schema."""
    import numpy as np
    import pandas as pd

    doubles = {
        "latitude", "longitude", "km_from_city_centre", "numberDoctors",
        "capacity", "year_established", "completeness",
    }
    integers = {c for c in OUTPUT_COLUMNS if c.startswith("n_")} | {"phone_shared_by_n", "dup_count"}
    types = {
        c: "array<string>" if c.endswith("_list") else
        "boolean" if c.startswith("flag_") else
        "double" if c in doubles else "long" if c in integers else "string"
        for c in OUTPUT_COLUMNS
    }
    records = []
    for raw in frame.to_dict(orient="records"):
        record = {}
        for column, kind in types.items():
            value = raw[column]
            if kind == "array<string>":
                value = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
            elif value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)):
                value = None
            elif kind == "boolean":
                value = bool(value)
            elif kind == "double":
                value = float(value)
            elif kind == "long":
                value = int(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                value = json.dumps([str(v) for v in value], ensure_ascii=False)
            else:
                value = str(value)
            record[column] = value
        records.append(record)
    schema = ", ".join(f"`{column}` {kind}" for column, kind in types.items())
    return records, schema


def evidence_records(facilities, config=None):
    """Apply B's actual scoring rules after explicit *_list field adaptation."""
    if config is None:
        config = json.loads(Path(__file__).with_name("integration").joinpath("services_snapshot.json").read_text())
    documents = defaultdict(set)
    for facility in facilities:
        uid = facility.get("unique_id")
        if is_missing(uid):
            continue
        record = {}
        for field in ("procedure", "equipment", "capability", "description", "specialty", "specialties"):
            if field in facility:
                record[field] = facility[field]
            elif f"{field}_list" in facility:
                record[field] = facility[f"{field}_list"]
        documents[str(uid)].add(json.dumps(record, ensure_ascii=False, allow_nan=False))
    rules = compile_rules(config)
    result = []
    for uid, sources in sorted(documents.items()):
        for service, level, score, quotes, direct, supporting in score_record(sorted(sources), rules):
            result.append({
                "unique_id": uid, "service": service,
                "evidence_level": level, "evidence_score": score,
                "quotes": [{"field": field, "sentence": text} for field, text in quotes],
                "n_direct": direct, "n_supporting": supporting,
            })
    return result


def prepare_missing_inputs(spark, out="workspace.datathon", max_rows=200000):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", out):
        raise ValueError("out must be catalog.schema with simple identifiers")
    facility_table = f"{out}.facilities_clean"
    evidence_table = f"{out}.facility_service_evidence"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {out}")
    created = []
    if not spark.catalog.tableExists(facility_table):
        raw = spark.table(SOURCE_TABLE).limit(max_rows + 1).toPandas()
        if len(raw) > max_rows:
            raise ValueError("Raw input exceeds the prototype row limit")
        clean = clean_facilities(raw)[OUTPUT_COLUMNS].copy()
        if len(clean) != len(raw):
            raise ValueError("Upstream cleaner changed row count")
        records, schema = facility_records(clean)
        spark.createDataFrame(records, schema).write.format("delta").mode("error").saveAsTable(facility_table)
        created.append(facility_table)
    if not spark.catalog.tableExists(evidence_table):
        clean_rows = spark.table(facility_table).limit(max_rows + 1).collect()
        if len(clean_rows) > max_rows:
            raise ValueError("Clean input exceeds the prototype row limit")
        records = evidence_records([row.asDict(recursive=True) for row in clean_rows])
        schema = (
            "unique_id string, service string, evidence_level string, evidence_score double, "
            "quotes array<struct<field:string,sentence:string>>, n_direct int, n_supporting int"
        )
        spark.createDataFrame(records, schema).write.format("delta").mode("error").saveAsTable(evidence_table)
        created.append(evidence_table)
    return {
        "created": created,
        "facility_rows": spark.table(facility_table).count(),
        "evidence_rows": spark.table(evidence_table).count(),
        "cleaner_commit": CLEANER_COMMIT, "service_commit": SERVICE_COMMIT,
    }
