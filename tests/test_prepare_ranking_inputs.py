import json
import unittest
from pathlib import Path

import pandas as pd

from prepare_ranking_inputs import clean_facilities, evidence_records, facility_records, prepare_missing_inputs


class PreparationTests(unittest.TestCase):
    def test_current_lists_are_scored_with_original_field_names(self):
        sample = json.loads((Path(__file__).resolve().parents[1] / "examples/04_ranking_current_sample.json").read_text())
        evidence = evidence_records(sample["facilities"])
        strong = next(row for row in evidence if row["unique_id"] == "sample-strong")
        self.assertEqual(strong["service"], "hemodialysis")
        self.assertEqual(strong["evidence_level"], "strong")
        self.assertIn({"field": "procedure", "sentence": "Hemodialysis treatment is provided."}, strong["quotes"])

    def test_veto_and_equipment_only_semantics_are_preserved(self):
        rows = [
            {"unique_id": "veto", "procedure_list": ["Hemodialysis is not available."]},
            {"unique_id": "equipment", "equipment_list": ["Hemodialysis machine"]},
        ]
        evidence = evidence_records(rows)
        self.assertFalse(any(row["unique_id"] == "veto" for row in evidence))
        self.assertEqual(evidence[0]["evidence_level"], "moderate")
        self.assertEqual(evidence[0]["n_direct"], 0)

    def test_cleaner_scalars_and_all_null_columns_have_explicit_schema(self):
        raw = pd.DataFrame([dict(
            unique_id="synthetic", name="Synthetic Facility", address_city="Test City",
            address_stateOrRegion="Bihar", address_zipOrPostcode="800001",
            address_line1="Test", address_line2=None, latitude=25.6, longitude=85.1,
            procedure='["Hemodialysis treatment is provided."]', equipment='[]',
            capability='[]', specialties='["Nephrology"]', description="Synthetic test.",
            numberDoctors=None, capacity=None, yearEstablished=None, facilityTypeId="hospital",
            operatorTypeId="private", officialPhone=None, officialWebsite=None,
            source_urls=[], cluster_id=None,
        )])
        records, schema = facility_records(clean_facilities(raw))
        self.assertIsNone(records[0]["numberDoctors"])
        self.assertIs(type(records[0]["flag_row_corrupted"]), bool)
        self.assertIn("`numberDoctors` double", schema)
        self.assertIn("`procedure_list` array<string>", schema)
        json.dumps(records, allow_nan=False)

    def test_existing_tables_are_not_rebuilt(self):
        class Catalog:
            def tableExists(self, name):
                return True
        class Table:
            def count(self):
                return 123
        class Spark:
            catalog = Catalog()
            def sql(self, query):
                if not query.startswith("CREATE SCHEMA IF NOT EXISTS"):
                    raise AssertionError("Unexpected mutation")
            def table(self, name):
                return Table()
        result = prepare_missing_inputs(Spark())
        self.assertEqual(result["created"], [])


if __name__ == "__main__":
    unittest.main()
