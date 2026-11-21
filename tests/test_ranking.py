import copy
import json
from pathlib import Path
import unittest

from ranking import Ranker, configure, distance_km, load_from_spark, recommend


SAMPLE = Path(__file__).resolve().parents[1] / "examples/04_ranking_sample.json"


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(SAMPLE.read_text())
        self.facilities = self.data["facilities"]
        self.evidence = self.data["evidence"]

    def ranker(self):
        return Ranker(self.facilities, self.evidence)

    def query(self):
        return self.ranker().recommend("dialysis", 25.5941, 85.1376, k=1)

    def test_k_one_preserves_both_fallback_groups(self):
        result = self.query()
        self.assertEqual(result["status"], "no_nearby_strong_evidence")
        self.assertEqual(result["recommendations"][0]["unique_id"], "sample-weak")
        self.assertEqual(result["nearby_weak_options"][0]["unique_id"], "sample-weak")
        self.assertEqual(result["nearest_strong_option"]["unique_id"], "sample-strong")
        self.assertGreater(result["nearest_strong_option"]["distance_km"], 60)
        self.assertEqual(result["excluded_rows"]["unusable_coordinates"], 1)

    def test_nearby_strong_and_public_function(self):
        configure(self.ranker())
        result = recommend(" Dialysis ", 26.40, 85.20)
        self.assertEqual(result["status"], "nearby_strong_evidence")
        self.assertEqual(result["recommendations"][0]["unique_id"], "sample-strong")

    def test_score_and_quote_are_auditable(self):
        item = self.query()["nearest_strong_option"]
        parts = item["score_breakdown"]
        self.assertAlmostEqual(item["score"], parts["evidence_score"] * parts["location_weight"] * parts["distance_factor"])
        self.assertEqual(item["quotes"], self.evidence[1]["quotes"])
        self.assertIn("provisional", item["explanation"])
        self.assertIn("straight line", item["explanation"])
        self.assertIn("Hemodialysis treatment is provided.", item["quotes"][0]["text"])
        json.dumps(self.query(), allow_nan=False)

    def test_nearest_strong_means_distance_not_top_score(self):
        other = copy.deepcopy(self.facilities[1])
        other.update(facility_id="closer", latitude=26.20)
        self.facilities.append(other)
        evidence = copy.deepcopy(self.evidence[1])
        evidence.update(unique_id="closer", evidence_score=0.01)
        self.evidence.append(evidence)
        result = self.query()
        self.assertEqual(result["nearest_strong_option"]["unique_id"], "closer")

    def test_urgent_distance_penalty_is_steeper(self):
        for row in list(self.evidence):
            urgent = copy.deepcopy(row)
            urgent["service"] = "trauma"
            self.evidence.append(urgent)
        ranker = self.ranker()
        routine = ranker.recommend("dialysis", 25.5941, 85.1376)
        urgent = ranker.recommend("trauma", 25.5941, 85.1376)
        self.assertLess(urgent["nearest_strong_option"]["score"], routine["nearest_strong_option"]["score"])
        self.assertEqual(urgent["policy"]["nearby_km"], 30)

    def test_no_service_alias_infers_emergency_c_section(self):
        self.assertEqual(self.ranker().recommend("emergency c-section", 25.6, 85.1)["status"], "unknown_service")

    def test_label_only_strong_is_downgraded(self):
        self.evidence[0]["evidence_level"] = "strong"
        self.evidence[0]["n_direct"] = 1
        item = self.query()["recommendations"][0]
        self.assertEqual(item["evidence_level"], "weak")
        self.assertEqual(item["original_evidence_level"], "strong")
        self.assertEqual(item["confidence"], "low")

    def test_strong_without_direct_count_is_downgraded(self):
        self.evidence[1]["n_direct"] = 0
        self.assertIsNone(self.query()["nearest_strong_option"])

    def test_duplicate_keys_fail_instead_of_fanout(self):
        self.facilities.append(copy.deepcopy(self.facilities[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate facility_id"):
            self.ranker()
        self.facilities.pop()
        self.evidence.append(copy.deepcopy(self.evidence[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate facility-service"):
            self.ranker()

    def test_possible_merged_identity_reduces_weight(self):
        self.facilities[1]["flag_duplicate_cluster"] = True
        item = self.query()["nearest_strong_option"]
        self.assertEqual(item["score_breakdown"]["location_weight"], 0.4)
        self.assertEqual(item["confidence"], "low")
        self.assertTrue(any("merged identity" in warning for warning in item["data_warnings"]))

    def test_nonfinite_coordinates_are_excluded(self):
        self.facilities[1]["latitude"] = float("nan")
        self.assertIsNone(self.query()["nearest_strong_option"])
        self.assertEqual(self.query()["status"], "no_usable_strong_evidence")

    def test_missing_facility_is_counted(self):
        self.facilities.pop(1)
        self.assertEqual(self.query()["excluded_rows"]["missing_facility_match"], 1)

    def test_quotes_require_fields_and_original_text(self):
        self.evidence[1]["quotes"] = [{"field": "procedure", "text": "Invented text"}]
        result = self.query()
        self.assertIsNone(result["nearest_strong_option"])
        self.assertEqual(result["excluded_rows"]["quote_not_in_source"], 1)
        self.evidence[1]["quotes"] = "Unstructured claim"
        self.assertEqual(self.query()["excluded_rows"]["missing_structured_quotes"], 1)

    def test_json_encoded_quotes_and_raw_source(self):
        self.evidence[1]["quotes"] = json.dumps(self.evidence[1]["quotes"])
        self.facilities[1]["procedure"] = json.dumps(self.facilities[1]["procedure_items"])
        self.assertIsNotNone(self.query()["nearest_strong_option"])

    def test_missing_source_field_limits_confidence(self):
        self.facilities[1].pop("procedure_items")
        item = self.query()["nearest_strong_option"]
        self.assertEqual(item["confidence"], "low")
        self.assertTrue(any("unavailable for text" in warning for warning in item["data_warnings"]))

    def test_input_and_schema_validation(self):
        for lat, lon, k in [(float("inf"), 85, 5), (91, 85, 5), (25, 181, 5), (25, 85, 0), (25, 85, True)]:
            with self.assertRaises(ValueError):
                self.ranker().recommend("dialysis", lat, lon, k)
        self.facilities[0]["flag_outside_india_bbox"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.ranker()

    def test_invalid_score_and_count_do_not_get_clamped(self):
        self.evidence[1]["evidence_score"] = 1.5
        self.assertIsNone(self.query()["nearest_strong_option"])
        self.evidence[1]["evidence_score"] = 0.9
        self.evidence[1]["n_direct"] = -1
        self.assertEqual(self.query()["excluded_rows"]["invalid_evidence"], 1)

    def test_no_nearby_options_does_not_promote_distant_weak_claim(self):
        result = self.ranker().recommend("dialysis", 19.076, 72.8777)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["nearby_weak_options"], [])
        self.assertIsNotNone(result["nearest_strong_option"])

    def test_distance_is_zero_at_same_point(self):
        self.assertEqual(distance_km(25, 85, 25, 85), 0)
        self.assertAlmostEqual(distance_km(0, 0, 0, 1), 111.195, places=2)

    def test_combined_score_order_and_deterministic_ties(self):
        self.facilities[1].update(latitude=25.60, longitude=85.14)
        result = self.query()
        self.assertEqual(result["recommendations"][0]["unique_id"], "sample-strong")
        self.evidence[1]["evidence_score"] = self.evidence[0]["evidence_score"]
        first = self.ranker().recommend("dialysis", 25.5941, 85.1376)
        self.facilities.reverse()
        self.evidence.reverse()
        second = self.ranker().recommend("dialysis", 25.5941, 85.1376)
        self.assertEqual(first["recommendations"], second["recommendations"])
        self.assertEqual(first["recommendations"][0]["unique_id"], "sample-strong")

    def test_all_unusable_and_unknown_service_are_distinct(self):
        self.facilities = []
        self.assertEqual(self.query()["status"], "no_usable_candidates")
        self.assertEqual(self.ranker().recommend("not_a_service", 25.6, 85.1)["status"], "unknown_service")

    def test_unknown_quality_flag_lowers_confidence(self):
        self.facilities[1]["flag_duplicate_cluster"] = None
        item = self.query()["nearest_strong_option"]
        self.assertEqual(item["score_breakdown"]["location_weight"], 0.4)
        self.assertEqual(item["confidence"], "low")


class CurrentProducerTests(unittest.TestCase):
    def setUp(self):
        path = SAMPLE.with_name("04_ranking_current_sample.json")
        self.data = json.loads(path.read_text())

    def query(self):
        return Ranker(self.data["facilities"], self.data["evidence"]).recommend(
            "hemodialysis", 25.5941, 85.1376
        )

    def test_current_schema_and_sentence_quotes(self):
        result = self.query()
        self.assertEqual(result["status"], "no_nearby_strong_evidence")
        strong = result["nearest_strong_option"]
        self.assertEqual(strong["score_breakdown"]["location_weight"], 0.9)
        self.assertEqual(strong["location_confidence"], "high")
        self.assertEqual(strong["quotes"][0]["text"], "Hemodialysis treatment is provided.")
        self.assertNotIn("unavailable for text", " ".join(strong["data_warnings"]))

    def test_corrupted_rows_never_recommended(self):
        self.data["facilities"][1]["flag_row_corrupted"] = True
        result = self.query()
        self.assertIsNone(result["nearest_strong_option"])
        self.assertEqual(result["excluded_rows"]["corrupted_facility"], 1)

    def test_upstream_low_location_trust_reduces_score(self):
        high = self.query()["nearest_strong_option"]
        self.data["facilities"][1]["location_confidence"] = "low"
        self.data["facilities"][1]["flag_coords_far_from_city"] = True
        self.data["facilities"][1]["quality_notes"] = "Coordinates >150 km from other facilities in the same city"
        low = self.query()["nearest_strong_option"]
        self.assertAlmostEqual(low["score"], high["score"] / 3)
        self.assertEqual(low["confidence"], "low")
        self.assertIn("flag_coords_far_from_city", low["data_warnings"])
        self.assertIn("Coordinates >150 km", " ".join(low["data_warnings"]))

    def test_unknown_corruption_status_is_excluded(self):
        self.data["facilities"][1]["flag_row_corrupted"] = None
        self.assertEqual(self.query()["excluded_rows"]["unknown_corruption_status"], 1)

    def test_current_missing_field_is_not_guessed(self):
        self.data["facilities"][0].pop("location_confidence")
        with self.assertRaisesRegex(ValueError, "location_confidence"):
            self.query()

    def test_spark_loader_excludes_every_ambiguous_id(self):
        facilities = copy.deepcopy(self.data["facilities"])
        duplicate = copy.deepcopy(facilities[1])
        duplicate["name"] = "Conflicting facility name"
        facilities.append(duplicate)
        evidence = copy.deepcopy(self.data["evidence"])

        class Frame:
            def __init__(self, rows):
                self.rows = rows
                self.columns = list(rows[0])

            def select(self, *columns):
                return Frame([{key: row.get(key) for key in columns} for row in self.rows])

            def limit(self, maximum):
                return Frame(self.rows[:maximum])

            def collect(self):
                return self.rows

        class Spark:
            def table(self, name):
                return Frame(facilities if name.endswith("facilities_clean") else evidence)

        ranker = load_from_spark(Spark())
        self.assertNotIn(duplicate["unique_id"], ranker.facilities)
        self.assertEqual(ranker.input_diagnostics["ambiguous_facility_ids_excluded"], 1)
        self.assertEqual(ranker.input_diagnostics["facility_rows_excluded_for_ambiguous_ids"], 2)
        result = ranker.recommend("hemodialysis", 25.5941, 85.1376)
        self.assertIsNone(result["nearest_strong_option"])
        self.assertEqual(result["excluded_rows"]["missing_facility_match"], 1)


if __name__ == "__main__":
    unittest.main()
