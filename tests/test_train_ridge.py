"""Tests for train_ridge.py on synthetic data (no Spark, no real labels)."""

import unittest

try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
    HAVE_DEPS = True
except ImportError:  # the ranking tests run without these; these tests need them
    HAVE_DEPS = False

import ranking

if HAVE_DEPS:
    import train_ridge as tr


def synthetic(n_queries=6, per_query=8, seed=0):
    """Labels fall with distance and rise with evidence, so the expected signs are known."""
    rng = np.random.default_rng(seed)
    rows, evidence, facilities = [], [], []
    for q in range(n_queries):
        qlat, qlon = 25.0 + q, 80.0 + q
        for i in range(per_query):
            uid = f"f{q}-{i}"
            lat, lon = qlat + rng.uniform(0.01, 0.5), qlon + rng.uniform(0.01, 0.5)
            ev = rng.choice([0.2, 0.55, 0.85, 0.9])
            km = float(tr.haversine_km(qlat, qlon, lat, lon))
            label = int(np.clip(round(3 * ev - km / 25 + rng.normal(0, 0.2)), 0, 3))
            rows.append(dict(row_id=f"R{q}{i}", query_id=f"Q{q}", facility_id=uid, service_code="hemodialysis",
                             query_lat=qlat, query_lon=qlon, label=label))
            evidence.append(dict(unique_id=uid, service="dialysis", evidence_score=ev, n_direct=int(ev > 0.6)))
            facilities.append(dict(unique_id=uid, facility_type="hospital", n_quality_flags=0,
                                   flag_capability_bloat=False, completeness=0.5,
                                   location_confidence=rng.choice(["high", "medium", "low"]),
                                   latitude=lat, longitude=lon))
    return pd.DataFrame(rows), pd.DataFrame(evidence), pd.DataFrame(facilities)


@unittest.skipUnless(HAVE_DEPS, "needs numpy, pandas and scikit-learn")
class TrainRidgeTests(unittest.TestCase):
    def setUp(self):
        self.df = tr.build_features(*synthetic())

    def test_location_weight_matches_ranking(self):
        for label, weight in ranking.LOCATION_WEIGHTS.items():
            self.assertEqual(tr.location_weight(label.upper()), weight)
        self.assertEqual(tr.location_weight("unknown"), 0.30)

    def test_baseline_equals_production_ranking_score(self):
        """The baseline must be the exact score ranking.Ranker gives users."""
        row = self.df.iloc[0]
        facility = dict(unique_id=row.unique_id, name="Synthetic", city_clean="X", state_clean="Y",
                        latitude=row.latitude, longitude=row.longitude, location_confidence=row.location_confidence,
                        flag_row_corrupted=False, flag_coords_missing=False, flag_coords_outside_india=False,
                        flag_coords_far_from_city=False, flag_pin_state_mismatch=False,
                        flag_possible_duplicate=False, quality_notes="", procedure_list=["Dialysis is provided"])
        evidence = dict(unique_id=row.unique_id, service="dialysis", evidence_level="strong",
                        evidence_score=row.evidence_score, n_direct=1, n_supporting=0,
                        quotes=[{"field": "procedure_list", "sentence": "Dialysis is provided"}])
        result = ranking.Ranker([facility], [evidence]).recommend("dialysis", row.query_lat, row.query_lon, k=1)
        candidate = (result["recommendations"] or [result["nearest_strong_option"]])[0]
        self.assertEqual(result["policy"]["decay_km"], tr.ROUTINE_DECAY_KM)
        self.assertAlmostEqual(candidate["score"], row.baseline_score, places=9)

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        rows = pd.DataFrame({"label": [3, 2, 1, 0], "score": [0.9, 0.5, 0.2, 0.1],
                             "distance_km": [1, 2, 3, 4], "facility_id": list("abcd")})
        m = tr.evaluate_query(rows, "score")
        self.assertAlmostEqual(m["ndcg_at_5"], 1.0)
        self.assertAlmostEqual(m["precision_at_3"], 2 / 3)

    def test_leave_one_query_out_scores_every_row_once(self):
        per_query, held_out = tr.leave_one_query_out(self.df)
        self.assertEqual(len(per_query), 2 * self.df["query_id"].nunique())
        self.assertTrue(held_out.notna().all())

    def test_learned_signs_match_the_synthetic_truth(self):
        weights, _ = tr.fit_weights(self.df)
        self.assertEqual(tr.sign_warnings(weights), [])
        flipped = weights.assign(standardized_weight=-weights["standardized_weight"])
        self.assertEqual(len(tr.sign_warnings(flipped)), 3)

    def test_road_distance_and_estimated_fallback(self):
        base, evidence, facilities = synthetic(n_queries=1, per_query=3)

        def fake_osrm(origin, points):
            ids = sorted(points)
            return {ids[0]: 12.5, ids[1]: None}, None  # third id missing entirely

        df = tr.build_features(base, evidence, facilities, distance_fn=fake_osrm).sort_values("unique_id")
        self.assertEqual(df["distance_basis"].tolist(), ["road", "estimated road", "estimated road"])
        self.assertEqual(df["distance_km"].iloc[0], 12.5)
        self.assertAlmostEqual(df["distance_km"].iloc[1], df["straight_km"].iloc[1] * tr.ROAD_FACTOR)


if __name__ == "__main__":
    unittest.main()
