"""Ridge calibration of the facility ranking against the 195 human relevance labels.

Used by train_ridge.ipynb. Kept free of Spark so it can be unit tested (tests/test_train_ridge.py).

- Features are rebuilt from the current evidence and cleaning tables, not from the frozen
  snapshot in the label file.
- The baseline is the production formula in ranking.py: evidence_score x location weight x
  exp(-distance / decay), with ranking.LOCATION_WEIGHTS, so the comparison is against what users
  actually get.
- Distance is the road distance the app uses (OSRM), with the app's estimate as fallback, so the
  learned distance weight matches how the app measures distance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import ranking

FEATURES = [
    "log_distance", "evidence_score", "log_direct", "location_weight",
    "is_hospital", "issue_count", "completeness", "capability_bloat",
]
ROUTINE_DECAY_KM = 75.0     # ranking.py policy for routine services (all 3 labelled services are routine)
ROAD_FACTOR = 1.4           # same fallback estimate as app/routing.py
SERVICE_RENAMES = {"hemodialysis": "dialysis"}  # the labels were drawn before the service rename


def location_weight(label) -> float:
    """Exactly the mapping ranking.py uses, including its default for unknown labels."""
    return ranking.LOCATION_WEIGHTS.get(str(label).strip().lower(), 0.30)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_features(base: pd.DataFrame, evidence: pd.DataFrame, facilities: pd.DataFrame,
                   distance_fn=None) -> pd.DataFrame:
    """Join labels to current evidence and facility data and compute model features.

    base: row_id, query_id, facility_id, service_code, query_lat, query_lon, label
    evidence: unique_id, service, evidence_score, n_direct, ...
    facilities: unique_id, facility_type, n_quality_flags, flag_capability_bloat, completeness,
                location_confidence, latitude, longitude
    distance_fn(origin, {facility_id: (lat, lon)}) -> ({facility_id: road km or None}, error);
                if None, straight-line distance is used (distance_basis = "straight").
    """
    df = (base.rename(columns={"facility_id": "unique_id"})
              .assign(service=lambda d: d["service_code"].replace(SERVICE_RENAMES))
              .merge(evidence.drop_duplicates(["unique_id", "service"]), on=["unique_id", "service"],
                     how="left", validate="many_to_one")
              .merge(facilities.drop_duplicates("unique_id"), on="unique_id", how="left", validate="many_to_one"))
    df["facility_id"] = df["unique_id"]
    df["straight_km"] = haversine_km(df["query_lat"], df["query_lon"], df["latitude"], df["longitude"])

    df["distance_km"], df["distance_basis"] = df["straight_km"], "straight"
    if distance_fn is not None:
        for _, rows in df.groupby("query_id"):
            origin = (float(rows["query_lat"].iloc[0]), float(rows["query_lon"].iloc[0]))
            points = {r.unique_id: (float(r.latitude), float(r.longitude)) for r in rows.itertuples()}
            roads, _error = distance_fn(origin, points)
            for idx, r in rows.iterrows():
                road = roads.get(r["unique_id"])
                if road is None:
                    df.loc[idx, ["distance_km", "distance_basis"]] = [r["straight_km"] * ROAD_FACTOR, "estimated road"]
                else:
                    df.loc[idx, ["distance_km", "distance_basis"]] = [road, "road"]

    df["location_weight"] = df["location_confidence"].map(location_weight)
    df["log_distance"] = np.log1p(df["distance_km"].astype(float))
    df["log_direct"] = np.log1p(pd.to_numeric(df["n_direct"], errors="coerce"))
    df["is_hospital"] = df["facility_type"].astype(str).str.lower().eq("hospital").astype(int)
    df["issue_count"] = pd.to_numeric(df["n_quality_flags"], errors="coerce")
    df["capability_bloat"] = df["flag_capability_bloat"].fillna(False).astype(int)
    df["baseline_score"] = baseline_score(df)
    return df


def baseline_score(df: pd.DataFrame) -> pd.Series:
    """The production formula from ranking.py for routine services."""
    return df["evidence_score"] * df["location_weight"] * np.exp(-df["distance_km"].astype(float) / ROUTINE_DECAY_KM)


def check_training_data(df: pd.DataFrame, n_rows: int = 195, n_queries: int = 15) -> None:
    assert len(df) == n_rows, f"Expected {n_rows} rows, got {len(df)}"
    assert df["row_id"].is_unique, "Duplicate row_id"
    assert df["label"].isin([0, 1, 2, 3]).all(), "Labels must be 0-3"
    assert df["evidence_score"].notna().all(), "Missing evidence_score after the join"
    assert df["latitude"].notna().all(), "Missing facility coordinates after the join"
    assert df["query_id"].nunique() == n_queries, "Unexpected number of queries"


def evaluate_query(rows: pd.DataFrame, score_col: str) -> dict:
    """NDCG@5 against the human labels, and Precision@3 (label >= 2 counts as relevant)."""
    from sklearn.metrics import ndcg_score
    ranked = rows.sort_values([score_col, "distance_km", "facility_id"], ascending=[False, True, True])
    return {
        "ndcg_at_5": float(ndcg_score([rows["label"].to_numpy()], [rows[score_col].to_numpy()], k=5)),
        "precision_at_3": float(ranked.head(3)["label"].ge(2).mean()),
    }


def new_model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def leave_one_query_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Train on 14 queries, score the held-out one; returns (per-query metrics, held-out scores)."""
    from sklearn.model_selection import LeaveOneGroupOut
    X, y, groups = df[FEATURES].astype(float), df["label"].astype(int), df["query_id"]
    held_out = pd.Series(np.nan, index=df.index)
    results = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]), "Query leaked into training"
        model = new_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        test = df.iloc[test_idx].copy()
        test["ridge_test_score"] = model.predict(X.iloc[test_idx])
        held_out.iloc[test_idx] = test["ridge_test_score"].to_numpy()
        for name, col in (("Current formula", "baseline_score"), ("Ridge", "ridge_test_score")):
            results.append({"query_id": test["query_id"].iloc[0], "model": name, **evaluate_query(test, col)})
    return pd.DataFrame(results), held_out


def fit_weights(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Fit on all labelled rows; return standardized weights (largest first) and the intercept."""
    model = new_model().fit(df[FEATURES].astype(float), df["label"].astype(int))
    ridge = model.named_steps["ridge"]
    weights = pd.DataFrame({"feature": FEATURES, "standardized_weight": ridge.coef_})
    weights = weights.reindex(weights["standardized_weight"].abs().sort_values(ascending=False).index)
    return weights.reset_index(drop=True), float(ridge.intercept_)


def sign_warnings(weights: pd.DataFrame) -> list[str]:
    """Signs we expect a priori; a violation means check the labels before adopting the weights."""
    expected = {"log_distance": -1, "evidence_score": +1, "location_weight": +1}
    w = dict(zip(weights["feature"], weights["standardized_weight"]))
    return [f"{f} has weight {w[f]:+.3f}, expected {'negative' if s < 0 else 'positive'}"
            for f, s in expected.items() if f in w and math.copysign(1, w[f]) != s]
