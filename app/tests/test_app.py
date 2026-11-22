"""Headless tests on synthetic data. Run from the repo root: python app/tests/test_app.py"""
import os
import sys

os.environ["APP_SAMPLE_DATA"] = "1"
os.environ.setdefault("APP_LLM", "stub")  # deterministic fake LLM
os.environ.setdefault("APP_ROUTING", "stub")  # deterministic fake road distances
HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
sys.path.insert(0, APP_DIR)

from streamlit.testing.v1 import AppTest  # noqa: E402

from llm_review import parse_answer, reconcile  # noqa: E402
from locations import PlaceIndex  # noqa: E402
from sample_data import build_sample  # noqa: E402
from search import Search  # noqa: E402

APP = os.path.join(APP_DIR, "app.py")


def run(service, place):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="service").set_value(service)
    at.text_input(key="place").input(place)
    at.button[-1].click().run()  # form submit button
    assert not at.exception, at.exception
    return at


def text(at):
    return "\n".join(m.value for m in at.markdown)


def test_strong_nearby_and_weak_folded():
    at = run("dialysis", "Patna")
    assert any("explicitly describe" in s.value for s in at.success)
    t = text(at)
    assert "Sample Patna Kidney Hospital" in t and "Maintenance hemodialysis" in t
    assert any("only a specialty tag" in e.label for e in at.expander)  # weak results are folded


def test_each_card_explains_its_evidence_level():
    at = run("dialysis", "Patna")
    t = text(at)
    assert "Why strong:** the listing states the service directly: “Maintenance hemodialysis" in t
    assert "Why only moderate:** the evidence appears only in the equipment list (“Dialysis machines”)" in t
    assert "Why only weak:** the only mention is the specialty tag “nephrology”" in t


def test_rural_point_shows_nearest_strong():
    at = run("dialysis", "25.9, 86.7")
    assert at.error and "does not prove none exists" in at.error[0].value
    assert any("Nearest strong evidence" in h.value for h in at.subheader)


def test_unknown_place_suggests():
    at = run("mri", "Jaipr")
    assert "Did you mean: Jaipur, Rajasthan" in at.error[0].value


def test_map_click_opens_details():
    at = run("dialysis", "Patna")
    at.session_state["map|dialysis|Patna"] = {
        "selection": {"indices": {"facilities": [1]}, "objects": {"facilities": [{"uid": "pat-moderate"}]}}}
    at.run()
    assert not at.exception, at.exception
    md = [m.value for m in at.markdown]
    # The badge sits on the clicked facility's card, right before its title
    i = next(n for n, v in enumerate(md) if "Selected on the map" in v)
    assert md[i + 1].startswith("### ") and "Sample Patna Care Hospital" in md[i + 1]
    assert sum("Selected on the map" in v for v in md) == 1


def test_clicking_weak_point_expands_its_section():
    at = run("dialysis", "Patna")
    at.session_state["map|dialysis|Patna"] = {
        "selection": {"indices": {"facilities": [3]}, "objects": {"facilities": [{"uid": "pat-weak"}]}}}
    at.run()
    assert not at.exception, at.exception
    weak = next(e for e in at.expander if "only a specialty tag" in e.label)
    assert weak.proto.expanded


def test_ai_review_agreement_and_disagreement():
    at = run("dialysis", "Patna")
    boxes = [x.value for x in at.success] + [x.value for x in at.warning] + [x.value for x in at.info]
    assert any("AI review agrees: strong." in v for v in boxes)
    # AI weaker than rules -> both ratings, both quotes, suggestion follows the AI
    down = next(v for v in boxes if "Hemodialysis unit for inpatients" in v and "read the same line differently" in v)
    assert "**Rules: strong**" in down and "**AI review: moderate**" in down
    assert down.count("Hemodialysis unit for inpatients") == 1  # the shared line is quoted once
    assert "we recommend treating this as moderate" in down
    # AI stronger, quote states availability -> upgrade accepted
    up_ok = next(v for v in boxes if "Dialysis facility available on site" in v and "Suggestion" in v)
    assert "we recommend treating this as strong" in up_ok
    # AI stronger, quote only names a machine -> rule rating kept
    up_no = next(v for v in boxes if "“Dialysis machines”" in v and "Suggestion" in v)
    assert "we recommend treating this as moderate" in up_no and "only names equipment" in up_no


def test_ranking_follows_ai_suggestion_and_explains_it():
    at = run("dialysis", "Patna")
    note = next(x.value for x in at.info if "so the list was re-ranked" in x.value)
    assert "Sample Patna New Hospital**: strong → **none**, removed from the list" in note
    assert "Sample Patna Kidney Centre**: moderate → **strong**" in note
    assert "Sample Patna Government Hospital**: strong → **moderate**" in note
    assert "same formula" in note and "strong 0.85, moderate 0.55, weak 0.20" in note
    titles = [m.value for m in at.markdown if m.value.startswith("### ")]
    assert not any("Sample Patna New Hospital" in t for t in titles[:5] if not t.startswith("### –"))
    removed = next(e for e in at.expander if e.label.startswith("Removed after the AI review"))
    assert any("Sample Patna New Hospital" in m.value for m in removed.markdown)
    assert any("Moved from" in c.value and "after the AI review" in c.value for c in at.caption)


def test_apply_reviews_scores_use_same_formula():
    from adjust import apply_reviews
    fac, evi, loc = build_sample()
    s, p = Search(fac, evi), PlaceIndex(fac, loc)
    patna = p.resolve("Patna")
    res = s.run("dialysis", patna.lat, patna.lon, k=10)
    gov = next(c for c in res["pool"] if c["unique_id"] == "pat-medium-loc")
    out = apply_reviews(res, {("pat-medium-loc", "dialysis"): {"level": "moderate", "quote": "Hemodialysis unit for inpatients.",
                                                                "reason": "Inpatients only."}}, "dialysis")
    adj = next(c for c in out["main"] if c["unique_id"] == "pat-medium-loc")
    b = gov["score_breakdown"]
    assert abs(adj["score"] - 0.55 * b["location_weight"] * b["distance_factor"]) < 1e-12
    assert adj["score_breakdown"]["rule_evidence_score"] == b["evidence_score"]


def test_road_distances_are_shown_and_used_for_ranking():
    at = run("dialysis", "Patna")
    t = text(at)
    assert "km by road**" in t and "straight line" not in t  # with routing, straight-line distance is not shown
    road_note = next(c.value for c in at.caption if "driving distances" in c.value)
    assert "**Sample Patna Riverside Hospital**: moved down (41.7 km by road)" in road_note


def test_road_distances_unavailable_falls_back_to_straight_line():
    import importlib
    os.environ["APP_ROUTING"] = "off"
    try:
        at = run("dialysis", "Patna")
        assert any("Road routing is unavailable" in c.value and "1.4 × the straight-line" in c.value for c in at.caption)
        t = text(at)
        assert "km by road (estimated)** ·" in t and "km in a straight line" in t
    finally:
        os.environ["APP_ROUTING"] = "stub"


def test_ai_quote_not_in_listing_is_discarded():
    at = run("mri", "Jaipur")
    assert any("not in this listing" in c.value for c in at.caption)


def test_reconcile_policy():
    assert reconcile("strong", {"level": "strong", "quote": "x"})["agree"]
    assert reconcile("strong", {"level": "none", "quote": "Unit being set up"})["recommended"] == "none"
    assert reconcile("moderate", {"level": "strong", "quote": "CT scan available 24x7"})["recommended"] == "strong"
    assert reconcile("moderate", {"level": "strong", "quote": "CT scanner used for FNAC guidance"})["recommended"] == "moderate"
    assert reconcile("strong", {"level": "none", "quote": ""})["recommended"] == "strong"


def test_weak_results_are_reviewed_too():
    at = run("dialysis", "Patna")
    weak = next(e for e in at.expander if "only a specialty tag" in e.label)
    shown = [x.value for x in list(weak.success) + list(weak.warning) + list(weak.info) + list(weak.caption)]
    assert any("AI review" in v for v in shown)


def test_parse_answer_rejects_bad_output():
    lines = [("procedure", "Provides dialysis treatment")]
    ok = parse_answer('Sure: {"level": "strong", "quote": "Provides dialysis treatment", "reason": "Stated."}', lines)
    assert ok["level"] == "strong" and ok["quote"] == "Provides dialysis treatment"
    assert "error" in parse_answer('{"level": "maybe", "quote": "", "reason": ""}', lines)
    assert "error" in parse_answer('{"level": "strong", "quote": "Made-up sentence", "reason": ""}', lines)
    assert "error" in parse_answer("no json here", lines)


def test_ranking_exclusions_and_grouping():
    fac, evi, loc = build_sample()
    s, p = Search(fac, evi), PlaceIndex(fac, loc)
    patna = p.resolve("Patna")
    r = s.run("dialysis", patna.lat, patna.lon)
    ids = [c["unique_id"] for c in r["main"]]
    assert ids[0] == "pat-strong" and "pat-weak" not in ids and "bad-row" not in ids
    assert [c["unique_id"] for c in r["weak"]] == ["pat-weak"]
    assert r["excluded"] == {"Record is corrupted (no usable name)": 1}
    jaipur = p.resolve("Jaipur")
    r = s.run("mri", jaipur.lat, jaipur.lon)
    assert r["excluded"] == {"Quoted evidence was not found in the facility's own text": 1}
    warn = next(c for c in s.run("dialysis", patna.lat, patna.lon, k=10)["main"] if c["unique_id"] == "pat-medium-loc")
    assert len(warn["warnings"]) == 1  # quality note and flag name are not duplicated


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
