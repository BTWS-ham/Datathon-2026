"""CareFinder: find the right care in India, with the evidence behind every answer."""

from __future__ import annotations

import json
import re

import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from adjust import LEVEL_SCORE, apply_reviews, apply_road
from data import load_tables, using_sample
from llm_review import RANK, make_reviewer, reconcile, review_many
from locations import PlaceIndex
from routing import ROAD_FACTOR, SOURCE_NAME, road_km
from search import Search, ranking, service_label  # search.py puts the repo root (ranking.py) on sys.path

st.set_page_config(page_title="CareFinder India", page_icon="🏥", layout="wide")

LEVEL_TEXT = {
    "strong": ("Strong evidence", "green", "The facility's own listing explicitly describes this service."),
    "moderate": ("Moderate evidence", "orange", "The listing names related equipment, but does not say the service is offered."),
    "weak": ("Weak evidence", "red", "Only a specialty or department hints at this service. Verify before travelling."),
    "none": ("No current evidence", "gray", "The AI review found the listing does not support this service now."),
}
CONF_TEXT = {"moderate": ("Moderate", "orange"), "low": ("Low", "red")}
FIELD_TEXT = {"procedure": "procedures list", "equipment": "equipment list", "capability": "capabilities list",
              "description": "description", "specialties": "specialty tags", "specialty": "specialty tags"}
EXAMPLES = [("dialysis", "Patna"), ("mri", "Jaipur"), ("ct_scan", "Lucknow"), ("c_section", "Lucknow")]


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading facility data…", ttl=3600)
def get_backend():
    fac, evi, loc, notes = load_tables()
    return Search(fac, evi), PlaceIndex(fac, loc), notes, fac


@st.cache_resource(show_spinner=False)
def get_reviewer():
    return make_reviewer()


try:
    search, places, load_notes, fac_df = get_backend()
except Exception as e:  # show a readable error instead of a stack trace
    st.error(f"Could not load data: {e}")
    st.stop()


# ---------------------------------------------------------------- helpers
def esc(s) -> str:
    """Escape dataset text so it renders literally in markdown."""
    s = str(s)
    for ch in "\\`*_[]#$|~":
        s = s.replace(ch, "\\" + ch)
    return s


def ext_url(u) -> str:
    """Dataset URLs often lack a scheme; without one the browser treats them as paths on this app."""
    u = str(u).strip()
    return u if re.match(r"^https?://", u, re.I) else "https://" + u.lstrip("/")


def road_label(c: dict) -> str:
    return "km by road (estimated)" if c.get("road_estimated") else "km by road"


def distance_text(c: dict) -> str:
    if c.get("road_estimated"):
        return (f"**≈ {c['road_km']:.1f} km by road (estimated)** · {c['straight_km']:.1f} km in a straight line; "
                "no road route was available, so the road distance is estimated")
    return f"**{c['distance_km']:.1f} km by road**"


def badge(text: str, color: str) -> str:
    return f":{color}-background[{text}]"


def field_text(field: str) -> str:
    return FIELD_TEXT.get(field.removesuffix("_list"), field)


def anchor_id(unique_id: str) -> str:
    return "card-" + re.sub(r"[^A-Za-z0-9_-]", "-", str(unique_id))


def scroll_to(anchor: str):
    """Smoothly scroll the page to a card anchor (the component iframe shares the app's origin)."""
    components.html(f"""<script>
      const id = {json.dumps(anchor)};
      let tries = 0;
      const go = () => {{
        const el = window.parent.document.getElementById(id);
        if (el) {{ el.scrollIntoView({{behavior: "smooth", block: "start"}}); }}
        else if (tries++ < 30) {{ setTimeout(go, 100); }}
      }};
      go();
    </script>""", height=0)


def evidence_reason(c: dict) -> str:
    """One plain-English sentence on why this card got its evidence level, citing its own quotes."""
    service = service_label(c.get("service", "this service")).lower()
    quotes = c["quotes"]
    tags = [q["text"] for q in quotes if q["field"].removesuffix("_list") in ("specialties", "specialty")]
    gear = [q["text"] for q in quotes if q["field"].removesuffix("_list") == "equipment"]
    stated = [q["text"] for q in quotes if q["field"].removesuffix("_list") in ("procedure", "capability", "description")]
    level = c["evidence_level"]
    if level == "strong":
        quote = (stated or gear or [quotes[0]["text"]])[0]
        return f"**Why strong:** the listing states the service directly: “{esc(quote)}”."
    if level == "moderate":
        quote = (gear or [quotes[0]["text"]])[0]
        # Moderate comes from equipment-list lines: either a machine name, or a service named
        # only in the equipment list. The wording below is accurate for both cases.
        return (f"**Why only moderate:** the evidence appears only in the equipment list (“{esc(quote)}”). "
                f"No procedure, capability or description sentence states that {esc(service)} is offered.")
    if tags and not stated and not gear:
        named = ", ".join(f"“{esc(t)}”" for t in tags[:3])
        return (f"**Why only weak:** the only mention is the specialty tag {named}. "
                f"A related department does not show that {esc(service)} is done here; nothing in the listing "
                f"says it provides {esc(service)} or names equipment for it.")
    quote = quotes[0]["text"]
    return (f"**Why only weak:** the listing only mentions a related department (“{esc(quote)}”), "
            f"not {esc(service)} itself. Call to confirm before referring.")


def render_review(c: dict, review: dict | None):
    """AI second opinion. Agreement: one line. Disagreement: both ratings with quotes, plus a suggestion."""
    if review is None:
        return
    if review.get("error"):
        st.caption(f"🤖 AI review unavailable: {esc(review['error'])}.")
        return
    rule, ai = c.get("rule_level", c["evidence_level"]), review["level"]
    reason = esc(review.get("reason") or "")
    verdict = reconcile(rule, review)
    if verdict["agree"]:
        st.success(f"🤖 **AI review agrees: {ai}.** {reason}")
        return
    rule_quote = c["quotes"][0] if c["quotes"] else None
    rule_line = (f"“{esc(rule_quote['text'])}” ({field_text(rule_quote['field'])})" if rule_quote
                 else "no quotable line")
    ai_line = f"“{esc(review['quote'])}”" if review.get("quote") else "no supporting line given"
    rec = verdict["recommended"]
    if rule_quote and review.get("quote") and rule_quote["text"].strip() == review["quote"].strip():
        # Same line, read differently: quote it once
        text = (f"⚖️ **The rules and the AI review read the same line differently:** {rule_line}  \n"
                f"- **Rules: {rule}** — matched the service keywords in this line.  \n"
                f"- **AI review: {ai}** — {reason}  \n\n")
    else:
        text = (f"⚖️ **The rules and the AI review disagree**  \n"
                f"- **Rules: {rule}** — based on {rule_line}  \n"
                f"- **AI review: {ai}** — {reason} Based on {ai_line}  \n\n")
    text += f"**Suggestion: we recommend treating this as {rec}** — {esc(verdict['why'])}"
    if RANK[rec] < RANK[rule]:
        st.warning(text + " Call to confirm before referring.")
    else:
        st.info(text)


def render_card(rank, c: dict, policy: dict, compact: bool = False, selected: bool = False,
                review: dict | None = None):
    """compact=True is used inside an expander, where Streamlit forbids nested expanders."""
    lvl_name, lvl_color, _ = LEVEL_TEXT[c["evidence_level"]]
    conf_name, conf_color = CONF_TEXT.get(c["confidence"], (c["confidence"], "gray"))
    parts = c["score_breakdown"]
    st.markdown(f'<div id="{anchor_id(c["unique_id"])}"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        if selected:
            st.markdown(badge("📍 Selected on the map", "violet"))
        st.markdown(f"### {rank}. {esc(c['name'])}")
        if c.get("ai_adjusted") and c.get("final_rank"):
            before = f"#{c['rule_rank']}" if c.get("rule_rank") else "outside the list"
            st.caption(f"↕️ Moved from {before} to #{c['final_rank']} after the AI review changed its rating "
                       f"from {c['rule_level']} to {c['evidence_level']}.")
        place = ", ".join(x for x in [c.get("city"), c.get("state")] if x)
        kind = f" · {esc(c['facility_type']).capitalize()}" if c.get("facility_type") else ""
        st.markdown(f"{esc(place)}{kind} · {distance_text(c)}")
        st.markdown(" ".join([
            badge(lvl_name, lvl_color),
            *([badge(f"AI-adjusted (rules: {c['rule_level']})", "violet")] if c.get("ai_adjusted") else []),
            badge(f"Confidence: {conf_name}", conf_color),
            badge(f"Location confidence: {c['location_confidence']}", "blue"),
        ]))
        if c.get("ai_adjusted"):
            st.markdown(f"**Rated {c['evidence_level']} after the AI review** (the keyword rules rated it "
                        f"{c['rule_level']}). The reasons for both are below.")
        else:
            st.markdown(evidence_reason(c))
        render_review(c, review)

        st.markdown("**From its own listing:**")
        for q in c["quotes"][:2 if compact else 4]:
            st.markdown(f"> “{esc(q['text'])}”  \n> — *{field_text(q['field'])}*")
        if len(c["quotes"]) > 4 and not compact:
            with st.expander(f"{len(c['quotes']) - 4} more quotes"):
                for q in c["quotes"][4:]:
                    st.markdown(f"- “{esc(q['text'])}” — *{field_text(q['field'])}*")

        if c["warnings"]:
            st.markdown("**⚠️ Check before you go:**")
            for w in c["warnings"]:
                st.markdown(f"- {esc(w)}")

        contact = []
        if c.get("website"):
            contact.append(f"[Website]({ext_url(c['website'])})")
        if c.get("phone"):
            phone = esc(c["phone"])
            if c["phone_shared_by_n"] >= 3:
                phone += f" *(shared by {c['phone_shared_by_n']} facilities; likely a chain or booking line)*"
            contact.append(f"📞 {phone}")
        if contact:
            st.markdown(" · ".join(contact))
        if c["source_urls"]:
            st.caption("Sources: " + " · ".join(f"[{i + 1}]({ext_url(u)})" for i, u in enumerate(c["source_urls"][:5])))

        if compact:
            st.caption(f"Score {c['score']:.3f} = evidence {parts['evidence_score']:.2f} × location "
                       f"{parts['location_weight']:.2f} × distance {parts['distance_factor']:.2f} · ID {c['unique_id']}")
            return
        with st.expander("How this score was calculated"):
            st.markdown(
                "| Factor | Value | Why |\n|---|---|---|\n"
                f"| Evidence score | {parts['evidence_score']:.2f} | {lvl_name.lower()}"
                + (f", AI-adjusted from {parts['rule_evidence_score']:.2f} ({c['rule_level']})"
                   if "rule_evidence_score" in parts else "") + " "
                f"({c['n_direct']} direct, {c['n_supporting']} supporting mentions) |\n"
                f"| Location weight | {parts['location_weight']:.2f} | location confidence **{c['location_confidence']}** |\n"
                f"| Distance factor | {parts['distance_factor']:.2f} | exp(−{c['distance_km']:.1f} {road_label(c)} ÷ "
                f"{parts['distance_decay_km']:.0f} km) |\n"
                f"| **Score** | **{c['score']:.3f}** | product of the three |"
            )
            st.caption(f"Facility ID: {c['unique_id']}")


MAP_COLORS = {"strong": [44, 160, 44], "moderate": [255, 127, 14], "weak": [214, 39, 40]}


def _zoom_for(km: float) -> float:
    for limit, zoom in ((3, 12.5), (8, 11.5), (20, 10.3), (45, 9.3), (90, 8.3), (200, 7.2)):
        if km <= limit:
            return zoom
    return 6.0


def render_map(loc, items: list[tuple[str, dict]], key: str):
    """Clickable map. items are (label, card) pairs; returns the clicked (label, card) or None."""
    rows = [{"label": label, "uid": c["unique_id"], "name": c["name"], "lat": c["latitude"], "lon": c["longitude"],
             "color": MAP_COLORS[c["evidence_level"]], "level": LEVEL_TEXT[c["evidence_level"]][0],
             "km": f"{'≈ ' if c.get('road_estimated') else ''}{c['distance_km']:.1f} {road_label(c)}"}
            for label, c in items]
    points = pd.DataFrame(rows)
    farthest = max([c["distance_km"] for _, c in items] + [1.0])
    layers = [
        pdk.Layer("ScatterplotLayer", id="search-point", data=pd.DataFrame([{"lat": loc.lat, "lon": loc.lon}]),
                  get_position="[lon, lat]", get_fill_color=[31, 119, 180], get_radius=9, radius_units="pixels",
                  stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2),
        pdk.Layer("ScatterplotLayer", id="facilities", data=points, get_position="[lon, lat]",
                  get_fill_color="color", get_radius=13, radius_units="pixels", pickable=True, auto_highlight=True,
                  stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
        pdk.Layer("TextLayer", id="labels", data=points, get_position="[lon, lat]", get_text="label",
                  get_size=12, get_color=[255, 255, 255], get_text_anchor="'middle'",
                  get_alignment_baseline="'center'", font_weight=700),
    ]
    deck = pdk.Deck(
        layers=layers, map_style=pdk.map_styles.CARTO_LIGHT,
        initial_view_state=pdk.ViewState(latitude=loc.lat, longitude=loc.lon, zoom=_zoom_for(farthest)),
        tooltip={"html": "<b>{label}. {name}</b><br/>{level} · {km}<br/><i>Click for details</i>"},
    )
    event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key=key, height=400)
    selection = (event or {}).get("selection") or {}   # PydeckState is dict-like
    picked = (selection.get("objects") or {}).get("facilities", [])
    if picked:
        uid = picked[0].get("uid")
        return next(((label, c) for label, c in items if c["unique_id"] == uid), None)
    return None


# ---------------------------------------------------------------- page
st.title("🏥 CareFinder India")
st.markdown("Find a facility that offers a specific service near a place. "
            "**Every recommendation shows the facility's own words as evidence** and says how much to trust it.")
if using_sample():
    st.warning("Showing SYNTHETIC sample data for development. These are not real facilities.")
for n in load_notes:
    st.caption(n)

tab_find, tab_how, tab_data = st.tabs(["🔎 Find care", "⚖️ How we decide", "📊 Data quality"])

with tab_find:
    services = search.services
    if not services:
        st.error("No service evidence is loaded yet.")
        st.stop()
    if "service" not in st.session_state:
        st.session_state.service = services[0]
        st.session_state.place = ""

    st.markdown("**Try:**")
    examples = [(s, p) for s, p in EXAMPLES if s in services]
    for col, (svc, place) in zip(st.columns(max(1, len(examples))), examples):
        if col.button(f"{service_label(svc)} near {place}", use_container_width=True):
            st.session_state.service, st.session_state.place = svc, place

    with st.form("search"):
        c1, c2, c3 = st.columns([2, 3, 1])
        service = c1.selectbox("What care is needed?", services, format_func=service_label, key="service")
        c2.text_input("Where is the patient?", key="place",
                      placeholder="City or district (Patna), PIN code (800001), or coordinates (25.59, 85.13)")
        k = c3.number_input("Results", 1, 10, 5)
        st.form_submit_button("Find care", type="primary", use_container_width=True)

    reviewer, reviewer_note = get_reviewer()
    use_ai = st.toggle("AI review of the top results", value=reviewer is not None, disabled=reviewer is None,
                       help="A language model reads each shown facility's own text and gives a second opinion "
                            "with a verbatim quote. It never changes the ranking.")
    if reviewer_note:
        st.caption(reviewer_note)

    place = st.session_state.place
    if not place:
        st.info("Choose a service and enter a place to see recommendations.")
    else:
        loc = places.resolve(place)
        if loc is None:
            sugg = places.suggest(place)
            st.error(f"Couldn't find “{place}”." + (f" Did you mean: {', '.join(sugg)}?" if sugg else
                                                      " Try a nearby city, a 6-digit PIN code, or coordinates."))
        else:
            res = search.run(service, loc.lat, loc.lon, k=int(k))
            candidates = res["pool"] + res["weak"] + ([res["nearest_strong"]] if res["nearest_strong"] else [])
            with st.spinner("Getting road distances…"):
                roads, road_error = road_km((loc.lat, loc.lon),
                                            {c["unique_id"]: (c["latitude"], c["longitude"]) for c in candidates})
            res = apply_road(res, roads, road_error)
            reviews = {}
            if use_ai and reviewer is not None:
                to_review = (res["pool"] + ([res["nearest_strong"]] if res["nearest_strong"] else [])
                             + res["weak"])
                jobs = [((c["unique_id"], service), service, search.facility_lines(c["unique_id"])) for c in to_review]
                with st.spinner(f"AI is reading {len(jobs)} listings…"):
                    reviews = review_many(reviewer, jobs)
                res = apply_reviews(res, reviews, service)
            else:
                res = dict(res, removed=[], changes=[])
            pol, label = res["policy"], service_label(service)
            radius = f"{pol['nearby_km']:.0f} km (straight-line search radius)"
            st.markdown(f"📍 **Searching around {esc(loc.label)}** — {esc(loc.precision)}")
            if loc.alternatives:
                st.caption("Other places with this name: " + "; ".join(loc.alternatives) +
                           ". Add the state, e.g. “Aurangabad, Bihar”.")

            if any(c["evidence_level"] == "strong" for c in res["main"]):
                st.success(f"Found facilities within {radius} whose listings explicitly describe **{label}**.")
            elif res["main"] or res["weak"]:
                st.warning(f"Within {radius}, no listing explicitly describes {label}; only weaker evidence was found. "
                           "Call ahead to confirm.")
            elif res["status"] in ("no_nearby_strong_evidence", "no_usable_strong_evidence"):
                st.error(f"No facility within {radius} has evidence of {label} in this dataset. "
                         "That does not prove none exists.")
            else:
                st.error(f"No usable evidence of {label} in this dataset. {res['message']}")

            def direction(ch):
                if not ch["road_rank"]:
                    return f"dropped out of the top {res['k']}"
                if not ch["straight_rank"]:
                    return "moved into the list"
                return "moved up" if ch["road_rank"] < ch["straight_rank"] else "moved down"
            # Directions only: the AI review may re-rank again, and the cards show the final positions
            moved = "".join(f"  \n- **{esc(ch['name'])}**: {direction(ch)} "
                            f"({'≈ ' if ch['estimated'] else ''}{ch['road_km']:.1f} km by road)"
                            for ch in res["road_changes"])
            moved = f" Compared with straight-line distance, ranking by road distance changed the order:{moved}" if moved else ""
            if res.get("road_error"):
                st.caption(f"🛣️ Road routing is unavailable right now ({esc(res['road_error'])}). Road distances are "
                           f"**estimated** as {ROAD_FACTOR} × the straight-line distance and marked as estimates.{moved}")
            else:
                n_est = sum(bool(c.get("road_estimated")) for c in res["main"] + res["weak"])
                st.caption(f"🛣️ Distances are driving distances on the {SOURCE_NAME}, without live traffic; the ranking "
                           "uses them in the distance factor."
                           + (f" {n_est} shown facilit{'y has' if n_est == 1 else 'ies have'} no road route, so "
                              f"{'its' if n_est == 1 else 'their'} distance is estimated ({ROAD_FACTOR} × straight line)."
                              if n_est else "") + moved)
            if res["changes"]:
                lines = []
                for ch in res["changes"]:
                    if ch["final"] == "none":
                        move = "removed from the list"
                    elif ch["final_rank"]:
                        move = f"moved from #{ch['rule_rank']} to #{ch['final_rank']}" if ch["rule_rank"] \
                            else f"moved into the list at #{ch['final_rank']}"
                    else:
                        move = "moved to the specialty-tag section" if ch["final"] == "weak" else "moved out of the top results"
                    lines.append(f"- **{esc(ch['name'])}**: {ch['rule']} → **{ch['final']}**, {move}")
                scores = ", ".join(f"{lvl} {v:.2f}" for lvl, v in LEVEL_SCORE.items())
                st.info("🤖 **The AI review changed the rating of "
                        f"{len(res['changes'])} facilit{'y' if len(res['changes']) == 1 else 'ies'}, so the list was re-ranked.**  \n"
                        + "  \n".join(lines) + "\n\n"
                        "How: the same formula is used (evidence score × location weight × distance factor). "
                        f"Only the evidence score changes, to the score the rules give the recommended rating ({scores}); "
                        "a facility rated *none* is removed. Only the facilities shown here (plus a few more) were reviewed.")

            items = [(str(i), c) for i, c in enumerate(res["main"], 1)]
            if res["nearest_strong"]:
                items.append((str(len(res["main"]) + 1), res["nearest_strong"]))
            items += [(f"W{i}", c) for i, c in enumerate(res["weak"], 1)]
            picked_uid = None
            if items:
                picked = render_map(loc, items, key=f"map|{service}|{place}")
                picked_uid = picked[1]["unique_id"] if picked else None
                st.caption("🔵 search point · 🟢 strong · 🟠 moderate · 🔴 weak evidence. "
                           "Numbers match the cards below (W = specialty tag only). "
                           "Hover a point for its name; click it to jump to its details.")
            for i, c in enumerate(res["main"], 1):
                render_card(i, c, pol, selected=c["unique_id"] == picked_uid,
                            review=reviews.get((c["unique_id"], service)))
            if res["nearest_strong"]:
                st.subheader(f"Nearest strong evidence (outside {radius})")
                st.caption("Farther away. Shown for planning, not as an emergency routing instruction.")
                c = res["nearest_strong"]
                render_card(len(res["main"]) + 1, c, pol, selected=c["unique_id"] == picked_uid,
                            review=reviews.get((c["unique_id"], service)))
            if res["weak"]:
                weak_picked = any(c["unique_id"] == picked_uid for c in res["weak"])
                with st.expander(f"Facilities with only a specialty tag nearby — {res['n_weak']} (unverified)",
                                 expanded=weak_picked):
                    st.caption("These listings only mention a related department. They are not evidence the service exists.")
                    for i, c in enumerate(res["weak"], 1):
                        render_card(f"W{i}", c, pol, compact=True, selected=c["unique_id"] == picked_uid,
                                    review=reviews.get((c["unique_id"], service)))
            if picked_uid:
                scroll_to(anchor_id(picked_uid))

            if res["removed"]:
                with st.expander(f"Removed after the AI review — {len(res['removed'])}"):
                    st.caption("The AI review found the listing does not support this service now "
                               "(for example it is planned, referred elsewhere or unavailable).")
                    for c in res["removed"]:
                        render_card("–", c, pol, compact=True, review=reviews.get((c["unique_id"], service)))

            with st.expander("What was searched and what was left out"):
                st.markdown(f"- {res['n_evidence_rows']} facility records mention {label}; "
                            f"{res['eligible']} passed all checks; {res['nearby_strong_count']} have strong evidence within {radius}.")
                for reason, n in sorted(res["excluded"].items(), key=lambda x: -x[1]):
                    st.markdown(f"- Excluded {n}: {reason}")
                amb = res["diagnostics"].get("ambiguous_facility_ids_excluded")
                if amb:
                    st.markdown(f"- Excluded {amb} facility IDs shared by more than one record")

    st.caption("CareFinder ranks text found on public listings. It cannot confirm that a service is available today. "
               "**Always call the facility before sending a patient.** In an emergency, call 108.")

with tab_how:
    st.header("How a recommendation is made")
    st.markdown(f"""
1. **Find the place.** A city or district resolves to the median location of trusted facilities there; a PIN code to the
   centre of that PIN area; coordinates are used as entered.
2. **Find evidence for the service** (notebook `03_service_evidence`, rules in `config/services.yaml`).
   Each facility's own procedures, capabilities, equipment and description are searched for service-specific phrases;
   sentences saying "planned", "referred", "not available" and similar are skipped.
   - **Strong**: the service itself is stated (e.g. "Provides dialysis treatment").
   - **Moderate**: only related equipment is listed (e.g. "Dialysis machines").
   - **Weak**: only a specialty tag (e.g. "nephrology"). A tag is not proof of the service.

   A human audit of 50 sampled facility–service pairs agreed with these rules in **42/50 (84%)** cases.
3. **Check the record** (notebook `01_clean_facilities`). Corrupted records, unusable coordinates and IDs shared by
   several records are excluded; other issues lower the location weight and are shown as warnings.
4. **Check the quote.** Every quote shown must appear word for word in the facility's own record.
5. **Score and rank** (`ranking.py`, notebook `04_ranking`). Candidates are found within a straight-line radius;
   the app then gets **driving distances** on the OpenStreetMap road network (OSRM, no live traffic) and uses them
   in the distance factor. If the routing service is unreachable or has no route, the road distance is estimated as
   1.4 × the straight-line distance and labelled "estimated", with the straight-line distance shown next to it.

   `score = evidence score × location weight × exp(−distance ÷ decay)`

| Location confidence | Weight |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in ranking.LOCATION_WEIGHTS.items()) + f"""

| | Routine services | Urgent services |
|---|---|---|
| "Nearby" radius | 60 km | 30 km |
| Distance decay | 75 km | 25 km |

6. **AI review and re-ranking.** A language model reads each shown facility's own text and gives its own rating
   with a verbatim quote (a quote not found in the listing is discarded). If it disagrees with the rules, both
   ratings and quotes are shown with a suggestion: an AI *downgrade* is accepted (it reads context such as
   "planned" or "inpatients only"); an AI *upgrade* is accepted only if its quote says the service is available or
   offered, not just that a machine or department exists. The list is then re-ranked with the **same formula**,
   replacing only the evidence score by the recommended rating's score (strong 0.85, moderate 0.55, weak 0.20);
   *none* removes the facility. Every change is listed above the results. In a 50-pair pilot the AI agreed with
   human labels 44/50 times versus 42/50 for the rules.
7. **Never hide the gap.** If nothing nearby has strong evidence, the page says so and separately shows the nearest
   facility that does. Weak-evidence facilities are collapsed into their own section.

**Trade-offs:** a close facility with moderate evidence can outrank a distant one with strong evidence, but it is
always labelled. Distances are road distances without live traffic. All weights are transparent prototype choices.
""")

with tab_data:
    st.header("How much can the data be trusted?")
    st.markdown("The dataset was extracted from web pages, so every field is treated as a claim. Each issue below is a "
                "rule in `01_clean_facilities`. Affected facilities are down-weighted, warned about or excluded, "
                "and the exclusions are counted on every search.")
    flags = {
        "Corrupted record (no usable name)": "flag_row_corrupted",
        "Map point >50 km from the rest of its city": "flag_coords_far_from_city",
        "Coordinates missing": "flag_coords_missing",
        "Coordinates outside India": "flag_coords_outside_india",
        "PIN code disagrees with state": "flag_pin_state_mismatch",
        "Phone shared with 3+ other facilities": "flag_phone_shared",
        "Possible duplicate (same name, same city)": "flag_possible_duplicate",
        "Small clinic listing ≥20 specialties": "flag_capability_bloat",
    }
    rows = [{"Issue": k, "Facilities": int(fac_df[v].fillna(False).astype(bool).sum())}
            for k, v in flags.items() if v in fac_df]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Facilities in dataset", f"{len(fac_df):,}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with c2:
        lc = fac_df["location_confidence"].value_counts().reindex(["high", "medium", "low"]).fillna(0).astype(int)
        st.markdown("**Location confidence**")
        st.dataframe(lc.rename("Facilities").rename_axis("Level").reset_index(), hide_index=True,
                     use_container_width=True)
        ev = pd.DataFrame([{"Service": service_label(s), "Level": r["evidence_level"]}
                           for s, rs in search.ranker.evidence.items() for r in rs])
        if len(ev):
            st.markdown("**Evidence found per service**")
            piv = ev.pivot_table(index="Service", columns="Level", aggfunc="size", fill_value=0)
            st.dataframe(piv.reindex(columns=[c for c in ["strong", "moderate", "weak"] if c in piv.columns]),
                         use_container_width=True)
