"""Entirely SYNTHETIC records for local UI development and tests. Not real facilities.

Columns and quote format mirror facilities_clean and facility_service_evidence.
"""

import json

import pandas as pd


def _fac(uid, name, city, state, pin, lat, lon, conf="high", ftype="hospital",
         procedure=(), equipment=(), capability=(), specialties=(), **flags):
    row = dict(
        unique_id=uid, name=name, facility_type=ftype, operator_type="private",
        city_clean=city, state_clean=state, pin_code=pin, latitude=lat, longitude=lon,
        location_confidence=conf, officialWebsite="https://example.org",
        officialPhone="+91 00000 00000", phone_shared_by_n=0,
        source_urls=json.dumps(["https://example.org/listing"]),
        description=f"{name} is a synthetic test record.",
        procedure_list=json.dumps(list(procedure)), equipment_list=json.dumps(list(equipment)),
        capability_list=json.dumps(list(capability)), specialties_list=json.dumps(list(specialties)),
        flag_row_corrupted=False, flag_coords_missing=False, flag_coords_outside_india=False,
        flag_coords_far_from_city=False, flag_pin_state_mismatch=False, flag_possible_duplicate=False,
        flag_phone_shared=False, flag_capability_bloat=False, quality_notes="",
    )
    row.update(flags)
    return row


def _ev(uid, service, level, score, quotes, n_direct=0, n_supporting=0):
    return dict(unique_id=uid, service=service, evidence_level=level, evidence_score=score,
                quotes=json.dumps([{"field": f if f == "description" else f + "_list", "sentence": s}
                                   for f, s in quotes]),
                n_direct=n_direct, n_supporting=n_supporting)


def build_sample():
    fac = []
    # Patna cluster (city centre ~25.61, 85.14)
    for i in range(4):
        fac.append(_fac(f"pat-filler-{i}", f"Sample Patna Clinic {i}", "Patna", "Bihar", "800001",
                        25.60 + i * 0.01, 85.13 + i * 0.01, specialties=["familyMedicine"]))
    fac += [
        _fac("pat-strong", "Sample Patna Kidney Hospital", "Patna", "Bihar", "800001", 25.62, 85.15,
             procedure=["Maintenance hemodialysis is offered six days a week."],
             equipment=["12 hemodialysis machines"], specialties=["nephrology"]),
        _fac("pat-moderate", "Sample Patna Care Hospital", "Patna", "Bihar", "800003", 25.63, 85.10,
             equipment=["Dialysis machines"], specialties=["nephrology"]),
        _fac("pat-avail", "Sample Patna Kidney Centre", "Patna", "Bihar", "800002", 25.64, 85.16,
             equipment=["Dialysis facility available on site"], specialties=["nephrology"]),
        # Close in a straight line but across the river: the road route is much longer
        _fac("pat-detour", "Sample Patna Riverside Hospital", "Patna", "Bihar", "800001", 25.632, 85.142,
             procedure=["Provides dialysis treatment"], specialties=["nephrology"]),
        _fac("pat-planned", "Sample Patna New Hospital", "Patna", "Bihar", "800001", 25.615, 85.145,
             procedure=["Dialysis unit being set up"], specialties=["nephrology"]),
        _fac("pat-weak", "Sample Patna Multispeciality", "Patna", "Bihar", "800002", 25.59, 85.12,
             specialties=["nephrology", "radiology"]),
        _fac("pat-medium-loc", "Sample Patna Government Hospital", "Patna", "Bihar", "800004", 25.70, 85.30,
             conf="medium", procedure=["Hemodialysis unit for inpatients."],
             flag_pin_state_mismatch=True, quality_notes="PIN code does not match the stated state"),
        _fac("bad-row", None, None, None, None, None, None, conf="low", flag_row_corrupted=True,
             flag_coords_missing=True),
    ]
    # Muzaffarpur, about 70 km from Patna: strong evidence outside the Patna radius for rural queries
    fac += [_fac(f"muz-{i}", f"Sample Muzaffarpur Clinic {i}", "Muzaffarpur", "Bihar", "842001",
                 26.12 + i * 0.005, 85.39, specialties=["familyMedicine"]) for i in range(3)]
    fac.append(_fac("muz-strong", "Sample Muzaffarpur Medical College", "Muzaffarpur", "Bihar", "842001",
                    26.125, 85.385, procedure=["Hemodialysis unit with 8 beds."], phone_shared_by_n=12,
                    flag_phone_shared=True))
    # Jaipur cluster for MRI
    fac += [_fac(f"jai-{i}", f"Sample Jaipur Clinic {i}", "Jaipur", "Rajasthan", "302001",
                 26.91 + i * 0.01, 75.79, specialties=["radiology"]) for i in range(3)]
    fac.append(_fac("jai-mri", "Sample Jaipur Imaging Centre", "Jaipur", "Rajasthan", "302001", 26.92, 75.80,
                    equipment=["3 Tesla MRI scanner"], procedure=["MRI scans of brain and spine."]))
    fac.append(_fac("jai-quote-bad", "Sample Jaipur Hospital", "Jaipur", "Rajasthan", "302001", 26.90, 75.78,
                    procedure=["General surgery."]))

    evi = [
        _ev("pat-strong", "dialysis", "strong", 0.9,
            [("procedure", "Maintenance hemodialysis is offered six days a week."),
             ("equipment", "12 hemodialysis machines")], n_direct=1, n_supporting=1),
        _ev("pat-moderate", "dialysis", "moderate", 0.55, [("equipment", "Dialysis machines")], n_supporting=1),
        _ev("pat-avail", "dialysis", "moderate", 0.55, [("equipment", "Dialysis facility available on site")],
            n_supporting=1),
        _ev("pat-detour", "dialysis", "strong", 0.85, [("procedure", "Provides dialysis treatment")], n_direct=1),
        # The keyword rules miss "being set up" here; the AI review should catch it
        _ev("pat-planned", "dialysis", "strong", 0.85, [("procedure", "Dialysis unit being set up")], n_direct=1),
        _ev("pat-weak", "dialysis", "weak", 0.2, [("specialties", "nephrology")]),
        _ev("pat-medium-loc", "dialysis", "strong", 0.85,
            [("procedure", "Hemodialysis unit for inpatients.")], n_direct=1),
        _ev("bad-row", "dialysis", "strong", 0.85, [("procedure", "x")], n_direct=1),
        _ev("muz-strong", "dialysis", "strong", 0.85, [("procedure", "Hemodialysis unit with 8 beds.")], n_direct=1),
        _ev("jai-mri", "mri", "strong", 0.9,
            [("procedure", "MRI scans of brain and spine."), ("equipment", "3 Tesla MRI scanner")],
            n_direct=1, n_supporting=1),
        _ev("jai-0", "mri", "weak", 0.2, [("specialties", "radiology")]),
        _ev("jai-quote-bad", "mri", "strong", 0.85, [("procedure", "MRI available 24x7.")], n_direct=1),
    ]
    loc = pd.DataFrame([
        dict(kind="pin", name="800001", state="Bihar", lat=25.61, lon=85.14, n_points=3, source="synthetic PIN"),
        dict(kind="district", name="Darbhanga", state="Bihar", lat=26.15, lon=85.90, n_points=40,
             source="synthetic district"),
    ])
    return pd.DataFrame(fac), pd.DataFrame(evi), loc
