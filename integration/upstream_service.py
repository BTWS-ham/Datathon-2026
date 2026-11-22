# Snapshot from 03_service_evidence.ipynb, cell 3, commit 6b4104bef87409e4fc000cc69a7e0e7905f7134b.
# Team upstream logic preserved verbatim for reproducible Part C integration.

import json
import re


def phrase_pattern(terms):
    # Word boundaries avoid substring matches such as "MRI" in longer names.
    phrases = [r"\s+".join(re.escape(word) for word in t.split())
               for t in sorted(terms, key=len, reverse=True)]
    return re.compile(r"(?<!\w)(?:" + "|".join(phrases) + r")(?!\w)", re.I)


def compile_rules(config):
    rules = {}
    for service, terms in config["services"].items():
        rules[service] = {
            key: phrase_pattern(terms[key])
            for key in ("direct", "supporting", "specialties")
        }
        rules[service]["red_flags"] = phrase_pattern(config["red_flags"])
        rules[service]["service_veto"] = phrase_pattern(terms["red_flags"])
    return rules


def passages(value):
    # Preserve each quoted substring exactly, including capitalization.
    if isinstance(value, list):
        for item in value:
            yield from passages(item)
    elif isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                items = json.loads(value)
            except ValueError:
                return
            if isinstance(items, list):
                yield from passages(items)
            return
        for line in re.split(r"(?<=[.!?])\s+|[\r\n]+", value):
            if line.strip():
                yield line.strip()


def score_record(source_records, rules):
    evidence = []
    snippets = set()
    for record in source_records:
        for field, value in json.loads(record).items():
            snippets.update((field, line) for line in passages(value))

    for service, rule in rules.items():
        hits = []
        blocked = False
        for field, line in sorted(snippets):
            # Remove equipment phrases before testing direct terms. A machine name
            # alone does not document that the facility performs the procedure.
            remainder, equipment_hits = rule["supporting"].subn(" ", line)
            direct = bool(rule["direct"].search(remainder))
            specialty = bool(rule["specialties"].search(line))
            if rule["service_veto"].search(line):
                blocked = True
                continue
            if rule["red_flags"].search(line):
                # A service-specific contradiction vetoes even a positive duplicate.
                if direct or equipment_hits:
                    blocked = True
                continue
            if direct and field in {"procedure", "capability", "description"}:
                kind = "direct"
            elif equipment_hits or (field == "equipment" and direct):
                kind = "supporting"
            elif specialty:
                kind = "specialty"
            else:
                continue
            hits.append((kind, field, line))
        if blocked or not hits:
            continue
        direct_count = sum(kind == "direct" for kind, _, _ in hits)
        support_count = sum(kind == "supporting" for kind, _, _ in hits)
        if direct_count:
            level, score = "strong", 0.90 if support_count else 0.85
        elif support_count:
            level, score = "moderate", 0.55
        else:
            level, score = "weak", 0.20
        priority = {"direct": 0, "supporting": 1, "specialty": 2}
        hits.sort(key=lambda h: (priority[h[0]], h[1], h[2]))
        quotes = [(field, line) for _, field, line in hits]
        evidence.append((service, level, score, quotes, direct_count, support_count))
    return evidence
