"""Prepare anonymous reviews and compare completed field judgments."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = ("exaone", "gemma", "exaone-lora", "gemma-lora")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(allow_pending=False):
    cases = {
        case["id"]: case
        for case in map(json.loads, (ROOT / "training/test/field-conditions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines())
    }
    results = read_json(HERE / "results.json")
    expected = {(case_id, seed) for case_id in cases for seed in (42, 123)}
    indexed = {label: {(row["id"], row["seed"]): row for row in results[label]["rows"]}
               for label in LABELS if label in results}
    if any(rows.keys() != expected for rows in indexed.values()) or (not allow_pending and len(indexed) != len(LABELS)):
        raise ValueError("Each model must cover the same requests and seeds")
    return cases, indexed


def anonymize(indexed):
    rng = random.Random(211)
    records = []
    for case_id, seed in sorted(indexed[LABELS[0]]):
        order = list(LABELS)
        rng.shuffle(order)
        for label in order:
            row = indexed.get(label, {}).get((case_id, seed))
            if row is not None:
                records.append((f"{case_id}@{seed}/{order.index(label) + 1}", label, row))
    return records


def summarize(cases, indexed, records, decisions):
    accepted = {label: set() for label in LABELS}
    failures = {label: Counter() for label in LABELS}
    for review_id, label, row in records:
        decision = decisions[review_id]
        if decision == "pass":
            if not row.get("checks", {}).get("schema"):
                raise ValueError(f"{review_id}: accepted output violates the schema")
            accepted[label].add((row["id"], row["seed"]))
        else:
            failures[label][decision] += 1
    summary = {}
    for label in LABELS:
        keys = accepted[label]
        domain_passes = Counter(cases[case_id]["domain"] for case_id, _ in keys)
        domain_totals = Counter(cases[case_id]["domain"] for case_id, _ in indexed[label])
        summary[label] = {
            "usable": len(keys),
            "word": sum(cases[case_id]["one_word"] for case_id, _ in keys),
            "sentence": sum(not cases[case_id]["one_word"] for case_id, _ in keys),
            "korean": sum(cases[case_id]["request"]["output_language"] == "Korean" for case_id, _ in keys),
            "english": sum(cases[case_id]["request"]["output_language"] == "English" for case_id, _ in keys),
            "pairs": sum((case_id[:-1] + "2", seed) in keys for case_id, seed in keys if case_id.endswith("-1")),
            "median_seconds": round(median(row["seconds"] for row in indexed[label].values()), 3),
            "total_seconds": round(sum(row["seconds"] for row in indexed[label].values()), 2),
            "domains": {
                domain: {"usable": domain_passes[domain], "total": total}
                for domain, total in sorted(domain_totals.items())
            },
            "failures": dict(failures[label]),
        }
    comparisons = {}
    for left, right in (("exaone", "gemma"), ("exaone-lora", "gemma-lora"),
                        ("exaone-lora", "exaone"), ("gemma-lora", "gemma")):
        comparisons[f"{left} vs {right}"] = {
            "left_only": len(accepted[left] - accepted[right]),
            "right_only": len(accepted[right] - accepted[left]),
            "both": len(accepted[left] & accepted[right]),
        }
    return {"models": summary, "paired_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "score"))
    args = parser.parse_args()
    cases, indexed = collect(allow_pending=args.action == "prepare")
    records = anonymize(indexed)
    blind = [
        {"review_id": review_id, "case_id": row["id"], "seed": row["seed"],
         "reference": cases[row["id"]]["target"], "output": row.get("output"),
         "schema_valid": row.get("checks", {}).get("schema", False)}
        for review_id, _, row in records
    ]
    payload = json.dumps(blind, ensure_ascii=False, sort_keys=True).encode("utf-8")
    revision = hashlib.sha256(payload).hexdigest()
    if args.action == "prepare":
        write_json(HERE / "blind.json", {"sha256": revision, "rows": blind})
        print(f"Prepared {len(blind)} anonymous outputs")
        return
    review = read_json(HERE / "decisions.json")
    if review["sha256"] != revision or review["decisions"].keys() != {row["review_id"] for row in blind}:
        raise ValueError("Review must match the current outputs and cover every response")
    summary = summarize(cases, indexed, records, review["decisions"])
    write_json(HERE / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
