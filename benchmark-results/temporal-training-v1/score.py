"""Summarize matched automatic temporal checks."""

import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def main():
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    summary = {}
    outcomes = {}
    cohorts = {
        "all": lambda row: True,
        "dates": lambda row: row["domain"] == "temporal-grant",
        "remaining": lambda row: row["role"] == "remaining_days",
        "additional": lambda row: row["role"] == "additional_days",
        "shipping": lambda row: row["domain"] == "shipping",
    }
    for adapter in ("baseline", "temporal"):
        summary[adapter] = {}
        for mode in ("raw", "roles"):
            rows = [row for row in results[adapter] if row["mode"] == mode]
            assert len(rows) == len({(row["id"], row["seed"]) for row in rows}) == 40
            metrics = {"median_seconds": round(median(row["seconds"] for row in rows), 3)}
            passed = set()
            for name, choose in cohorts.items():
                subset = [row for row in rows if choose(row)]
                accepted = {(row["id"], row["seed"]) for row in subset
                            if row.get("checks") and all(row["checks"].values())}
                metrics[name] = {"passed": len(accepted), "total": len(subset)}
                if name == "all":
                    passed = accepted
            summary[adapter][mode] = metrics
            outcomes[adapter, mode] = passed
    comparisons = {}
    for before, after in ((("baseline", "raw"), ("temporal", "raw")),
                          (("baseline", "raw"), ("baseline", "roles")),
                          (("temporal", "raw"), ("temporal", "roles"))):
        old, new = outcomes[before], outcomes[after]
        comparisons[f"{'/'.join(before)} -> {'/'.join(after)}"] = {
            "new_pass": len(new - old), "regression": len(old - new), "both_pass": len(old & new),
        }
    report = {"automatic": summary, "paired_changes": comparisons}
    (HERE / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
