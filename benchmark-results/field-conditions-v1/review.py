"""Summarize reviewed outputs from the fixed field-conditions run."""

import hashlib
import json
from collections import Counter
from pathlib import Path

FOLDER = Path(__file__).resolve().parent


def main():
    # Git normalizes JSON text to LF, so the fingerprint must ignore checkout line endings.
    source = (FOLDER / "results.json").read_text(encoding="utf-8").encode("utf-8")
    evaluation = json.loads(source)
    review = json.loads((FOLDER / "review.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source).hexdigest() != review["results_sha256"]:
        raise ValueError("Outputs have changed. Review the new run before scoring it.")

    for name, result in evaluation.items():
        rows = result["rows"]
        candidates = {f"{row['id']}@{row['seed']}" for row in rows if row["pass"]}
        decisions = review["decisions"][name]
        if candidates != decisions.keys():
            raise ValueError(f"{name}: review must cover every automatic pass")
        accepted = {key for key, reason in decisions.items() if reason == "usable"}
        pairs = 0
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            if first["seed"] != second["seed"] or first["id"].rsplit("-", 1)[0] != second["id"].rsplit("-", 1)[0]:
                raise ValueError(f"{name}: invalid contrast pair")
            pairs += all(f"{row['id']}@{row['seed']}" in accepted for row in (first, second))
        print(f"{name}: {len(accepted)}/{len(rows)} confirmed usable, {pairs}/{len(rows) // 2} confirmed pairs")
        print(dict(Counter(decisions.values())))


if __name__ == "__main__":
    main()
