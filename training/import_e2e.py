"""Select English restaurant descriptions with explicit matches for every input fact."""
import csv
import json
import re
import sys
from pathlib import Path


REVISION = "164acdf986146d76f18893b3de9488725deda4c2"
SOURCE = "https://github.com/tuetschek/e2e-cleaning"
SLOTS = {"name": "restaurant_name", "food": "cuisine", "area": "area", "near": "landmark",
         "eatType": "venue_type"}


def main():
    source = Path(sys.argv[1])
    output = Path(__file__).resolve().parent / "source/e2e-selected.jsonl"
    rows = []
    restaurants = set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=2):
            facts = {key.strip(): value.strip() for key, value in re.findall(r"([\w ]+)\[([^\]]*)\]", row["mr"])}
            if not {"name", "food", "area"} <= facts.keys() or facts.keys() - SLOTS.keys():
                continue
            target = row["ref"].strip()
            if facts["name"] in restaurants or len(target) > 180:
                continue
            if not all(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", target, re.I)
                       for value in facts.values()):
                continue
            restaurants.add(facts["name"])
            rows.append({
                "id": f"e2e-{index}", "split_group": "restaurant-e2e",
                "source": SOURCE, "source_revision": REVISION,
                "source_id": f"cleaned-data/train-fixed.no-ol.csv:{index}", "license": "CC-BY-SA-4.0",
                "request": {
                    "field_name": "description", "path": "$.description", "target_type": "string",
                    "field_schema": {
                        "type": "string", "maxLength": 180,
                        "description": "Write a short restaurant description that mentions every fact in the context.",
                    },
                    "context": {SLOTS[key]: value for key, value in facts.items()},
                    "output_language": "English",
                },
                "target": target,
            })
            if len(rows) == 16:
                break
    if len(rows) != 16:
        raise ValueError(f"Expected 16 eligible restaurants, found {len(rows)}.")
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
