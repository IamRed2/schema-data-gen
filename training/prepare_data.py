import json
from pathlib import Path

from jsonschema import Draft202012Validator


DATA_DIR = Path(__file__).resolve().parent


def make_row(identifier, group, field, schema, context, language, text):
    return {
        "id": identifier,
        "split_group": group,
        "source": "project-authored",
        "request": {
            "field_name": field, "path": f"$.{field}", "field_schema": schema,
            "target_type": schema["type"], "context": context, "output_language": language,
        },
        "target": text,
    }


def context_rows(schema, record, group, language, identifier, context=None, field="value", path="$"):
    context = {} if context is None else context
    if schema["type"] == "object":
        previous = {}
        for name, child in schema["properties"].items():
            yield from context_rows(
                child, record[name], group, language, identifier, context | previous, name, f"{path}.{name}"
            )
            previous[name] = record[name]
    elif schema["type"] == "array":
        for index, value in enumerate(record):
            yield from context_rows(
                schema["items"], value, group, language, identifier, context, field, f"{path}[{index}]"
            )
    elif schema.get("description"):
        row = make_row(f"{identifier}-{path}", group, field, schema, context, language, record)
        row["request"]["path"] = path
        yield row


def main():
    examples = json.loads((DATA_DIR / "source/record-examples.json").read_text(encoding="utf-8"))
    splits = {"train": [], "test": []}
    domains = ("support", "support", "software", "review", "notification", "internal-support")
    for index, example in enumerate(examples):
        domain = domains[index // 2]
        split = "test" if domain == "internal-support" else "train"
        for language, records in example["answers"].items():
            for record_index, record in enumerate(records):
                for name, schema in example["schema"]["properties"].items():
                    if schema.get("type") != "string" or "enum" in schema:
                        continue
                    row = make_row(
                        f"record-{index}-{language}-{record_index}-{name}", domain, name, schema,
                        {key: value for key, value in record.items() if key != name},
                        language, record[name],
                    )
                    splits[split].append(row)

    # These examples exercise the same scalar-to-text format used during inference.
    scalar_cases = [
        ("train", "person", "name", {"type": "string"}, "서지민", "Jamie Brooks"),
        ("train", "person", "age", {"type": "integer", "minimum": 18, "maximum": 90}, "27", "32"),
        ("train", "person", "is_active", {"type": "boolean"}, "true", "false"),
        ("train", "person", "rating", {"type": "number", "minimum": 0, "maximum": 5}, "4.2", "3.5"),
        ("train", "person", "deleted_at", {"type": "null"}, "null", "null"),
        ("train", "tools", "label", {"type": "string", "description": "주방용품 이름 한 단어."}, "주전자", "kettle"),
        ("train", "tools", "label", {"type": "string", "description": "주방용품의 결함을 한 문장으로 기록한다."},
         "주전자 손잡이에 금이 가 있습니다.", "The kettle handle is cracked."),
        ("test", "garden", "plant_name", {"type": "string"}, "바질", "basil"),
        ("test", "garden", "pot_count", {"type": "integer", "minimum": 1, "maximum": 10}, "3", "5"),
        ("test", "garden", "needs_water", {"type": "boolean"}, "true", "false"),
        ("test", "garden", "height_cm", {"type": "number", "minimum": 0}, "12.5", "16.2"),
        ("test", "garden", "harvested_at", {"type": "null"}, "null", "null"),
        ("train", "person", "age", {"type": "integer", "minimum": 6, "maximum": 12}, "8", "11"),
        ("train", "person", "age", {"type": "integer", "minimum": 65, "maximum": 80}, "67", "74"),
        ("train", "person", "age", {"type": "integer", "minimum": 18, "maximum": 19}, "18", "19"),
        ("train", "weather", "temperature", {"type": "number", "minimum": -15, "maximum": -2}, "-9.5", "-3"),
        ("train", "weather", "temperature", {"type": "number", "minimum": 24, "maximum": 29}, "24.5", "28"),
        ("train", "weather", "humidity", {"type": "integer", "exclusiveMinimum": 39, "exclusiveMaximum": 44}, "40", "43"),
        ("train", "weather", "rainfall_mm", {"type": "number", "minimum": 0.5, "maximum": 3, "multipleOf": 0.5}, "1.5", "2.5"),
        ("train", "shipping", "parcel_count", {"type": "integer", "minimum": 2}, "3", "6"),
        ("train", "shipping", "parcel_count", {"type": "integer", "maximum": 2}, "1", "2"),
        ("train", "shipping", "weight_kg", {"type": "number", "exclusiveMinimum": 0.5, "exclusiveMaximum": 1.5}, "0.75", "1.25"),
        ("train", "tools", "label", {"type": "string", "minLength": 3, "maxLength": 3, "description": "물건 이름 한 단어."}, "주전자", "cup"),
        ("train", "tools", "label", {"type": "string", "minLength": 5, "maxLength": 10, "description": "Short name for a kitchen item."}, "손잡이컵받침", "tea kettle"),
        ("test", "garden", "pot_count", {"type": "integer", "exclusiveMinimum": 1, "exclusiveMaximum": 4}, "2", "3"),
        ("test", "garden", "pot_count", {"type": "integer", "minimum": 20, "maximum": 24, "multipleOf": 2}, "20", "24"),
        ("test", "garden", "height_cm", {"type": "number", "exclusiveMinimum": 6.5, "maximum": 8}, "7.25", "8"),
        ("test", "garden", "plant_name", {"type": "string", "minLength": 4, "maxLength": 6, "description": "Name of an edible plant."}, "방울토마토", "chives"),
    ]
    for index, (split, group, field, schema, korean, english) in enumerate(scalar_cases):
        for language, target in (("Korean", korean), ("English", english)):
            splits[split].append(make_row(
                f"scalar-{index}-{language}", group, field, schema, {}, language, target
            ))

    source_dir = DATA_DIR / "source"
    e2e_file = source_dir / "e2e-selected.jsonl"
    splits["train"].extend(json.loads(line) for line in e2e_file.read_text(encoding="utf-8").splitlines())
    for case in json.loads((source_dir / "context-examples.json").read_text(encoding="utf-8")):
        schema = json.loads((source_dir / case["schema"]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for language, records in case["answers"].items():
            for index, record in enumerate(records):
                validator.validate(record)
                group = case["group"]
                splits[case["split"]].extend(
                    context_rows(schema, record, group, language, f"{group}-{language}-{index}")
                )
    for split, rows in splits.items():
        folder = DATA_DIR / split
        folder.mkdir(exist_ok=True)
        (folder / "values.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    manifest = {
        "sources": ["Project-authored examples", "Cleaned E2E NLG, CC-BY-SA-4.0"],
        "split_rule": "Domains grouped together; translated description variants share a split",
        "splits": {
            split: {
                "count": len(rows),
                "groups": sorted({row["split_group"] for row in rows}),
            }
            for split, rows in splits.items()
        },
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
