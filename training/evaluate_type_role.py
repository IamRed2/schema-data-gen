import argparse
import json
import random
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schema_generator.value_generator import parse_value, request_value


def run_case(case, task, runner, model, adapter, seed):
    infer_type = task == "type"
    schema = {"type": "string"} if infer_type else {"type": "integer", "description": case["description"]}
    request = {
        "field_name": case["field"], "path": f"$.{case['field']}", "field_schema": schema,
        "target_type": schema["type"], "output_language": case["language"], "context": {},
    }
    if infer_type:
        request["task"] = "Infer the described field's data type from its name and description."
        request["answer_labels"] = ["string", "integer", "number", "boolean", "object", "array", "null"]
        request["instruction"] = "Return exactly one English type label. target_type is the response format."
        request["field_description"] = case["description"]
        schema = {"enum": request["answer_labels"]}
    else:
        request["scenario"] = case["scenario"]

    output = None
    error = None
    try:
        output = request_value(runner, model, adapter, seed, request)
        value = parse_value(output, request["target_type"])
        Draft202012Validator(schema).validate(value)
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        error = str(exc)
        value = None
    return {
        "id": case["id"], "task": task, "request": request, "expected": case["expected"],
        "output": output, "valid": error is None, "correct": error is None and value == case["expected"],
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/model.gguf")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file.")
    cases = json.loads((ROOT / "training/test/type-role-cases.json").read_text(encoding="utf-8"))
    runner = ROOT / "runtime/llama.cpp/llama-completion.exe"
    seed = random.Random(42).randrange(2**32 - 1)
    rows = []
    for task in ("type", "role"):
        for case in cases[f"{task}_cases"]:
            row = run_case(case, task, runner, args.model.resolve(), args.adapter.resolve(), seed)
            rows.append(row)
            print(f"{row['id']}: valid={row['valid']}, correct={row['correct']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seed": seed, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")


if __name__ == "__main__":
    main()
