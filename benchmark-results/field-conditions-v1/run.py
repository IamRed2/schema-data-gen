"""Measure checkable requirements on held-out field requests."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from schema_generator.llama_server_backend import load_server_backend

LOCAL = ROOT / ".training"
EXAONE = LOCAL / "exaone-4.0-1.2b-q4.gguf"
MODELS = {
    "exaone": (EXAONE, None),
    "exaone-lora": (EXAONE, LOCAL / "exaone-docstyle-lora.gguf"),
    "gemma": (LOCAL / "gemma-3-1b-it-Q4_K_M.gguf", None),
    "gemma-lora": (LOCAL / "gemma-3-1b-it-Q4_K_M.gguf", LOCAL / "gemma-docstyle-lora.gguf"),
    "qwen": (ROOT / "models/qwen3.5-0.8b-q4.gguf", None),
}


def mentions(output, phrase):
    escaped = re.escape(phrase.casefold())
    if phrase.isdecimal():
        escaped = rf"(?<!\d){escaped}(?!\d)"
    return re.search(escaped, output.casefold()) is not None


def checks(case, output):
    request = case["request"]
    hangul = bool(re.search("[가-힣]", output))
    return {
        "schema": Draft202012Validator(request["field_schema"]).is_valid(output),
        "value_only": not output.lstrip().startswith(("{", "[", "```"))
        and not re.match(rf"\s*{re.escape(request['field_name'])}\s*:", output, re.IGNORECASE),
        "language_proxy": hangul if request["output_language"] == "Korean"
        else not hangul and bool(re.search("[a-zA-Z]", output)),
        "required": all(any(mentions(output, form) for form in forms) for forms in case["required"]),
        "distractor_free": not any(mentions(output, term) for term in case["forbidden"]),
        "shape": not case["one_word"] or bool(re.fullmatch(r"[^\W\d_]+", output.strip())),
    }


def summarize(rows):
    summary = {"total": len(rows)}
    for metric in ("schema", "value_only", "language_proxy", "required", "distractor_free", "shape"):
        summary[metric] = sum(row.get("checks", {}).get(metric, False) for row in rows)
    summary["all"] = sum(row["pass"] for row in rows)
    summary["pairs"] = sum(
        all(row["pass"] for row in rows[offset:offset + 2])
        for offset in range(0, len(rows), 2)
    )
    summary["seconds"] = round(sum(row["seconds"] for row in rows), 2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true", help="Apply current checks to saved outputs")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("models", nargs="*", metavar="MODEL", help=", ".join(MODELS))
    args = parser.parse_args()
    if unknown := set(args.models) - MODELS.keys():
        parser.error(f"unknown models: {', '.join(sorted(unknown))}")
    cases = [json.loads(line) for line in (ROOT / "training/test/field-conditions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    for case in cases:
        assert all(checks(case, case["target"]).values()), case["id"]
    path = args.output
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if args.rescore:
        by_id = {case["id"]: case for case in cases}
        for label, result in results.items():
            for row in result["rows"]:
                if "output" in row:
                    row["checks"] = checks(by_id[row["id"]], row["output"])
                    row["pass"] = all(row["checks"].values())
            result["summary"] = summarize(result["rows"])
            print(label, result["summary"])
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    for label in args.models or ("exaone", "exaone-lora", "gemma", "qwen"):
        model, adapter = MODELS[label]
        rows = []
        with load_server_backend(LOCAL / "llama-vulkan/llama-server.exe", model, adapter, "gpu") as backend:
            for seed in (42, 123):
                for case in cases:
                    start = time.perf_counter()
                    row = {"id": case["id"], "seed": seed, "pass": False}
                    try:
                        output = backend(case["request"], seed)
                        metrics = checks(case, output)
                        row.update(output=output, checks=metrics)
                        row["pass"] = all(metrics.values())
                    except (RuntimeError, ValueError) as error:
                        row["error"] = str(error)
                    row["seconds"] = round(time.perf_counter() - start, 3)
                    rows.append(row)
        results[label] = {"summary": summarize(rows), "rows": rows}
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(label, results[label]["summary"], flush=True)


if __name__ == "__main__":
    main()
