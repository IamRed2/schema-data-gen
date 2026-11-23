import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator


APP_DIR = Path(__file__).resolve().parent
CASES = (
    ("support-request", "examples/support.en.schema.json", "message", "고객 문의"),
    ("support-reply", "training/support-reply.en.schema.json", "message", "상담원 답변"),
    ("maintenance-notice", "training/maintenance.schema.json", "notice", "관리자 점검 안내"),
    ("maintenance-request", "training/maintenance-request.schema.json", "notice", "입주자 수리 요청"),
    ("inventory-label", "training/inventory-label.schema.json", "text", "물품 이름 한 단어"),
    ("inventory-note", "training/inventory-note.schema.json", "text", "물품 상태 한 문장"),
    ("activity-title", "training/activity-title.schema.json", "text", "활동 제목 짧은 문구"),
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(model, adapter, mode, label, case, language, args):
    case_id, schema_name, text_field, role = case
    schema_path = APP_DIR / schema_name
    command = [
        sys.executable, str(APP_DIR / "generate.py"), str(schema_path),
        "--model", str(model), "--count", str(args.count), "--language", language,
        "--seed", str(args.seed), "--mode", mode,
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    if args.runner is not None:
        command.extend(["--runner", str(args.runner)])

    started = time.perf_counter()
    try:
        process = subprocess.run(
            command, capture_output=True, encoding="utf-8", errors="replace", timeout=args.timeout
        )
        if process.returncode:
            raise ValueError(process.stderr.strip() or f"Exit code {process.returncode}")
        records = json.loads(process.stdout)
        if not isinstance(records, list):
            raise ValueError("Expected a JSON array from generate.py")
        error = None
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        records, error = [], str(exc)
    elapsed = time.perf_counter() - started

    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8-sig")))
    valid = [record for record in records if validator.is_valid(record)]
    distinct = len({json.dumps(record, sort_keys=True, ensure_ascii=False) for record in valid})
    sample_name = f"{label}-{case_id}-{language}.json"
    write_json(args.out / sample_name, records)
    row = {
        "model": label,
        "case": case_id,
        "schema": schema_name,
        "language": language,
        "expected_role": role,
        "text_field": text_field,
        "requested": args.count,
        "received": len(records),
        "valid": len(valid),
        "seconds": round(elapsed, 3),
        "valid_per_second": round(len(valid) / elapsed, 3),
        "exact_duplicates": len(valid) - distinct,
        "duplicate_rate": round((len(valid) - distinct) / len(valid), 3) if valid else None,
        "sample": sample_name,
        "error": error,
    }
    print(f"{label} / {case_id} / {language}: {len(valid)}/{args.count}, {elapsed:.1f}s", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description="Compare local schema-generation backbones.")
    parser.add_argument("--baseline", type=Path, default=APP_DIR / "model.gguf")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-adapter", type=Path)
    parser.add_argument("--candidate-mode", choices=("auto", "record", "field"), default="auto")
    parser.add_argument("--case", action="append", choices=tuple(case[0] for case in CASES),
                        help="Limit evaluation to selected cases; repeat to select more than one")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--count", type=int, default=4, help="Records per model, schema and language")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300, help="Seconds per evaluation run")
    parser.add_argument("--out", required=True, type=Path, help="New directory for report and samples")
    args = parser.parse_args()
    if args.count < 1 or args.timeout <= 0:
        parser.error("--count and --timeout must be positive")
    for path in (args.baseline, args.candidate, args.baseline_adapter, args.runner):
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(str(error))

    models = {
        "baseline": (args.baseline, args.baseline_adapter, "record"),
        "candidate": (args.candidate, None, args.candidate_mode),
    }
    summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "models": {
            label: {
                "file": str(model.resolve()),
                "weight_bytes": model.stat().st_size + (adapter.stat().st_size if adapter else 0),
                "adapter": str(adapter.resolve()) if adapter else None,
                "mode": mode,
            }
            for label, (model, adapter, mode) in models.items()
        },
        "runs": [],
    }
    for case in CASES:
        if args.case and case[0] not in args.case:
            continue
        for language in ("ko", "en"):
            for label, (model, adapter, mode) in models.items():
                summary["runs"].append(evaluate(model, adapter, mode, label, case, language, args))
                write_json(args.out / "summary.json", summary)
    return int(any(row["valid"] != row["requested"] for row in summary["runs"]))


if __name__ == "__main__":
    sys.exit(main())
