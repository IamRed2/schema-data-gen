"""Compare date training and explicit temporal context with a fixed EXAONE backbone."""

import argparse
import calendar
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = ROOT / ".training"
sys.path.insert(0, str(ROOT))

from schema_generator.llama_server_backend import load_server_backend

ADAPTERS = {
    "baseline": "exaone-docstyle-lora.gguf",
    "temporal": "exaone-temporal-lora.gguf",
}
ROLE_LABELS = {
    "applications_close_on": ("신청서 제출 마감일 {date}", "Application submission deadline: {date}"),
    "results_on": ("심사 결과 발표일 {date}", "Results announcement date: {date}"),
    "original_estimate_days": ("기존 예상 기간 {value}일", "Original estimate: {value} days"),
    "additional_days": ("기존 예상보다 {value}일 더 걸림", "Takes {value} days longer than the original estimate"),
    "remaining_days": ("오늘부터 완료까지 {value}일 남음", "{value} days remaining from today until completion"),
}
SUBJECTS = {"shipping": "delivery", "temporal-repair": "repair", "temporal-grant": "grant"}


def date_forms(value):
    day = date.fromisoformat(value)
    return (value, f"{day.year}년 {day.month}월 {day.day}일", f"{day.month}월 {day.day}일",
            f"{calendar.month_name[day.month]} {day.day}, {day.year}",
            f"{calendar.month_name[day.month]} {day.day}")


def enrich(request):
    lang = int(request["output_language"] == "English")
    context = dict(request["context"])
    roles = []
    for key, value in context.items():
        if key in ROLE_LABELS:
            label = ROLE_LABELS[key][lang]
            rendering = date_forms(value)[3 if lang else 1] if key.endswith("_on") else None
            roles.append(label.format(date=rendering, value=value))
    context["temporal_roles"] = "; ".join(roles)
    return dict(request, context=context)


def read_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def evaluate(row, output):
    request = row["request"]
    context = request["context"]
    english = request["output_language"] == "English"
    normalized = " ".join(output.casefold().split())
    role = row["role"]
    expected = context[role]
    schema_ok = Draft202012Validator(request["field_schema"]).is_valid(output)
    hangul = bool(re.search("[가-힣]", output))
    language = (not hangul and bool(re.search("[a-z]", normalized))) if english else hangul
    value_only = not output.lstrip().startswith(("{", "[", "```")) and not re.match(
        r"\s*(notice|message)\s*:", output, re.IGNORECASE
    )
    if role.endswith("_on"):
        value_hit = any(form.casefold() in normalized for form in date_forms(expected))
        cue = (r"\b(due|deadline|submit\w*|clos\w*)\b" if role == "applications_close_on"
               else r"\b(announc\w*|publish\w*|releas\w*)\b") if english else (
                   r"마감|제출|까지" if role == "applications_close_on" else r"발표|공개|게시"
               )
        role_hit = bool(re.search(cue, normalized))
    else:
        numeric = rf"(?<!\d){expected}(?!\d)"
        value_hit = bool(re.search(numeric + (r"\s*days?\b" if english else r"\s*일"), normalized))
        if english:
            pattern = (numeric + r"\s+(?:(?:more|additional|extra)\s+days?|days?\s+(?:longer|more))"
                       if role == "additional_days" else r"\bin\s+" + numeric + r"\s+days?\b")
        else:
            pattern = (numeric + r"\s*일(?:이|을)?\s*(?:가량\s*)?(?:더|추가로)"
                       if role == "additional_days" else numeric + r"\s*일\s*(?:후|뒤|남)")
        role_hit = bool(re.search(pattern, normalized))
    return {"schema": schema_ok, "language": language, "value_only": value_only,
            "temporal_value": value_hit, "temporal_role_proxy": role_hit}


def cases():
    items = []
    for row in read_rows("training/test/values.jsonl"):
        domain = row["split_group"]
        if domain not in SUBJECTS:
            continue
        index = int(row["id"].rsplit("-", 1)[1])
        if domain == "temporal-grant":
            role = ("applications_close_on", "results_on")[index % 2]
        else:
            role = ("additional_days", "remaining_days")[index % 2]
        items.append(dict(row, domain=domain, role=role))
    for row in read_rows("training/test/field-conditions.jsonl"):
        if row["domain"] == "shipping":
            items.append(dict(row, role="additional_days", target=None))
    assert len(items) == 20
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", choices=tuple(ADAPTERS))
    args = parser.parse_args()
    path = HERE / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    items = cases()
    for row in items:
        if row["target"]:
            assert all(evaluate(row, row["target"]).values()), row["id"]
    records = []
    with load_server_backend(LOCAL / "llama-vulkan/llama-server.exe", LOCAL / "exaone-4.0-1.2b-q4.gguf",
                             LOCAL / ADAPTERS[args.adapter], "gpu") as generate:
        for seed in (42, 123):
            for case in items:
                for mode in ("raw", "roles"):
                    request = case["request"] if mode == "raw" else enrich(case["request"])
                    started = time.perf_counter()
                    row = {key: case[key] for key in ("id", "domain", "role")}
                    row.update(seed=seed, mode=mode, request=request, reference=case["target"])
                    try:
                        output = generate(request, seed)
                        row.update(output=output, checks=evaluate(case, output))
                    except (RuntimeError, ValueError) as error:
                        row["error"] = str(error)
                    row["seconds"] = round(time.perf_counter() - started, 3)
                    records.append(row)
            print(f"{args.adapter}: {seed}, {len(records)} outputs", flush=True)
    results[args.adapter] = records
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
