"""Compare context selection on held-out bilingual field generation."""

import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from statistics import median

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = ROOT / ".training"
sys.path.insert(0, str(ROOT))

from schema_generator.llama_server_backend import load_server_backend

spec = importlib.util.spec_from_file_location("field_conditions", HERE.parent / "field-conditions-v1/run.py")
field_conditions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(field_conditions)

FACTS = {
    "event": "행사명 / event name",
    "capacity": "최대 정원 / maximum attendee capacity",
    "booked": "현재 예약자 수 / current attendee bookings",
    "remaining": "남은 자리 수 / remaining seats",
    "start_time": "시작 시각 / start time",
    "product": "상품명 / product name",
    "additional_days": "추가 배송 소요 일수 / additional delivery time in days",
    "original_estimate_days": "최초 예상 배송 기간 / original delivery estimate in days",
    "plan": "요금제 이름 / subscription plan name",
    "new_price_usd": "새 월 요금 / new monthly subscription price in USD",
    "previous_price_usd": "기존 월 요금 / previous monthly subscription price in USD",
    "item_name": "표시할 재고 품목명 / inventory item name for display",
    "related_item": "관련된 다른 품목 / related inventory item",
    "building": "시설명 / facility building name",
    "closed_floor": "점검으로 폐쇄되는 층 / floor closed for maintenance",
    "open_floor": "이용할 수 있는 층 / floor open to visitors",
    "invoice_id": "청구서 번호 / invoice ID",
    "outstanding_usd": "미결제 금액 / outstanding invoice amount in USD",
    "paid_usd": "이미 결제된 금액 / paid invoice amount in USD",
}

GOLD = {
    "reservation": ("event", "capacity", "start_time"),
    "shipping": ("product", "additional_days"),
    "subscription": ("plan", "new_price_usd"),
    "inventory": ("item_name",),
    "facilities": ("building", "closed_floor"),
    "billing": ("invoice_id", "outstanding_usd"),
}


def terms(text):
    words = set(re.findall(r"[a-z]+", text.casefold()))
    for chunk in re.findall(r"[가-힣]+", text):
        words.update(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return words


def context_selections(cases):
    torch.set_num_threads(4)
    model_dir = LOCAL / "multilingual-e5-small"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    model_load = time.perf_counter() - started

    @torch.inference_mode()
    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1)

    started = time.perf_counter()
    candidates = dict(zip(FACTS, encode([f"passage: {key}. {hint}" for key, hint in FACTS.items()]), strict=True))
    index_time = time.perf_counter() - started
    cache = {}
    results = {}
    for case in cases:
        request = case["request"]
        keys = list(request["context"])
        gold = GOLD[case["domain"]]
        description = request["field_schema"]["description"]
        query = f"{request['field_name']}. {description}"
        cache_key = (query, tuple(keys), len(gold))
        if cache_key not in cache:
            started = time.perf_counter()
            query_terms = terms(query)
            lexical = {key: len(query_terms & terms(key + " " + FACTS[key])) for key in keys}
            lexical_time = time.perf_counter() - started
            started = time.perf_counter()
            query_embedding = encode([f"query: {query}"])[0]
            dense = {key: float(query_embedding @ candidates[key]) for key in keys}
            dense_time = time.perf_counter() - started
            cache[cache_key] = (lexical, dense, lexical_time, dense_time)
        lexical, dense, lexical_time, dense_time = cache[cache_key]
        top = lambda scores: sorted(keys, key=scores.__getitem__, reverse=True)[:len(gold)]
        results[case["id"]] = {
            "gold": list(gold),
            "full": keys,
            "keyword": top(lexical),
            "semantic": top(dense),
            "oracle": list(gold),
            "keyword_scores": lexical,
            "semantic_scores": dense,
            "keyword_ms": round(1000 * lexical_time, 3),
            "semantic_ms": round(1000 * dense_time, 3),
        }
    return results, {"encoder_load_seconds": round(model_load, 3), "index_seconds": round(index_time, 3),
                     "unique_queries": len(cache)}


def main():
    cases = [json.loads(line) for line in (ROOT / "training/test/field-conditions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    selections, timing = context_selections(cases)
    modes = ("full", "keyword", "semantic", "oracle")
    rows = {mode: [] for mode in modes}
    responses = {}
    with load_server_backend(LOCAL / "llama-vulkan/llama-server.exe", LOCAL / "exaone-4.0-1.2b-q4.gguf",
                             LOCAL / "exaone-docstyle-lora.gguf", "gpu") as generate:
        for seed in (42, 123):
            for case in cases:
                case_id = case["id"]
                original = case["request"]
                for mode in modes:
                    selected = set(selections[case_id][mode])
                    # Preserve the source field order across selection modes.
                    context = {key: value for key, value in original["context"].items() if key in selected}
                    request = dict(original, context=context)
                    cache_key = (case_id, seed, tuple(context))
                    if cache_key not in responses:
                        started = time.perf_counter()
                        response = {}
                        try:
                            output = generate(request, seed)
                            metrics = field_conditions.checks(case, output)
                            response.update(output=output, checks=metrics, passed=all(metrics.values()))
                        except (RuntimeError, ValueError) as error:
                            response.update(error=str(error), passed=False)
                        response["seconds"] = round(time.perf_counter() - started, 3)
                        responses[cache_key] = response
                    rows[mode].append(dict(responses[cache_key], id=case_id, seed=seed, keys=list(context)))
            print(f"Seed {seed} complete, {len(responses)} unique requests", flush=True)

    summary = {}
    for mode in modes:
        summary[mode] = {
            "selection_exact": sum(set(s[mode]) == set(s["gold"]) for s in selections.values()),
            "selection_contains_all": sum(set(s["gold"]) <= set(s[mode]) for s in selections.values()),
            "automatic_pass": sum(row["passed"] for row in rows[mode]),
            "median_generation_seconds": round(median(row["seconds"] for row in rows[mode]), 3),
        }
    timing["unique_generation_requests"] = len(responses)
    for mode in ("keyword", "semantic"):
        timing[f"{mode}_median_ms"] = round(median(s[f"{mode}_ms"] for s in selections.values()), 3)
    result = {"timing": timing, "summary": summary, "selections": selections, "rows": rows}
    (HERE / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(timing=timing, summary=summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
