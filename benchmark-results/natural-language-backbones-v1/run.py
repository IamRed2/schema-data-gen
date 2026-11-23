"""Compare instruction checkpoints without project LoRA on the same field requests."""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator

from schema_generator.llama_server_backend import load_server_backend


CASES = [
    ("customer-ko", "Korean", "message", "고객이 등록한 문의 본문. 환불 지연과 고객이 요청한 후속 조치를 포함합니다.",
     {"category": "refund", "item": "무선 키보드", "waiting_days": 7},
     "무선 키보드 환불을 7일째 기다리는 고객이 처리 확인을 요청"),
    ("agent-ko", "Korean", "message", "상담원이 고객에게 발송하는 답변. 확인된 환불 상태와 후속 조치를 포함합니다.",
     {"category": "refund", "item": "무선 키보드", "refund_status": "승인 완료", "next_action": "영업일 3일 내 입금"},
     "상담원이 무선 키보드 환불 승인 및 영업일 3일 내 입금을 안내"),
    ("event-ko", "Korean", "notice", "참가자에게 표시되는 예약 안내문. 행사명과 정원, 시작 시각을 포함합니다.",
     {"event": "유리공예 체험", "capacity": 18, "bookings": 11, "start_time": "오후 2시", "end_time": "오후 4시"},
     "유리공예 체험의 정원 18명과 시작 시각 오후 2시"),
    ("catalog-ko", "Korean", "description", "상품 상세 페이지에 표시되는 소개. 상품의 용도와 주요 특징을 포함합니다.",
     {"category": "camping", "product_name": "접이식 캠핑 테이블", "feature": "높이 조절 가능"},
     "접이식 캠핑 테이블의 야외 사용과 높이 조절 특징"),
    ("incident-ko", "Korean", "title", "운영 장애 목록에 표시되는 제목. 영향을 받은 기능과 증상을 요약합니다.",
     {"component": "모바일 결제", "symptom": "결제 승인 후 주문 확인 화면이 열리지 않음"},
     "모바일 결제 승인 이후 주문 확인 화면 미표시를 요약"),
    ("name-ko", "Korean", "name", None, {}, "자연스러운 한국어 사람 이름"),
    ("customer-en", "English", "message", "Customer-submitted support message describing the delivery problem and requested action.",
     {"category": "delivery", "item": "desk lamp", "tracking_status": "delivered", "received": False},
     "Customer asks for help with desk lamp marked delivered but missing"),
    ("agent-en", "English", "message", "Support agent reply stating the confirmed shipping problem and the next step.",
     {"category": "delivery", "item": "desk lamp", "finding": "delivered to the wrong building", "next_action": "send a replacement tomorrow"},
     "Agent confirms wrong building and promises a replacement tomorrow"),
    ("event-en", "English", "notice", "Public reservation notice including the event name, capacity and start time.",
     {"event": "Paper Marbling Workshop", "capacity": 14, "bookings": 8, "start_time": "10 a.m.", "end_time": "noon"},
     "Paper Marbling Workshop has capacity 14 and starts at 10 a.m."),
    ("catalog-en", "English", "description", "Product page copy stating the intended use and key feature of the named item.",
     {"category": "kitchen", "product_name": "digital kitchen scale", "feature": "tare function"},
     "Kitchen weighing and the tare function of the digital kitchen scale"),
    ("incident-en", "English", "title", "Incident title summarizing the affected feature and its observed failure.",
     {"component": "password reset", "symptom": "reset link expires immediately"},
     "Password reset link expires immediately"),
    ("name-en", "English", "name", None, {}, "Plausible English-language human name"),
]

MODELS = {
    "exaone": ROOT / ".training/exaone-4.0-1.2b-q4.gguf",
    "gemma": ROOT / ".training/gemma-3-1b-it-Q4_K_M.gguf",
    "qwen": ROOT / "models/qwen3.5-0.8b-q4.gguf",
}

natural = "--natural" in sys.argv[1:]
labels = [arg for arg in sys.argv[1:] if arg != "--natural"]


def natural_messages(request):
    field = request["field_name"]
    language = request["output_language"]
    description = request["field_schema"].get("description")
    facts = "\n".join(f"- {key}: {value}" for key, value in request["context"].items())
    prompt = f"Write the actual value of the field '{field}' in {language}."
    if description:
        prompt += f"\nField definition: {description}"
    if facts:
        prompt += f"\nUse these facts:\n{facts}"
    prompt += "\nReturn a short fictional value as one JSON string. Do not repeat the definition or explain your answer."
    return [{"role": "user", "content": prompt}]


results_file = Path(__file__).with_name("natural-results.json" if natural else "results.json")
results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else {}
server = ROOT / ".training/llama-vulkan/llama-server.exe"
for label in labels or MODELS:
    model = MODELS[label]
    rows = []
    try:
        started = time.perf_counter()
        with load_server_backend(server, model, device="gpu",
                                 format_messages=natural_messages if natural else None) as backend:
            startup = time.perf_counter() - started
            for case_id, language, field, description, context, expected in CASES:
                schema = {"type": "string", "minLength": 2, "maxLength": 180}
                if description:
                    schema["description"] = description
                request = {"field_name": field, "path": "$." + field, "field_schema": schema,
                           "target_type": "string", "context": context, "output_language": language}
                row = {"id": case_id, "request": request, "expected": expected, "seed": 42}
                before = time.perf_counter()
                try:
                    output = backend(request, 42)
                    row.update(output=output, schema_valid=Draft202012Validator(schema).is_valid(output),
                               contains_hangul=bool(re.search("[가-힣]", output)))
                except (RuntimeError, ValueError) as exc:
                    row["error"] = str(exc)
                row["seconds"] = round(time.perf_counter() - before, 3)
                rows.append(row)
        results[label] = {"model": model.name, "bytes": model.stat().st_size,
                          "startup_seconds": round(startup, 3), "rows": rows}
    except (OSError, RuntimeError) as exc:
        results[label] = {"error": str(exc)}
    results_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(label, results[label].get("error", f"{len(rows)} outputs, {sum(row['seconds'] for row in rows):.2f}s"), flush=True)
