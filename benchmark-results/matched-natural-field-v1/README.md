# EXAONE과 Gemma의 동일 조건 필드 생성 비교

EXAONE 4.0 1.2B와 Gemma 3 1B IT의 배포 가중치와 프로젝트 LoRA 적용 결과를 비교한다. EXAONE은 앞서 같은 파일로 학습한 어댑터를 사용했다. Gemma 학습 가중치는 [Unsloth의 Gemma 3 1B IT 체크포인트](https://huggingface.co/unsloth/gemma-3-1b-it/tree/5b11413a10db4e486ef16a20101fd028f8f2499c)다.

용량 정리 후 추론용 Gemma GGUF와 어댑터로 평가를 재실행할 수 있다. Gemma 학습이나 LoRA 변환을 다시 하려면 위 리비전의 원본 가중치·설정·토크나이저를 `.training/gemma-3-1b-it`에 복원한다. 당시 학습 데이터는 170건으로 늘어나기 전의 138건이며, 현재 JSONL에서 `temporal-` 그룹을 제외하면 복원할 수 있다.

학습 설정은 138개 예제, 138스텝, 시드 42, Intel XPU, FP32, AdamW 학습률 0.0003, LoRA rank 8과 alpha 16, `q_proj`·`v_proj`로 맞췄다. 학습용 JSONL의 SHA-256은 `0204b15ba3f86ca470ec9b34fa1e39a6ebb98f2ec12fe37ded4f30cd993acc65`다. 모델별 토크나이저와 채팅 템플릿을 사용한다.

평가에는 [고정 필드 요청 24건](../../training/test/field-conditions.jsonl)과 시드 42·123을 사용했다. 두 모델 모두 Q4_K_M GGUF, F16 LoRA, 동일 Vulkan 실행기, 현행 필드 프롬프트와 JSON 문자열 제약을 사용한다. 앞선 평가와 조건이 같은 미적용 두 모델과 EXAONE LoRA의 출력을 재사용했다.

## 판정 기준

- 요청 언어, 스키마, 필드 형태를 지키고 해당 필드의 값으로 바로 쓸 수 있어야 한다.
- 필요한 사실을 모두 포함하고 숫자·단위·역할·시간의 관계를 정확히 표현해야 한다.
- 문맥에 있는 추가 사실은 정확하고 해당 필드에 관련되면 허용한다. 문맥에 없는 새 사실과 확정되지 않은 조치는 감점한다.
- 모델명을 제거하고 요청별로 후보 순서를 섞었다. 자동 검사 탈락도 포함해 사람의 판정을 기록한다. 동일 문장일 필요는 없다.
- 같은 요청에서 양쪽이 모두 통과했는지, 한쪽만 통과했는지 기록한다. 한 단어와 문장, 한국어와 영어, 준비 이후 생성 시간을 별도로 보고한다.

## 결과

| 구성 | 사용 가능 | 문장 | 한 단어 | 한국어 | 영어 | 문맥 쌍 | 48건 생성 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXAONE | 0/48 | 0/40 | 0/8 | 0/24 | 0/24 | 0/24 | 44.15초 |
| Gemma | 8/48 | 0/40 | 8/8 | 4/24 | 4/24 | 4/24 | 76.92초 |
| EXAONE + LoRA | 17/48 | 15/40 | 2/8 | 10/24 | 7/24 | 6/24 | 52.13초 |
| Gemma + LoRA | 10/48 | 2/40 | 8/8 | 4/24 | 6/24 | 5/24 | 52.58초 |

학습한 모델의 동일 요청 48건 중 EXAONE만 통과한 요청은 15건, Gemma만 통과한 요청은 8건, 둘 다 통과한 요청은 2건이었다. Gemma는 품목명에서 강했고 학습 후 영어 시설 안내 두 건을 추가로 통과했다. EXAONE의 주요 강점은 구독과 청구서 문장이었다. 두 학습 모델 모두 배송 안내에서 추가 소요 기간을 잘못 표현하는 등 8건 모두 탈락했다. 아래 시간은 모델 시작을 뺀 한 번의 측정이다. 두 LoRA 모델의 전체 생성 시간은 약 0.45초 차이였다.

192건 모두 자동 스키마 검사와 별도로 검수했다. 검수자는 한 명이며, 세 기존 구성을 먼저 보고 신규 출력을 나중에 같은 기준으로 검토했다. 일부 문맥은 다른 시드에서 같은 출력을 내므로 48건을 독립적인 48개 업무 사례로 해석하기 어렵다. 이번 결과는 현재 프롬프트와 138스텝 설정에서 EXAONE을 문장 생성용 모델로 유지할 근거다.

[출력 원문](results.json) · [모델명을 가린 검수표](blind.json) · [판정 기록](decisions.json) · [집계와 요청별 짝 비교](summary.json)

```powershell
. .training/xpu-venv/Scripts/Activate.ps1
py training/train.py --base .training/gemma-3-1b-it --device xpu --steps 138 --out .training/gemma-docstyle-v1
py .training/llama.cpp-b12502/convert_lora_to_gguf.py --base .training/gemma-3-1b-it --outtype f16 --outfile .training/gemma-docstyle-lora.gguf .training/gemma-docstyle-v1
py benchmark-results/field-conditions-v1/run.py --output benchmark-results/matched-natural-field-v1/results.json exaone gemma exaone-lora gemma-lora
py benchmark-results/matched-natural-field-v1/review.py prepare
py benchmark-results/matched-natural-field-v1/review.py score
```
