# 프로젝트 구조와 처리 흐름

## 생성 처리 흐름

`py generate.py <스키마> --language ko|en`으로 실행합니다. Python이 JSON 구조를 만들고, 모델이 필드 값의 문자열을 생성하면 Python이 타입을 변환하고 검증합니다.

| 위치 | 역할 |
|---|---|
| `generate.py` | CLI 옵션 처리, 모델·실행기 경로 확인, 결과 배열의 콘솔 출력 또는 파일 저장 |
| `value_generator.py` | 스키마 순회, 객체·배열 조립, enum·const 선택, 모델 값 요청, 숫자·불리언·null 파싱, 필드 재시도와 검증 |
| `runtime/llama.cpp/` | 서버 없이 로컬 추론을 실행하는 실행 파일과 DLL |
| `model.gguf` | 기본 Qwen3-0.6B 양자화 가중치 |
| `schema-lora.gguf` | 이전 레코드 단위 학습으로 만든 초기 LoRA. 새 값 단위 학습 결과는 별도 경로에 생성 |
| `qwen3.5-0.8b-q4.gguf` | 비교용 백본 가중치 |
| `examples/` | 사용자가 CLI로 실행할 예제 스키마 |
| `outputs/` | 생성 결과 파일 |
| `benchmark.py` | 같은 스키마·언어·시드에서 두 모델이나 실행 방식을 비교 |
| `benchmark-results/` | 이전 비교 결과와 생성 샘플. 이전 구현의 측정치는 당시 구현에 해당 |
| `training/train.py` | 필드 입력을 받아 값 문자열을 예측하는 LoRA 학습, test 데이터의 토큰 손실 측정 |
| `training/prepare_data.py` | 기존 프로젝트 예제를 값 단위 데이터로 변환하고 스칼라 타입 예제를 추가 |
| `training/train/values.jsonl` | 가중치 업데이트에 사용하는 입력·정답 |
| `training/test/values.jsonl` | 가중치 업데이트에서 제외한 도메인의 입력·정답 |
| `training/source/` | 직접 작성한 레코드와 문맥 스키마, E2E NLG에서 선별한 식당 소개 예제 |
| `training/*.schema.json` | 추론 비교용 스키마. 일부는 이전 학습에 사용하지 않은 분야 |
| `training/README.md` | 학습 환경 준비, 데이터 형식, 실행·변환 방법 |
| `.training/` | 큰 원본 가중치, 학습용 가상환경, llama.cpp 변환 소스, 학습 산출물 |
| `docs/model-research.md` | 관련 프로젝트와 백본 조사 |
| `docs/training-data-plan.md` | 외부 데이터 후보와 추후 가공 계획 |

## 현재 구현의 범위

값 생성은 필드명·설명·타입·이미 생성한 문맥을 함께 사용합니다. 일반 문자열은 그대로 쓰고 다른 스칼라는 JSON 텍스트를 파싱합니다. 객체·배열은 Python이 구성합니다. 참조·조합·조건부 스키마는 현재 오류로 알립니다. 레코드 단위 생성은 비교용 `--mode record`로 실행할 수 있습니다.

기본 GGUF 경로에서는 값 단위 모델 호출마다 실행 파일을 새로 띄우므로 로딩 시간이 누적됩니다. 내용의 자연스러움과 의미 일치는 스키마 검사 외의 샘플 검수가 필요합니다. 학습 데이터에는 프로젝트 예제와 정제된 E2E NLG 선별 예제가 들어 있습니다.
