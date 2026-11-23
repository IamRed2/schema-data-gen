# 학습 실행

프로젝트 루트에서 아래 명령을 실행합니다. 학습 코드는 `training/train.py`이며 현재 값 단위 생성과 같은 프롬프트를 사용합니다. 이름·설명·타입·문맥·출력 언어를 입력받고 값 하나를 JSON 문자열로 반환하도록 학습합니다.

## 폴더

- `train/values.jsonl`은 학습용입니다.
- `test/values.jsonl`은 가중치 업데이트에 사용하지 않는 평가용입니다.
- `source/record-examples.json`은 기존에 직접 작성한 원본입니다.
- `prepare_data.py`는 원본을 변환하고 예제 데이터를 추가합니다. 실행하면 train·test 파일과 manifest를 다시 씁니다.
- `manifest.json`에는 건수와 분할 그룹을 기록합니다.

같은 도메인과 설명 번역 변형은 같은 분할에 두었습니다. 이 test 분할은 값 단위 학습의 평가용이며, 초기 원본 일부는 이전 레코드 단위 LoRA에 사용됐습니다.

- `source/e2e-selected.jsonl`은 Ondřej Dušek, David M. Howcroft, Verena Rieser의 정제된 E2E NLG 학습 CSV에서 고른 영어 식당 소개 16건입니다. 원문을 정답으로 유지하고 입력 속성 이름을 현재 형식에 맞게 바꿨습니다. 식당별 한 건을 선택하고, 이름·음식 종류·지역과 존재하는 추가 속성이 모두 정답에 문자로 등장하는 행을 골랐습니다. 각 행에 원본·리비전·행 번호·라이선스를 남겼습니다. [원본과 논문 정보](https://github.com/tuetschek/e2e-cleaning) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- `source/context-library.schema.json`은 학습용입니다. 지점·책 제목·연장 일수를 모두 써야 하는 한국어·영어 중첩 안내 예제를 담았습니다.
- `source/context-event.schema.json`은 평가용입니다. 행사명과 정확한 좌석 수를 한 문장에 반영하고, 관련 태그 두 개를 생성합니다. `source/context-examples.json`에 두 스키마의 입력과 정답 레코드가 있습니다.

공개 예제를 다시 선별하려면 해당 리비전에서 `cleaned-data/train-fixed.no-ol.csv`를 받아 `py training/import_e2e.py <CSV 경로>`를 실행합니다. 일반적인 학습 데이터 재생성에는 저장된 선별 파일을 사용합니다.

범위 학습에는 같은 필드명에 서로 다른 범위를 준 예제가 있습니다. 정수와 소수의 상한·하한, 경곗값 제외, 배수, 문자열 길이를 포함합니다. 평가용 정원 예제는 학습에서 제외했습니다.

`source/temporal-examples.json`은 Python이 준비한 날짜와 기간을 자연어 필드에 반영하는 예제입니다. 학습 32건은 제작 지연·전시·라이선스·서비스 점검이고, 평가 16건은 지원금 일정·기기 수리입니다. 한국어와 영어를 같은 수로 넣었습니다. 개막과 폐막, 갱신 시작과 만료, 기존 예상 대비 추가 소요와 오늘부터 남은 기간을 구분합니다. ISO 날짜, 윤일, 연도 경계, 자정을 넘는 한국 표준시 점검을 포함합니다.

## 환경 준비

학습에는 Python 3.12를 권합니다. 프로젝트에 이미 준비한 `.training/venv`와 `.training/base`를 사용할 수 있습니다. 새 환경을 만들 때는 다음을 실행합니다.

```powershell
py -3.12 -m venv .training/venv
. .training/venv/Scripts/Activate.ps1
py -m pip install -r training/requirements.txt
```

CPU용 PyTorch를 선택하려면 학습 의존성 설치 전에 아래 명령을 실행합니다.

```powershell
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

학습은 Hugging Face 형식의 Qwen3-0.6B 원본 가중치를 사용합니다. `.training/base/`에 `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`을 준비합니다. 추론용 4비트 `models/model.gguf`와 별도의 학습 입력입니다.

## 실행

```powershell
. .training/venv/Scripts/Activate.ps1
py training/prepare_data.py
py training/train.py --steps 48 --out .training/value-adapter-v1
```

`--data`와 `--test-data`로 입력 파일, `--base`로 원본 모델 경로, `--threads`로 CPU 스레드 수를 지정할 수 있습니다. 출력 폴더는 새 경로를 사용합니다. 마지막에 LoRA 가중치와 `metrics.json`을 저장합니다. test의 평균 토큰 손실은 정답 문자열 예측 지표입니다. 생성 값의 자연스러움은 실제 추론 샘플에서도 확인합니다.

## 데이터 형식

한 줄에 JSON 객체 하나를 저장합니다. `request`는 `schema_generator/value_generator.py`가 모델에 전달하는 입력과 같은 형식입니다. `target`은 항상 문자열입니다. 정수 정답도 `"27"`, 불리언 정답도 `"true"`로 저장하며 학습 전에 타입 파싱과 스키마 검증을 합니다.

```json
{"id":"age-en-1","source":"project-authored","split_group":"person","request":{"field_name":"age","path":"$.age","field_schema":{"type":"integer","minimum":18,"maximum":90},"target_type":"integer","context":{},"output_language":"English"},"target":"27"}
```

## GGUF 변환과 사용

전체 상황을 읽는 학습 예제는 `source/scenario-examples.json`에 있습니다. 같은 상황에서 제목과 안내문을 각각 요청하고, 같은 요청에서 주제와 수치가 바뀐 상황도 포함합니다. 현장 적용 시 `--scenario`로 입력한 자연어 상황이 모든 값 생성 요청에 전달됩니다.

```powershell
py generate.py training/source/context-event.schema.json --adapter models/schema-value-scenario-lora.gguf --scenario "초보자 대상 도자기 컵 만들기 체험입니다. 앞치마와 재료가 제공됩니다." --language ko --count 1
```

현재 준비된 llama.cpp 변환 소스를 사용하는 예시입니다. 학습용 가상환경이 활성화된 상태에서 실행합니다.

```powershell
py .training/llama.cpp-b12502/convert_lora_to_gguf.py --base .training/base --outtype f16 --outfile models/schema-value-lora.gguf .training/value-adapter-v1
py generate.py examples/person.schema.json --adapter models/schema-value-lora.gguf --count 2 --language ko --out outputs/person.json
```

변환 도구를 다른 위치에 준비했다면 해당 경로로 바꿉니다. 새 어댑터는 기존 기본 GGUF와 함께 사용합니다.
