# JSON Schema 기반 합성 데이터 생성기

필드의 `description`과 같은 레코드 안의 다른 필드를 참고해 합성 데이터를 생성합니다. 스키마 설명은 한국어와 영어로 작성할 수 있으며, 생성할 자연어 값의 언어는 `--language ko` 또는 `--language en`으로 지정합니다.

`description`에는 필드의 의미, 표시 위치, 다른 필드와의 관계를 적습니다. 예를 들어 고객 문의 본문은 `고객 문의 본문. category에 해당하는 문제와 고객이 요청한 조치를 포함합니다.`로 정의할 수 있습니다. 검증 가능한 길이와 범위는 `minLength`, `maxLength`, `minimum`, `maximum` 같은 스키마 키워드로 지정합니다.

## 실행

Python 3.10 이상이 필요합니다. Windows에서는 [llama.cpp 릴리스](https://github.com/ggml-org/llama.cpp/releases)의 CPU용 `llama-completion.exe`와 DLL을 `runtime/llama.cpp/`에 둡니다. 양자화한 기본 GGUF 모델은 `models/model.gguf`에 둡니다. 다른 경로의 파일은 `--runner`와 `--model`로 지정할 수 있습니다. 생성기는 로컬 실행 파일을 직접 호출하며 서버나 API 키가 필요 없습니다.

용량을 줄이려면 모델은 4비트 양자화 파일 하나를 배치합니다. 초기 평가 후보인 [Qwen3-0.6B Q4_K_M](https://huggingface.co/QuantFactory/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B.Q4_K_M.gguf)은 가중치 파일 기준 약 484 MB입니다. 실행 파일과 DLL 용량은 별도입니다.

```powershell
py -m pip install -r requirements.txt
py generate.py examples/support.ko.schema.json --out outputs/support.ko.json --count 2 --language ko
py generate.py examples/support.ko.schema.json --count 2 --language en
```

Windows에서는 배치 파일에도 같은 인수를 넘길 수 있습니다. 상대 경로는 명령을 실행한 폴더를 기준으로 해석합니다.

```powershell
.\generate.bat examples/support.ko.schema.json --language ko --count 2 --out outputs/support.ko.json
```

명령 형식은 `py generate.py <스키마 파일> [--out <출력 위치>] --language ko|en [--count <건수>]`입니다. `--out`에 파일 경로를 지정하면 해당 파일에 저장하고, 생략하면 표준 출력으로 보냅니다. 기존 파일을 지정하면 생성이 완료된 뒤 덮어씁니다.

`--count`를 생략하면 5건, `--seed`를 생략하면 42를 사용합니다. 값 단위 생성에서는 문자열 길이 제한을 가능한 경우 모델 출력에도 적용합니다. 범위나 배수가 지정된 수치, 불리언, null, 지원하는 문자열 형식은 Python에서 먼저 생성합니다. 모든 필드가 이 경로로 처리되면 모델 파일 없이 실행할 수 있습니다.

출력은 UTF-8 JSON 배열입니다. Python이 객체·배열을 만들고, 모델이 생성한 값은 스키마 타입에 맞게 파싱합니다. 각 값과 최종 레코드를 검증하며, 모델 값이 검증에 실패하면 해당 필드만 최대 세 번 요청합니다. 날짜를 재현하려면 시드와 함께 `--reference-date YYYY-MM-DD`를 지정합니다.

입력부터 저장까지의 세부 순서는 [데이터 생성 순서와 방식](docs/data-generation-flow.md)에 정리했습니다.

## 필드 단위 생성

Intel NPU·GPU 생성과 GPU 학습 설정은 [가속 장치 실행 방법](docs/accelerators.md)을 참고합니다.

필드명과 설명을 함께 읽어 값을 생성합니다. `name`, `age`처럼 의미가 분명한 필드는 설명을 생략할 수 있습니다. 설명은 의미와 출력 형태를 구체화합니다. 이름·나이는 설명을 생략하고 직업·자기소개에는 설명을 붙인 예제는 `examples/person.schema.json`입니다.

기본값인 `--mode auto`와 `--mode field`는 모두 Python 구조 생성 경로를 사용합니다. 규칙으로 처리하기 어려운 필드는 필드명과 설명, 타입, 앞서 생성한 관련 값을 모델에 전달합니다. 모델은 JSON 문자열 하나에 값의 텍스트를 담습니다. 정수 필드에서 모델이 `"27"`을 반환하면 Python이 정수로 변환합니다. 문자열 필드는 텍스트를 그대로 사용합니다.

```powershell
py generate.py training/inventory-label.schema.json --count 4 --language ko --out outputs/inventory.ko.json
```

객체·배열·문자열·정수·실수·불리언·null 및 타입 목록을 지원합니다. `enum`과 `const`는 Python이 허용된 값을 선택합니다. 객체는 선언된 선택 필드도 생성하고, 배열 길이는 최소·최대 제약을 지키면서 기본적으로 0~3개에서 선택합니다. `prefixItems`가 있으면 그 길이도 고려합니다. 참조, 조합, 조건부 스키마 등 아직 지원하지 않는 생성 제약은 오류로 알립니다. 최종 객체·배열의 제약을 만족하지 못하면 오류를 반환합니다.

기존 레코드 전체 생성은 `--mode record`로 비교할 수 있습니다. 값 단위 경로는 매 값마다 실행 파일을 호출하므로 모델 로딩 시간이 누적됩니다. 이전 필드 모드의 속도 측정치는 이 구현의 성능을 나타내지 않습니다.

[필드 단위 생성 비교 결과](benchmark-results/field-v2/README.md)에서 같은 0.6B 모델을 사용한 처리량과 샘플을 확인할 수 있습니다.

Vulkan GPU에서 모델을 한 번만 로드하려면 `--server-runner`를 지정합니다. [GGUF 가속 실행 방법](docs/accelerators.md)

## 전용 모델

Qwen3-0.6B 기본 모델에 GGUF LoRA 어댑터를 붙이면 기존 가중치 한 벌을 공유할 수 있습니다. `models/schema-lora.gguf`를 적용하려면 다음과 같이 실행합니다.

```powershell
py generate.py examples/support.ko.schema.json --adapter models/schema-lora.gguf --out outputs/support.ko.json --count 4 --language ko
```

첫 어댑터는 약 2.3 MB입니다. 기본 GGUF와 합쳐 가중치가 약 486.5 MB이며, `.training`은 학습과 재현에 쓰는 작업 폴더입니다. 어댑터는 초기 실험용입니다. 시설 점검 검증에서 출력 언어와 스키마 형식은 맞았지만, 한국어 표현과 작성자 역할 구분에 오류가 있었습니다.

`training/train.py`는 `training/train/values.jsonl`로 값 단위 LoRA를 학습하고 `training/test/values.jsonl`로 토큰 손실을 평가합니다. [학습 실행 방법](training/README.md)에 환경 준비와 데이터 형식, GGUF 변환 절차를 정리했습니다. 이전 레코드 단위 학습 원본은 `training/source/record-examples.json`에 보관했습니다.

```powershell
py training/train.py --steps 48 --out .training/value-adapter-v1
py .training/llama.cpp-b12502/convert_lora_to_gguf.py --base .training/base --outtype f16 --outfile models/schema-value-lora.gguf .training/value-adapter-v1
```

`training/maintenance.schema.json`과 `training/maintenance-request.schema.json`은 학습에 사용하지 않은 비교용 스키마입니다. 두 파일은 `notice`의 설명만 달라서, 같은 필드에서 관리자의 점검 안내와 입주자의 수리 요청이 각각 생성되는지 확인할 수 있습니다. `ko`와 `en`에서 각각 언어, 작성자 역할, 시설 유형, 문장 자연스러움을 확인합니다.

## 샘플 확인

`support.ko.schema.json`과 `support.en.schema.json`에 각각 `--language ko`와 `--language en`을 적용해 네 가지 조합을 실행합니다. `category`에는 스키마에 정의된 `refund` 또는 `delivery`가 나오고, `message`는 해당 주제의 구체적인 고객 문의여야 합니다. 자연어의 언어와 설명 부합 여부는 샘플을 읽어 확인합니다. 설명만 “상담원이 고객에게 보낸 답변을 한 문장으로 작성한다”로 바꿔 생성된 문장의 역할이 바뀌는지도 확인할 수 있습니다.

## 백본 비교

[관련 프로젝트와 데이터 조사](docs/model-research.md)에 후보와 평가 기준을 정리했습니다. 비교할 모델 GGUF를 준비한 뒤 실행합니다.

[학습 데이터 선정 문서](docs/training-data-plan.md)에 구조화 생성·다국어 자료의 우선순위와 가공 방법을 정리했습니다.

```powershell
py tools/benchmark.py --candidate models/qwen3.5-0.8b-q4.gguf --out benchmark-results/backbone-v2 --count 4
```

새 결과 폴더에 `summary.json`과 조합별 생성 샘플이 저장됩니다. 기본 비교는 두 백본을 사용합니다. 기존 LoRA를 기준 모델에 적용하려면 `--baseline-adapter models/schema-lora.gguf`를 추가합니다.

같은 모델에서 레코드 전체와 필드 단위 생성을 비교하려면 아래와 같이 실행합니다. `--case`를 생략하면 전체 스키마로 평가합니다.

```powershell
py tools/benchmark.py --candidate models/model.gguf --candidate-mode field --case inventory-label --case activity-title --case inventory-note --out benchmark-results/field-v3 --count 4
```

## 폴더 구성

[프로젝트 구조와 처리 흐름](docs/project-structure.md)에 각 파일이 하는 일과 연결 관계를 정리했습니다.

- 프로젝트 루트에 실행 스크립트와 GGUF 모델을 둡니다.
- `runtime/llama.cpp/`에 추론 실행 파일과 DLL을 둡니다.
- `examples/`는 실행 예제, `training/`은 학습 데이터와 평가용 스키마입니다.
- `outputs/`는 생성된 사용자 JSON, `benchmark-results/`는 비교 결과와 샘플입니다.
- `docs/`는 조사와 평가 기준, `.training/`은 학습 환경과 별도로 보관한 런타임 구성 요소입니다.
