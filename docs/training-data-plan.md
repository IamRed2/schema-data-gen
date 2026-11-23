# 학습 데이터 선정

조사일은 2026-11-19이다. 목표는 여러 분야에서 필드 설명과 관련 값의 문맥을 읽고, 지정한 언어로 단어·짧은 문구·문장을 생성하는 것이다. 다음 선정은 데이터 카드에 근거한 우선순위이며 원본 전체의 샘플 품질 검수는 아직 진행하지 않았다.

## 우선 검토할 원본

| 우선순위 | 원본 | 확인한 내용 | 가공할 부분 |
|---|---|---|---|
| 1 | NVIDIA Nemotron Structured Outputs v2 | 영어 구조화 출력 자료. CC BY 4.0. Direct Generation의 JSON 9,382건, Diversified Tasks의 JSON 6,734건을 공개한다. 후자는 스키마만으로 현실적인 값을 생성하는 과제를 포함한다. | JSON 중 스키마 기반 생성 과제를 선별한다. RL용 자료이므로 정답 레코드가 제공되는지 행별로 확인하고, 필요하면 별도로 생성·검수한다. 스키마 설명과 결과 값의 의미 일치를 검사한다. |
| 2 | Cohere Labs Aya Dataset | 약 20만 건, 한국어·영어를 포함한 다국어 지시·응답 자료. Apache 2.0. `inputs`, `targets`, `language_code` 열을 제공한다. | 한국어·영어 중 이름 짓기, 짧은 창작, 설명 작성 등 실제 값 생성으로 바꿀 수 있는 예제를 선별한다. 원문이나 질문이 필요한 작업은 문맥을 보존한다. |
| 3 | Bitext Customer Support | 영어 문의·답변 26,872쌍. CDLA-Sharing-1.0. 범주와 의도가 있다. | 고객 문의와 상담원 답변을 서로 다른 필드 설명으로 구성한다. 고객 지원 영역의 일부로 사용한다. |
| 보완 | AI Hub 상담 전사 | 한국어 고객·상담원 발화를 제공하는 후보. | 접근·이용 조건을 확인한 뒤 완결된 짧은 발화를 선별한다. 여러 화자의 발화를 올바른 역할에 연결한다. |

출처는 [NVIDIA 데이터 카드](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2), [Aya 데이터 카드](https://huggingface.co/datasets/CohereLabs/aya_dataset), [Bitext 데이터 카드](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), [AI Hub 한국어 상담 음성](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=100)이다.

## 별도로 만들어야 할 핵심 예제

공개 자료에서 검수한 값을 씨앗으로 사용하고, 다음 조건을 교차한 프로젝트 전용 데이터를 만든다.

- 분야는 물품, 행사, 교육, 작업 기록, 일상, 시설, 제품, 고객 지원 등으로 넓힌다.
- 같은 필드명에 설명만 바꾼 예제를 넣는다. 물품명과 물품 상태, 활동 제목과 활동 소감, 안내문과 요청문을 짝으로 만든다.
- 한국어·영어 설명과 한국어·영어 출력의 네 조합을 포함한다.
- 다른 필드의 값이 바뀌면 생성 값의 주제도 바뀌게 한다.
- 단어·짧은 문구·문장 형태를 각각 명시한다. 한 단어는 공백 기준인지 의미 단위인지 설명에 적는다.
- 의미를 직접 드러내는 필드명과 `text`, `value` 같은 필드명을 함께 사용한다.
- 이름·나이·도시처럼 익숙한 필드는 설명을 생략한 예제도 포함한다. 설명이 있는 예제는 필드명과 설명을 함께 해석해야 정답이 나오도록 구성한다. 예를 들어 `name`에 “행사에 사용할 가상의 단체 이름”을 지정하면 단체명을 생성하게 한다.

외부 데이터를 가공할 때는 `source_revision`, `source_id`, `license`, `domain`, `description_language` 메타데이터를 보존한다. 현재 `training/train.py`는 필드 단위 추론과 같은 `request`와 문자열 `target`, 분할용 `split_group`을 읽는다. 직접 작성한 초기 데이터는 `training/train/values.jsonl`과 `training/test/values.jsonl`에 배치했다. [학습 데이터 형식](../training/README.md)을 참고한다.

## 첫 데이터셋 제안

초기 규모는 검수된 2,000~5,000건으로 시작한다. 양을 먼저 채우기보다 언어 네 조합과 표현 형태 세 종류가 고르게 들어가는지 확인한다. 원본과 설명의 변형은 같은 분할에 두고, 도메인과 스키마 형태를 일부 통째로 평가셋에 남긴다. 단어·문구·문장 형태, 출력 언어, 필드 간 관계, 중복, 원본 스키마 준수를 검사한 뒤 자연스러움과 설명 부합을 샘플 검수한다.

## 보류하거나 평가에 사용할 자료

- SchemaBench와 JSONSchemaBench는 스키마 구조의 다양성과 제약 평가에 활용한다. 자연스러운 정답 값을 확보하는 작업이 추가로 필요하다. [SchemaBench](https://github.com/thunlp/SchemaReinforcementLearning) · [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench)
- K2-Feedback은 제공자가 직접 학습 사용을 권하지 않고, 평가용 성격을 명시했다. 한국어 평가 기준의 참고 자료로 둔다. [데이터 카드](https://huggingface.co/datasets/HAERAE-HUB/K2-Feedback)
- Tulu 3 전체 혼합물은 각 원본의 조건과 작업이 다양하다. 이번 작은 데이터셋은 관련 원본을 개별적으로 선정한다. [데이터 카드](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
