# 스키마 기반 합성 데이터 모델 조사

목표는 다양한 분야의 JSON Schema를 받아 자연스러운 테스트 레코드를 로컬에서 대량 생성하는 것이다. 한국어와 영어 설명을 읽고, 지정한 출력 언어로 필드 의미에 맞는 단어·짧은 문구·문장을 생성한다. 예를 들어 같은 `text` 필드에 물품명 설명을 주면 “주전자”, 상태 기록 설명을 주면 “주전자 손잡이에 금이 가 있습니다.”처럼 출력한다. 자료형과 범주, 길이 등 스키마 제약도 적용한다.

현재 기준 모델은 Qwen3-0.6B Q4_K_M이다. 기존 2.3 MB LoRA는 학습에서 제외한 스키마에서도 형식을 맞췄지만, 어색한 한국어와 작성자 역할 혼동이 관찰됐다.

## 관련 프로젝트

| 프로젝트 | 사용한 데이터와 모델 | 적용할 점 |
|---|---|---|
| Schema Reinforcement Learning | JSON Schema Store와 GitHub에서 수집한 스키마 약 4만 개. Llama 3.2 3B와 Llama 3.1 8B로 실험했고 3B 가중치를 공개했다. | 실제 스키마 구조를 참고하고 유사한 스키마는 같은 분할에 둔다. [저장소](https://github.com/thunlp/SchemaReinforcementLearning) · [논문](https://aclanthology.org/2025.acl-long.243/) |
| OpenSML JSONLite 175M | OpenSML-Base-175M을 추가 사전학습하고 지도 미세조정했다. 모델 카드에 SmolLM-Corpus, Dolly 15K, SQuAD v2가 기재돼 있다. | 필드 설명이 있는 JSON 생성의 초소형 사례다. 공개 검증 범위는 짧은 영어 원문의 정보 추출이다. [모델 카드](https://huggingface.co/OpenSML/OpenSML-JSONLite-175M) |
| NVIDIA NeMo Data Designer | 설정 가능한 LLM, 시드 데이터, 통계 샘플러로 열 단위로 데이터를 생성한다. | 대량 생성에서 날짜·ID·범주를 코드로 정하고 설명에 맞는 자유문을 로컬 모델로 채우는 설계를 검토한다. [저장소](https://github.com/NVIDIA-NeMo/DataDesigner) · [컬럼 설계](https://docs.nvidia.com/nemo/datadesigner/concepts/columns) |
| JSONSchemaBench | 실제 스키마 약 1만 개를 제공한다. | 중첩 객체, 배열, 다양한 제약의 형식 평가에 사용한다. 콘텐츠 품질은 프로젝트 자체 평가에 포함한다. [저장소](https://github.com/guidance-ai/jsonschemabench) |

## 학습 데이터 후보

- **영어** — Bitext Customer Support의 문의·답변 26,872쌍에는 10개 범주와 27개 의도가 있다. `instruction`과 `response`로 고객 문의 및 상담원 답변용 레코드를 구성할 수 있다. [데이터 카드](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **한국어** — AI Hub의 가상 시나리오 상담 전사와 의도가 태깅된 민간 고객응대 자료에서, 한두 문장으로 완결되는 고객·상담원 발화를 선별할 수 있다. 실제 반영 전에 접근과 이용 조건을 확인한다. [한국어 상담 음성](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=100) · [민간분야 고객응대 음성](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71616)
- **평가셋** — 같은 필드명과 타입을 유지하면서 설명에 명시된 역할만 바꾼다. 고객 문의와 상담원 답변으로 익숙한 영역을 보고, 점검 안내와 수리 요청으로 이전 LoRA가 학습하지 않은 영역을 본다. 학습 데이터가 늘어나면 레코드 단위 무작위 분할보다 도메인·스키마·의도를 묶어 분할한다.

## 백본과 평가 기준

| 백본 | 4비트 가중치 크기 | 비교 목적 |
|---|---:|---|
| Qwen3-0.6B | 484 MB | 현재 기준. [GGUF](https://huggingface.co/QuantFactory/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B.Q4_K_M.gguf) |
| Qwen3.5-0.8B | 542 MB | 현재에 가까운 크기의 후보. [공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [평가용 GGUF](https://huggingface.co/axteam/Qwen3.5-0.8B-GGUF/blob/main/Qwen3.5-0.8B-Q4_K_M.gguf) |
| Qwen3-1.7B | 약 1.28 GB | 용량 상한을 올릴 수 있을 때의 후속 후보. [GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF) |

`tools/benchmark.py`로 같은 로컬 실행기와 시드를 사용해 고객 지원, 시설 관리, 재고, 활동 분야의 일곱 스키마와 두 출력 언어를 비교한다. 재고 평가 쌍은 `text`의 설명만 바꿔 단어와 문장을 구분하고, 활동 제목은 짧은 문구를 확인한다. 온디스크 가중치 크기, 유효 건수, 경과 시간, 초당 유효 레코드 수, 완전히 동일한 레코드의 중복률을 기록한다. 시간에는 모델 로딩과 검증, 재시도가 포함된다. 생성기가 검증을 통과한 배치를 모아 반환하므로, 유효 건수는 최종 반환률을 나타낸다.

샘플 파일에서는 각 레코드의 언어, 작성자 역할, 설명의 구체적 요건, 카테고리 일치, 문장의 자연스러움을 확인한다. 예를 들어 관리자 점검 안내에는 점검 시각과 대안이 둘 다 들어가야 하고, 수리 요청에는 고장 증상과 요청 조치가 있어야 한다. 짧은 비교 실행은 품질과 속도를 살펴보는 데 사용한다. 대량 처리량은 같은 명령의 `--count`를 늘려 측정한다.

실제 실행 수치와 샘플 관찰은 [백본 비교 1차 결과](../benchmark-results/backbone-v1/README.md)에 정리했다.
