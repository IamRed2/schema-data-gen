# 날짜·기간 학습과 문맥 표기의 효과

EXAONE 4.0 1.2B Q4_K_M과 Intel Arc Vulkan 실행기를 고정했다. 기존 EXAONE LoRA는 날짜 보강 전 학습 데이터 138건으로 138스텝을 수행한 어댑터다. 현재 학습 JSONL에서 `temporal-` 그룹을 제외하면 기존 학습 파일의 SHA-256과 일치한다. 신규 LoRA는 기존 자료와 날짜·기간 예제 32건을 합친 170건에서 같은 시드, 학습률, rank, 138스텝으로 학습했다. 고정 스텝에서 신규 데이터 24건과 기존 데이터 114건이 실제 학습에 쓰였다. 동일 스텝 예산에서 학습 데이터 구성을 바꾼 비교다.

평가는 지원금 일정 8건, 수리 기간 8건, 기존 배송 기간 4건을 시드 42·123으로 각각 생성한다. 두 어댑터 모두 아래 두 입력으로 평가한다.

- **원래 키와 값**. 예를 들어 `additional_days: 7`.
- **역할 표기 추가**. 원래 문맥에 `기존 예상보다 7일 더 걸림`을 한국어 또는 영어로 덧붙인다. 정답 필드를 고르거나 다른 후보를 제거하는 처리는 두 방식 모두 사용하지 않는다.

자동 검사는 스키마·언어 문자권·값 형태와 날짜 또는 기간 표현의 존재를 확인한다. 역할 단서 검사는 예상 숫자나 날짜 옆의 표현을 기준으로 하는 대용 지표다. 예를 들어 날짜와 마감 관련 표현이 각각 등장해도 둘의 연결이 틀릴 수 있다. 배송 안내는 실제로 기존 예상보다 얼마나 더 걸리는지 직접 확인한다.

## 결과

| 학습 데이터 | 입력 | 날짜·기간 자동 검사 | 지원금 날짜 자동 검사 | 배송 수동 검수 |
|---|---|---:|---:|---:|
| 기존 | 원래 키와 값 | 18/40 | 12/16 | 0/8 |
| 기존 | 역할 표기 추가 | 31/40 | 14/16 | 1/8 |
| 날짜 보강 | 원래 키와 값 | 13/40 | 7/16 | 0/8 |
| 날짜 보강 | 역할 표기 추가 | 27/40 | 12/16 | 1/8 |

배송 출력 32건은 상품과 추가 일수가 맞고 기존 예상 대비 지연이라는 뜻이 분명하며, 근거 없는 도착·배송 약속이 없는 경우에만 통과시켰다. 기존 어댑터의 역할 표기 중 영어 `shipping-en-1` 시드 123, 신규 어댑터의 역할 표기 중 한국어 `shipping-ko-1` 시드 42가 통과했다. 자동 검사에서는 각각 5/8, 2/8이었으나, 수동 검수에서는 추가 소요를 맞히고도 같은 날짜에 배송한다고 약속하는 등의 오류를 제외했다.

역할 표기를 추가하면 두 어댑터 모두 자동 점수가 상승했지만 실제 배송 문장의 정확도는 낮았다. 같은 138스텝에서는 데이터 보강만으로 개선이 확인되지 않았다. 학습 데이터 전체를 사용한 결과나 다른 학습 시드에 대해서는 이 비교로 판단하기 어렵다. 새 어댑터는 `.training/exaone-temporal-lora.gguf`에 저장했다.

첫 XPU 실행은 메모리 부족으로 중단됐다. 재실행에서는 정답 예측에 쓰는 위치의 로짓만 계산했다. 축소 EXAONE 모델로 변경 전후의 교차 엔트로피 손실이 같은지 확인했다.

[160개 출력과 평가 입력](results.json) · [자동 검사와 요청별 비교](summary.json)

```powershell
. .training/xpu-venv/Scripts/Activate.ps1
py training/train.py --base .training/exaone-base --device xpu --steps 138 --out .training/exaone-temporal-v1
py .training/llama.cpp-b12502/convert_lora_to_gguf.py --base .training/exaone-base --outtype f16 --outfile .training/exaone-temporal-lora.gguf .training/exaone-temporal-v1
py benchmark-results/temporal-training-v1/run.py baseline
py benchmark-results/temporal-training-v1/run.py temporal
py benchmark-results/temporal-training-v1/score.py
```
