# 생성 테스트용 JSON 스키마

웹 서비스에서 자주 쓰는 데이터 모델 여섯 가지입니다. `.ko.schema.json`은 `--language ko`, `.en.schema.json`은 `--language en`으로 실행합니다. 학습용 JSONL과 별도로 관리합니다.

| 스키마 | 확인할 내용 |
|---|---|
| `customer-profile.ko` | 설명 없는 이름과 나이, 이메일 형식, 중첩 배송지 |
| `product-listing.en` | 상품 분류와 상세 설명의 일치, 소수 단가, 중복 없는 태그 |
| `purchase-order.en` | 주문 항목 배열, 수량과 단가, 중첩 배송지, 고객 배송 요청 |
| `support-ticket.ko` | 문의 분류·제목·본문의 일치, 요청 조치, 미해결 상태의 null |
| `event-booking.ko` | 행사명·장소·정원이 예약 안내문에 정확히 반영되는지 확인 |
| `api-error.en` | 고정 오류 코드, UUID 형식, 응답 문구와 재시도 대기 시간의 일치 |

프로젝트 루트에서 실행합니다.

```powershell
py generate.py training/test/schemas/event-booking.ko.schema.json --language ko --count 5 --seed 42 --out outputs/event-booking.ko.json
```

형식과 범위는 생성 시 자동으로 검증합니다. 필드 사이의 의미 일치는 생성 결과를 직접 확인합니다.
