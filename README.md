# LSF-KR Ticket Forecast Dashboard

**LE SSERAFIM VR Concert : Invitation** (롯데시네마 월드타워 3관) 티켓 판매 예측 대시보드.

tws-ticket-forecast 후속 — 분석 로직 업그레이드 + Slack 자동 수집 제거, 수동 CSV 업로드 방식.

https://YOUR_ID.github.io/lsf-kr-ticket-forecast/

---

## 주요 특징

- **3-모델 앙상블** (Logistic + Gompertz + Bass diffusion) · AICc 기반 가중평균
- **Moving block bootstrap + Dirichlet Monte Carlo** 로 P5–P95 신뢰구간
- **1차 + 2차 판매 분리 예측**: 현재 예매 가능한 회차(1차)와 아직 미오픈인 후반 회차(2차)를 별도 컴포넌트로 추정 후 합산
- **Top-down ↔ Bottom-up reconciliation**: 전체 앙상블 예측과 회차별 합이 자동 일치
- **Special 회차 자동 감지**: VIP/사전할당으로 추정되는 회차(만석 또는 큰 판매점프)는 별도 플래그 처리 후 일반 모델에서 제외

---

## 사용법 (스냅샷 추가할 때마다)

```bash
# 1. 슬랙 등에서 받은 CSV 를 snapshots/ 에 넣기
cp ~/Downloads/LotteCinema_*.csv snapshots/

# 2. 분석 실행 (data.json 갱신)
python3 analyze.py

# 3. 커밋 푸시
git add snapshots/ data.json
git commit -m "update: $(date +%Y-%m-%d_%H%M)"
git push
```

약 1분 후 GitHub Pages 가 자동으로 배포 갱신.

---

## 최초 세팅

```bash
# 의존성
pip install pandas numpy scipy

# 1) GitHub 레포 생성 (빈 레포로)
git init
git add .
git commit -m "init"
git branch -M main
git remote add origin https://github.com/YOUR_ID/lsf-kr-ticket-forecast.git
git push -u origin main

# 2) GitHub Pages 켜기
#    Settings → Pages → Source: Deploy from a branch
#    Branch: main / (root) → Save
```

약 1분 후 `https://YOUR_ID.github.io/lsf-kr-ticket-forecast/` 접속 가능.

---

## 파일 구조

```
.
├── index.html         # 단일 SPA 대시보드 (Chart.js 사용)
├── analyze.py         # 스냅샷 로딩 + 3-모델 앙상블 + 부트스트랩 → data.json
├── data.json          # analyze.py 가 생성 (대시보드가 fetch)
├── snapshots/         # CSV 원본 저장 (파일명에 YYYYMMDD_HHMMSS 포함 필수)
│   └── LotteCinema_..._YYYYMMDD_HHMMSS_Full.csv
└── README.md
```

### CSV 스키마 (Lotte Cinema export)

```
Date, Start Time, Screen, # of Tickets Sold, # of Seats, Occupancy Rate
```

파일명에 스냅샷 타임스탬프가 `YYYYMMDD_HHMMSS` 형식으로 반드시 들어있어야 함
(예: `LotteCinema_르세라핌 브이알 콘서트 _ 인비테이션_20260420_142928_Full.csv`).

---

## 대시보드 기능

1. **총 누적 판매 + 앙상블 예측 곡선** — P5/P25/P50/P75/P95 밴드, 개별 3모델 토글, 속도 보정 슬라이더
2. **모델 비교 테이블** — 각 모델의 K̂, AICc, Akaike 가중치
3. **스냅샷별 판매 속도** — 시간당 판매량 추이 (J-curve 감쇠 시각화)
4. **요일 × 시간대 히트맵** — 현재 점유율 / 예측 최종 점유율 토글, Special 회차 별표 표시
5. **회차별 예측 테이블** — 정렬·필터·Special 숨기기 지원, P5–P95 표시
6. **CSV 드래그 미리보기** — 정식 분석 없이 브라우저에서 누적 판매량만 빠르게 확인

---

## 분석 파라미터 (analyze.py 상단에서 조정)

### 이벤트 스케줄
| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| `EVENT_START_DATE` | `2026-04-15` | 상영 시작일 |
| `EVENT_END_DATE` | `2026-05-26` | 상영 종료일 (이 날까지 포함, 42일) |
| `SHOWINGS_PER_DAY` | 8 | 평일 기준 하루 회차 수 |
| `SEATS_PER_SHOWING` | 158 | 회차당 좌석 수 (월드타워 3관) |
| `OPENING_DAY_SHOWINGS` | 5 | 오픈 첫날(부분 영업) 회차 수 |

위 값으로 총 `333회차 × 158석 = 52,614석` 이 자동 계산됨. CSV에 후반 회차가 아직 없어도 K 상한이 바르게 잡힘.

### 모델/부트스트랩
| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| `BOOTSTRAP_ITERATIONS` | 1000 | Moving block bootstrap 반복 횟수 |
| `BLOCK_SIZE` | 2 | 잔차 부트스트랩 블록 크기 |
| `DIRICHLET_ALPHA` | 10 | 앙상블 가중치 MC Dirichlet 집중도 |
| `SPECIAL_JUMP` | 80 | "Special" 판단 기준: 단일 스냅샷 판매 증가량 |
| `SPECIAL_OCC` | 0.95 | "Special" 판단 기준: 점유율 임계 |

---

## 문제 해결

- **"data.json 을 찾을 수 없어요"** → `python3 analyze.py` 를 먼저 실행. snapshots/ 에 CSV 최소 1개 필요.
- **분석이 실패하거나 NaN** → 스냅샷이 3개 미만이면 모델 fit 불가. 4개 이상 누적 후 다시 시도.
- **GitHub Pages 가 안 뜸** → Settings → Pages 에서 Source 가 `main` / `(root)` 인지 확인. 첫 배포는 1–3분 소요.
