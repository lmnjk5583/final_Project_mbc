# 개발 기준서 부록 — Phase 2 이후 (dev_guide_phase2.md)

> Phase 1: 규칙 기반 jam_score — **완료**
> Phase 2: GRU 40% 혼합 — **완료** (`gru_module.py` 포함 src/ 전체 작동 중)
> Phase 3: GRU 65% primary + 미래 예측 — **구현 중** (§13 참조)
> 최종 수정: 2026-04-03

---

## 11. 카메라 PTZ 대응 — Phase 2 추가 내용

> Phase 1 처리 내용은 `dev_guide.md §11` 참조.

### Phase 2 (GRU 있음)

Phase 1 reset 코드에 아래 한 줄 추가:

```python
    self.gru_module.reset()  # buffer.clear() + hidden_state=zeros + warmup=30
```

---

## 12. GRU 설계 명세 (Phase 2)

```
입력:  [x_{t-29}, ..., x_t]  T=30프레임, 7차원
GRU:   layers=2, hidden=64
출력:  FC(64→32 ReLU) → FC(32→3 Softmax)
       [p_smooth, p_slow, p_congested]

gru_score = p_slow × 0.5 + p_congested × 1.0   (0~1)

최종: final_jam = 0.60 × rule_jam + 0.40 × gru_score

학습:
  초기 — 학습 구간 데이터로 x_{t+1} 예측 (MSE, 자기지도)
  온라인 — SMOOTH 구간만 가중치 업데이트 + replay_buffer 혼합

camera_switch 후:
  feature_buffer.clear()
  hidden_state = zeros
  warmup_remaining = 30 → 이 기간은 rule_jam만 사용
```

GRU 가중치는 화각 무관 (feature가 비율이므로). 재학습 불필요.

---

## 14. 알림 등급 / 조치 권고

| 등급 | 레벨 | 색상 |
|------|------|------|
| INFO | SMOOTH | 초록 |
| CAUTION | SLOW | 노랑 |
| WARNING/CRITICAL | CONGESTED | 빨강 |

| 레벨 | 조치 | 근거 |
|------|------|------|
| SLOW | VSL 하향 권고 | Papageorgiou 2006 |
| SLOW | VMS 서행 안내 | MDPI 2024 |
| CONGESTED | VMS 우회 안내 | MDPI 2024 |
| CONGESTED | 램프 미터링 | ALINEA 1991 |
| CONGESTED 5분+ | 순찰대 출동 | FHWA CHART |

---

## 17. 기술 스택

| 영역 | 기술 | 버전 |
|------|------|------|
| 감지 | YOLO11n | 8.x |
| 추적 | ByteTrack | - |
| 영상 | OpenCV | 4.x |
| 수치 | NumPy | 1.x |
| 딥러닝 (Phase 2) | PyTorch | 2.x |
| 백엔드 | Flask + Flask-SocketIO | 3.x + 5.x |
| DB | MySQL 8.0 + SQLAlchemy | - |
| API | Flask-RESTX (Swagger) | - |
| 프론트 | React 18 + Chart.js/Recharts | - |
| 배포 | Docker + Compose | 24.x |

---

## 13. Phase 3 설계 명세 (구현 중)

> Phase 2 안정화 후 착수. GRU를 primary 모델로 전환 + **미래 정체 예측** 추가.

### 13-A. GRU primary 전환 (블렌드 비율 변경)

```
최종 blend: final_jam = 0.35 × rule_jam + 0.65 × gru_score

전환 조건:
  - Phase 2에서 gru_score와 rule_jam의 MAE < 0.05 로 수렴 확인 후 전환
  - 온라인 학습 누적 스텝 ≥ 500회 달성 시

Phase 3 추가 작업:
  - gru_module.py: gru_blend_ratio 0.40 → 0.65 로 변경 (config.py에서 제어)
  - congestion_judge.py: Phase 3 모드에서 fallback 조건 완화
    (LCS ≥ 0.8이어도 GRU warmup 완료 시 GRU 우선 사용)
  - 모니터링: gru_score vs rule_jam 괴리 > 0.3 이면 경고 로그 출력

config.py 변경:
  gru_blend_ratio: 0.40 → 0.65
```

### 13-B. 미래 정체 예측 (자기회귀 롤아웃) — ✅ 구현 완료

```
구조 변경 (_GRUNet):
  기존: GRU(7→64) → FC(64→32→3) Softmax (분류 전용)
  추가: pred_head = Linear(64→7)  ← 다음 프레임 feature 예측 헤드

자기회귀 롤아웃 (predict_future):
  1. 현재 버퍼 [x_{t-29}, ..., x_t] → GRU → hidden_state h_t, 출력 o_t
  2. pred_head(o_t) → x̂_{t+1}  (7차원 예측 feature)
  3. x̂_{t+1}를 GRU 단일 스텝 입력 → h_{t+1}, o_{t+1}
  4. 분류 헤드(o_{t+1}) → [p_smooth, p_slow, p_congested]
  5. 2~4 반복 × N스텝

반환값:
  [{"step": 1, "p_smooth": 0.12, "p_slow": 0.71, "p_congested": 0.17,
    "gru_score": 0.52}, ...]

config.py 파라미터:
  gru_forecast_steps: 150  (기본 150프레임 ≈ 5초@30fps)

pretrain 개선:
  기존: last_out[:, :7] 해킹 (hidden 앞 7차원을 feature로 근사)
  수정: pred_head(last_out) 사용 (정식 예측 헤드, MSE 학습)

학습 흐름:
  - pretrain(): MSE Loss로 GRU + pred_head 가중치 학습 (x_{t+1} 예측)
  - online_step(): CrossEntropy로 GRU + 분류 헤드(FC1+FC2) 학습
  - pred_head는 pretrain에서만 학습됨 (온라인 학습 대상 아님)

웹 연동:
  predict_future() 결과를 WebSocket /ws/traffic 채널에 forecast 필드로 포함
  프론트에서 gru_score 추이를 시계열 차트로 시각화
```

### Phase 3 전환 체크리스트
| 항목 | 기준 | 확인 방법 | 상태 |
|------|------|-----------|------|
| pred_head 추가 | _GRUNet에 Linear(64→7) | 코드 확인 | ✅ |
| pretrain 해킹 제거 | pred_head(last_out) 사용 | 코드 확인 | ✅ |
| predict_future 구현 | 자기회귀 롤아웃 N스텝 | 코드 확인 | ✅ |
| GRU 수렴 | MAE(gru, rule) < 0.05 | frame_log.csv gru_score 컬럼 분석 | ⬜ |
| 온라인 학습 | 누적 스텝 ≥ 500 | gru_module.py online_step_count | ⬜ |
| 실영상 검증 | SMOOTH/SLOW/CONGESTED 오탐률 < 10% | run_test.py + 수동 검토 | ⬜ |
| 블렌드 비율 전환 | gru_blend_ratio 0.40 → 0.65 | config.py 변경 | ⬜ |

---

## 18. 인터페이스 (팀원 연동 — WebSocket)

| 채널 | 방향 | 주기 | 내용 |
|------|------|------|------|
| `/ws/traffic` | 서버→클라이언트 | 1초 | 정체 상태 갱신 |
| `/ws/alerts` | 서버→클라이언트 | 즉시 | 이상징후 알림 |
