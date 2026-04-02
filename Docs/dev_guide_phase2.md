# 개발 기준서 부록 — Phase 2 이후 (dev_guide_phase2.md)

> Phase 1: 규칙 기반 jam_score — **완료**
> Phase 2: GRU 40% 혼합 — **완료** (`gru_module.py` 포함 src/ 전체 작동 중)
> Phase 3: GRU 65% primary — **미구현** (§13 참조)
> 최종 수정: 2026-04-02

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

## 13. Phase 3 설계 명세 (미구현)

> Phase 2 안정화 후 착수. GRU를 primary 모델로 전환, rule은 보조 역할.

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

### Phase 3 전환 체크리스트
| 항목 | 기준 | 확인 방법 |
|------|------|-----------|
| GRU 수렴 | MAE(gru, rule) < 0.05 | frame_log.csv gru_score 컬럼 분석 |
| 온라인 학습 | 누적 스텝 ≥ 500 | gru_module.py online_step_count |
| 실영상 검증 | SMOOTH/SLOW/CONGESTED 오탐률 < 10% | run_test.py + 수동 검토 |

---

## 18. 인터페이스 (팀원 연동 — WebSocket)

| 채널 | 방향 | 주기 | 내용 |
|------|------|------|------|
| `/ws/traffic` | 서버→클라이언트 | 1초 | 정체 상태 갱신 |
| `/ws/alerts` | 서버→클라이언트 | 즉시 | 이상징후 알림 |
