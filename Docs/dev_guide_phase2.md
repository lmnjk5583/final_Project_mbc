# 개발 기준서 부록 — Phase 2 이후 (dev_guide_phase2.md)

> **Phase 1 완료 후 참조. Phase 1 개발 중에는 `dev_guide.md`만 읽는다.**
> 최종 수정: 2026-03-27

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

## 18. 인터페이스 (팀원 연동 — WebSocket)

| 채널 | 방향 | 주기 | 내용 |
|------|------|------|------|
| `/ws/traffic` | 서버→클라이언트 | 1초 | 정체 상태 갱신 |
| `/ws/alerts` | 서버→클라이언트 | 즉시 | 이상징후 알림 |
