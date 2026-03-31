# 고속도로 이상징후 탐지 시스템 — 리서치 보고서

> 역주행 탐지 완전 분석 + 정체 해결 전략 논문 근거
> 최종 수정: 2026-03-27

---

## 1. 시스템 전체 구조

```
run_wrongway.py → Detector.run()
    ├── tracker.py       YOLO + ByteTrack 래퍼
    ├── flow_map.py      15×15 Grid Flow Map (EMA 학습)
    ├── judge.py         역주행 판별 (cos_sim + 투표 + 히스테리시스)
    ├── id_manager.py    W1/W2 라벨 + ID 재매칭 (3중 게이트)
    ├── camera_switch.py 장면 전환 감지
    ├── bbox_stabilizer.py BBox EMA 안정화
    ├── visualizer.py    시각화 (키보드 토글)
    └── logger.py        CSV 3종 로깅

저장:
    results/results_N.mp4
    models/flow_map.npy
    logs/run_YYYYMMDD_HHMMSS/{frame_log, track_log, events_log}.csv
```

---

## 2. 모듈 핵심 요약

### config.py
- `detect_only=True`: 기존 flow_map.npy 로드 후 탐지만 실행
- `detect_only=False`: 앞 learning_frames(500) 동안 학습 후 탐지 전환
- `enable_online_flow_update`: 탐지 중 정상 차량 벡터로 flow_map 점진 갱신

### state.py — 주요 상태 변수

| 변수 | 타입 | 설명 |
|------|------|------|
| `trajectories` | `dict[id→list[(cx,cy)]]` | 차량별 중심점 궤적 |
| `wrong_way_count` | `dict[id→int]` | 역주행 의심 누적 카운트 |
| `wrong_way_ids` | `set` | 역주행 확정 ID |
| `first_seen_frame` | `dict[id→int]` | ID 첫 등장 프레임 |

### flow_map.py
- **학습**: 이동벡터 단위화 → 셀(r,c) EMA 갱신 (`alpha=0.10`)
- **보간**: 이중 선형 보간으로 경계 불연속 제거
- **평활화**: count<5인 셀을 주변 3×3 평균으로 채움 (학습 완료·재학습 완료·150프레임마다)

### judge.py — 역주행 판별 파이프라인
```
① 이미 확정 → True 즉시 반환
② 원근 속도 게이트: scale=0.3+0.7*(cy/h), threshold=base*scale
③ 궤적 샘플링 (최대 8포인트) → cos_sim = ndx*flow_vx + ndy*flow_vy
④ 투표: disagree_ratio = disagree/total
⑤ ratio≥0.6 → count+1 → count≥4 → 확정 / else → count=max(0,count-2)
```

### id_manager.py — 3중 재매칭 게이트
1. 시간: 사라진 지 45프레임 이내
2. 거리: 이전 위치 120px 이내
3. 방향: 여전히 cos_sim ≤ -0.5

### camera_switch.py
- 감지: `adj_diff > avg_diff×5.0` OR `ref_diff > 40` 4회 연속
- 처리: reset_for_relearn() + flow.reset() + 재학습 300프레임 + 쿨다운 150프레임

### logger.py

| 파일 | 단위 | 주요 컬럼 |
|------|------|-----------|
| `frame_log.csv` | 프레임 | 시간, 추적수, 역주행확정수 |
| `track_log.csv` | 프레임×트랙 | ID, bbox, 속도, cos, 판정 |
| `events_log.csv` | 확정 이벤트 | 등장→의심→확정 프레임/초 |

### visualizer.py — 키보드 토글
`T`궤적 / `D`방향화살표 / `F`흐름장 / `S`속도 / `I`정보패널 / `V`투표디버그 / `P`cos값 / `Q`종료

---

## 3. 핵심 알고리즘

### EMA
- Flow Map: `flow[r,c] = 0.9×flow[r,c] + 0.1×new_vec` (α=0.10)
- BBox 안정화: `smoothed = 0.5×current + 0.5×prev` (α=0.50)

### 코사인 유사도
`cos_sim = ndx×flow_vx + ndy×flow_vy` | +1=정상, -1=역주행, 임계값=-0.5

### 원근 보정 속도 게이트
`scale = 0.3 + 0.7×(cy/frame_h)` → 위쪽(멀리)=낮은 임계값, 아래쪽=높은 임계값

### 히스테리시스
의심 +1 → 4회 이상 확정 / 정상 판정 시 max(0, count-2) (이중 감소)

---

## 4. 파라미터 완전 정리

### Flow Map / 학습
| 파라미터 | 현재값 | 권장범위 | 의미 |
|----------|--------|----------|------|
| `grid_size` | 15 | 10~25 | 그리드 해상도 |
| `alpha` | 0.10 | 0.05~0.20 | EMA 학습 속도 |
| `min_samples` | 5 | 3~10 | 공간 평활화 기준 |
| `learning_frames` | 500 | 200~500 | 초기 학습 프레임 수 |

### 역주행 판별
| 파라미터 | 현재값 | 권장범위 | 의미 |
|----------|--------|----------|------|
| `velocity_window` | 15 | 8~20 | 속도/방향 계산 간격 |
| `base_speed_threshold` | 7.0 | 5~12 | 기본 속도 게이트 |
| `cos_threshold` | -0.5 | -0.3~-0.7 | 역방향 판정 기준 |
| `vote_threshold` | 0.6 | 0.55~0.75 | disagree 비율 의심 기준 |
| `wrong_count_threshold` | 4 | 3~6 | 의심→확정 카운트 |
| `min_move_distance` | 20.0 | 10~25 | 최소 누적 이동거리 |
| `min_move_per_frame` | 1.5 | 1.0~3.0 | 프레임당 이동거리 |

### ID 관리 / 카메라 전환
| 파라미터 | 현재값 | 권장범위 | 의미 |
|----------|--------|----------|------|
| `id_match_distance` | 120 | 80~200 | 재매칭 허용 거리(px) |
| `reappear_frame_limit` | 45 | 30~60 | 재매칭 시간 제한(프레임) |
| `stale_threshold` | 90 | 60~120 | ID 삭제까지 프레임 수 |
| `relearn_frames` | 300 | 120~300 | 전환 후 재학습 프레임 |
| `cooldown_frames` | 150 | 90~240 | 재학습 후 쿨다운 |
| `switch_confirm_needed` | 4 | 2~5 | 전환 확정 연속 의심 횟수 |
| `bbox_stab_alpha` | 0.5 | 0.3~0.8 | BBox EMA 반응 속도 |

---

## 5. 개선 히스토리 (10단계)

| 단계 | 핵심 문제 | 해결책 |
|------|-----------|--------|
| 1 | 고정 ROI 한계 | Grid Flow Map으로 전환 |
| 2 | 셀 경계 벡터 불연속 | 이중 선형 보간 |
| 3 | Occlusion ID switch | ByteTrack persist + 궤적 저장 |
| 4 | track_id 바뀔 때마다 라벨 변경 | display_id_map W1/W2 고정 |
| 5 | 재등장 ID 정상 처리 | 3중 게이트 재매칭 |
| 6 | 사라진 구간 오탐 | 다중 포인트 투표 + 히스테리시스 |
| 7 | 카메라 전환 시 오탐 폭발 | grayscale 축소 감지 + 재학습 |
| 8 | 정량 평가 부재 | 등장→의심→확정 소요시간 측정 |
| 9 | 정지차 역주행 오탐 | 이중 거리 필터 (누적+프레임당) |
| 10 | BBox jitter 방향 오염 | BBoxStabilizer EMA |

---

## 6. 강점 / 한계

**강점**: 무감독 학습, 장면 일반화, 실시간, ID 연속성, 정량 평가 기반
**한계**: 절대 km/h 미변환, 정체 판별 없음, 시계열 미활용, YOLO 미특화

---

## 7. 논문 리서치

### 7.1 YOLO + ByteTrack 기반 교통류

**[논문 1]** Liu et al. 2025 — *Vehicle Flow Detection with YOLOv8n+ByteTrack* (MDPI WEV)
mAP 62.8%, MOTA 72.16%. NWD loss로 소형 객체 강화. 현재 프로젝트와 동일 구조.

**[논문 2]** *YOLOv8 Real-Time Traffic Analysis* (IJIRSET 2024)
가상 카운팅 라인으로 차량 카운팅·속도·점유율 산출.

### 7.2 교통 정체 감지

**[논문 3]** Chakraborty et al. 2018 — YOLO vs AlexNet 정체 이진 분류
정확도 91.2%(YOLO) vs 90.5%(AlexNet). YOLO가 이미지 품질 변화에 강건.

**[논문 4]** Chen et al. 2013 — MOFV + 점유율로 SMOOTH/SLOW/CONGESTED 3단계 분류
개별 차량 감지 없이 광학 흐름 값만 사용. 현재 Flow Map과 연계 가능.

**[논문 5]** *Small Parallel Residual CNN* (Nature 2025)
ResNet 기반 CCTV 정체 분류. TrafficNet·CCTRIB 데이터셋.

### 7.3 교통류 이론 (Fundamental Diagram)

**[자료 1]** Wikipedia — *Fundamental Diagram*
Flow-Density-Speed 3변수 관계. 용량(Capacity)에서 자유류→혼잡류 상전이.

**[논문 6]** Panayiotou et al. 2025 — 딥러닝+TFD 하이브리드
flow만으로 정체 단정 불가(비단조). TFD 결합 필요.

**[논문 7]** *FD-Markov-LSTM* (ScienceDirect)
기본 다이어그램+Markov+LSTM. 베이징·LA에서 MAE 39% 감소.

### 7.4 딥러닝 기반 정체 예측

**[논문 8]** *EfficientNet+LSTM Ensemble* (Nature 2025)
퍼지 로직으로 정체 레벨(저/중/고) 분류.

**[논문 9]** *CCTV→교통류 예측 end-to-end* (Transportmetrica B 2024)
현재 프로젝트와 동일한 CCTV 기반 파이프라인.

**[논문 10]** *Traffic Congestion Prediction 총람* (MDPI Smart Cities 2025)
GRU가 LSTM 대비 계산 비용 낮음. 앙상블 기법 정확도 향상.

**[논문 11]** *Highway Traffic Flow Prediction* (Nature 2022)
교통량·밀도·속도 예측 + 반복 업데이트 + 지능형 우회 결정.

### 7.5 역주행·이상 감지

**[논문 12]** *DL Algorithms for Traffic Forecasting* (Wiley 2024)
교통류 예측 4분야(flow, speed, congestion, spatial-temporal) 리뷰.

**[논문 13]** *YOLOv8 Traffic Violation Detection* (2025)
역주행+정체 통합 시스템. 정체 레벨 기반 신호 타이밍 동적 조정.

### 7.6 시스템 구현 참고

**[자료 2]** CA DOT *TrafficVision* — CCTV 24개 동시 모니터링, 역주행+정체 통합, ATMS 연동.
**[자료 3]** Yellow Systems 2024 — Hangzhou 정체 15% 감소, Pittsburgh Surtrac 통행시간 25% 감소.

---

## 8. 정체 해결 전략 (논문 근거)

### 8.1 VMS (가변 전광판)
**[논문 1]** MDPI Infrastructures 2024
- 정체 지속 시간 **-25%**, 우회 이용률 **+18%**, 2차 사고 **-30%**
- 감지 후 **3분 이내** 송출 시 최대 효과
- FHWA: 정체 상류 1~3km 지점 설치 최적

### 8.2 VSL (가변 속도 제한)
**[논문 2]** Papageorgiou et al. 2006
- 여행시간 **-4%**, 속도 분산 **-12~20%**, 충격파 형성 억제
- 단계적 하향: 100→80→60 km/h
- SLOW 판정 시 속도 제한 하향 권고

### 8.3 Ramp Metering
**[논문 3]** ALINEA (Papageorgiou 1991, 인용 1,000회+)
- TTS **-10~50%**, 정체 지속 시간 **-55%**
- `q_ramp(k+1) = q_ramp(k) + K×(o_des - o(k))` (K≈70 veh/h/%)
**[논문 4]** Cazorla 2022 — 본선 속도 **+13~26%**, 사고율 **-10~25%**

### 8.4 Incident Management
**[자료 1]** FHWA CHART — 사고 처리 시간 **-11분**, 2차 사고 **-10%**
**[자료 2]** TRL — 골든타임: 감지 후 **10분 이내** 대응. CCTV→관제→순찰대 자동화.

### 8.5 Route Diversion
**[논문 5]** TRR 2022 — VMS+우회 안내 통합 시 단일 전략 대비 현저히 효과적
**[논문 6]** PLOS ONE 2018 — 다중 경로 최적화, 인근 도로 용량 제약 필수

### 8.6 한국 사례: 서울 FTMS
VMS 260개 + 램프미터링 30개 + LCS 33개 통합. 시스템 이용자 **520% 증가**.
아키텍처: CCTV → FTMS 서버 → VMS/램프미터링 자동 제어 + 운영자 확인.

---

## 9. 정체 레벨별 조치 매핑

| 정체 레벨 | 조치 우선순위 | 관제센터 표시 |
|-----------|--------------|--------------|
| **SLOW** | ① VSL 하향 (논문 2) → ② VMS 서행 안내 (논문 1) | 노랑 배너 + 속도 제한 권고 |
| **CONGESTED** | ① VMS 우회 → ② 램프 미터링 권고 → ③ 순찰 출동 | 빨강 배너 + 조치 버튼 3종 |
| **CONGESTED 지속** | ④ 사고 의심 → 현장 확인 (자료 2) | 긴급 알림 + 순찰 출동 버튼 |

---

## 10. YOLO 추가학습 (Fine-tuning)

```
고속도로 CCTV 이미지+라벨 수집 (임시/ 폴더)
    ↓ 이미지 증강 (밝기·반전 등), 클래스 0=차량, 학습:검증=8:2
    ↓ YOLO11n Fine-tuning (기존 best.pt에서)
    ↓ mAP50·오탐율 평가
    ↓ config.py model_path 교체 (나머지 코드 수정 불필요)
```

목적: 야간·역광·원거리 소형 차량·트럭/버스 혼재 환경에서 오탐/미탐 감소
