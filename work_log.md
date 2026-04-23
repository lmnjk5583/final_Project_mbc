# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-23 (125차 — waiting_stable 무한 대기 수정 + HistoricalPredictor 보간) [대원]

### 오늘 한 작업

#### ① waiting_stable 최대 대기 시간 상한 추가

**문제:** 새벽 API 끊김이 반복되면 `waiting_stable` 상태에서 안정 타이머가 계속 리셋 → 플로우맵 재학습이 영원히 시작되지 않는 현상.

**원인:** `(B) 안정 대기` 구간에서 `_cur_diff > _stability_thr`이 뜰 때마다 `stable_since_frame`을 현재 프레임으로 리셋. API 끊김 시 diff가 반복적으로 임계값을 초과해 카운터 무한 리셋.

**수정:**
- `src/state.py`: `waiting_stable_entered_frame` 필드 추가 (최초 진입 시점 기록)
- `src/config.py`: `waiting_stable_max_sec = 30.0` 추가 (최대 대기 시간)
- `src/detector.py`:
  - (A) 진입 시 `waiting_stable_entered_frame` 기록
  - (B) `_total_waited >= _max_wait_frames`이면 diff 무시하고 강제 재학습 (`⏱️` 로그 출력)
  - (C) 재학습 중단 후 대기 복귀 시에도 `waiting_stable_entered_frame` 갱신

#### ② HistoricalPredictor 빈 슬롯 보간

**문제:** 재학습·학습 모드 구간은 jam_score가 기록되지 않아 해당 시간대 슬롯이 비어있음 → `predict()`가 None 반환 → 패널 "Training..." 표시.

**수정:** `historical_predictor.py`에 `_interpolate()` 추가.
- 타깃 슬롯이 비면 전후 ±12슬롯(±60분) 이내 최근 기록에서 선형 보간
- 양쪽 모두 있으면 거리 비율 가중 평균 / 한쪽만 있으면 최근접 이웃
- 신뢰도: 양측 중 낮은 값 × 간격 페널티 (`1 - gap/(12×2)`)
- 60분 이내 데이터 없으면 None 유지 (Training... 표시)
- 반환 딕셔너리에 `"interpolated": bool` 추가
- CSV는 수정하지 않음 (런타임 보간만)

### 수정 파일
`src/state.py`, `src/config.py`, `src/detector.py`, `src/historical_predictor.py`

---

## 2026-04-22 (123차 — GRU 전면 제거 + HistoricalPredictor 도입 + 재연결 잼스코어 안정화) [대원]

### 오늘 한 작업

#### ① GRU 모듈 전면 제거

**배경:** GRU는 90프레임(15초) 입력으로 5분 후를 예측 — 구조적 한계로 예측 품질 불신.  
API 끊김으로 인한 jam_score 변동에도 GRU가 노이즈를 증폭하는 부작용 존재.

- `src/gru_module.py` 삭제
- `tests/test_gru_module.py` 삭제
- `src/traffic_analyzer.py`: `gru_module` 파라미터 및 블렌딩 로직 제거, `final_jam = rule_jam` 직결
- `src/detector.py`: GRU import, 초기화, 피처 수집, pretrain/online_step, 세션 flush 전부 제거
- `src/config.py`: GRU 파라미터 섹션 전부 제거 (gru_hidden, gru_layers, gru_seq_len, gru_blend_ratio 등)
- `tests/test_traffic_analyzer.py`: GRU Mock 클래스 및 TestAnchoring(TA-09/TA-10) 제거

#### ② HistoricalPredictor 신규 도입

**설계:** 시각별(hour × 5분 슬롯) jam_score 이력을 CSV에 누적 → 5분 후 정체 수준 예측.

- 288슬롯/일 (`slot_id = hour * 12 + minute // 5`)
- 매 5분 창마다 jam_score 중앙값을 flush (API 순간 블립 내성)
- `predict()` → 해당 슬롯 데이터 없으면 `None` → 패널 "Training..." 표시
- CSV 저장 경로: `flow_map_path.parent/hist_jam_a.csv`, `hist_jam_b.csv`
- 신뢰도: `min_conf_samples=14` (약 70분 누적 시 100%)
- 종료 시 `flush_current()` 호출 → 마지막 미완성 창 저장

#### ③ 재연결 후 잼스코어 안정화 (2단계)

**문제:** API 끊김 → 재연결 시 jam_score가 순간 급락 (두 가지 원인).

**1단계 — post_skip_grace_frames (TA 업데이트 차단):**
- 재연결 직후 IDManager 속도 이력 미구성 → nm=0 차량이 stop_count 증가
- `config.py`: `post_skip_grace_frames = 30` 추가 (6fps 기준 5초)
- `detector.py`: displacement skip / timestamp gap / freeze reconnect 3개 지점에서 `_last_skip_frame` 갱신
- `_in_grace` 조건 충족 시 `traffic_analyzer.update()` 차단

**2단계 — 체류 시간 소급 부여 (dwell retroactive credit):**
- grace 기간 후 새 tid가 등록될 때 `dwell_cell_ratio`가 0으로 리셋되는 문제
- `feature_extractor.py`: 새 tid 첫 등장 시 해당 셀의 `cell_dwell_ema > 0.3`이면
  `first_frame = frame_num - dwell_thr_frames`으로 소급 설정 → 즉시 체류 셀로 인정
- `cell_dwell_ema`는 ID 무관 셀 기반 → 재연결 후에도 이전 혼잡도 보존됨

### 수정 파일
`src/gru_module.py` (삭제), `tests/test_gru_module.py` (삭제),  
`src/traffic_analyzer.py`, `src/detector.py`, `src/config.py`,  
`src/feature_extractor.py`, `src/historical_predictor.py` (신규),  
`tests/test_traffic_analyzer.py`

---

> 이전 항목(120~122차 / 2026-04-19)은 `work_log_archive.md`로 이관됨.
