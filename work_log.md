# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

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
