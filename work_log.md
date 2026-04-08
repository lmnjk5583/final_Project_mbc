# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-07 (81~90차 — 대규모 리팩토링: 데드코드 제거 + 역주행 오탐 개선) [대원]

### 오늘 한 작업

**데드코드 완전 제거**
- `src/baseline_stats.py`, `src/passage_tracker.py`, `src/bbox_stabilizer.py` 삭제
- `detector.py`: PassageTracker/BaselineStats import 제거, flow.load() bool만 반환, on_entry/on_exit/reset/enough_passages 블록 제거
- `id_manager.py`: passage_tracker 파라미터 제거
- `traffic_analyzer.py`: set_baseline() 인자 없는 버전으로 단순화
- `feature_extractor.py`: BaselineStats import 제거, set_ready() 메서드 추가, count_ratio 반환 제거
- `flow_map.py`: save/load에서 baseline_stats 직렬화 제거, 버전 3으로 bump
- `state.py`: entry_positions 필드 제거

**역주행 오탐 개선**
- `judge.py`: `min_wrongway_track_age=30` — 신규 등장 30프레임 이내 판정 차단
- `judge.py`: `direction_change_cos_threshold=0.3` — 72°+ 방향 급변 시 guard 트리거 (기존 90°+)
- `detector.py`: footpoint EMA smoothing(alpha=0.4) — bbox jitter가 trajectory에 전파되기 전 흡수
- `config.py`: `wrong_count_threshold=15`, `velocity_window=10`

**jam_score 수식 재보정 (count 제거)**
- `congestion_judge.py`: count_contribution 제거, `0.80×slow + 0.70×stop + 0.45×sqrt(bbox)`
- `config.py`: `smooth_jam_threshold=0.25`, `slow_jam_threshold=0.60`
- 4차선 도로 차량 대수 과다→CONGESTED 오판 방지

**config.py 정리**
- CongestionPredictor용 `free_flow_speed=100.0`, `prediction_history_window=30` 복구
- `enable_online_flow_update`, `pixels_per_meter`, `default_lcs`, `min_passage_dist` 등 불필요 파라미터 제거

### 수정 파일
`src/config.py`, `src/congestion_judge.py`, `src/detector.py`, `src/feature_extractor.py`,
`src/traffic_analyzer.py`, `src/flow_map.py`, `src/state.py`, `src/id_manager.py`,
`src/judge.py`, `run_test.py`, `run_wrongway.py`
삭제: `src/baseline_stats.py`, `src/passage_tracker.py`, `src/bbox_stabilizer.py`

### 발생 오류 / 확인 사항
- `enable_online_flow_update` unexpected keyword → run_test.py에서 제거
- `prediction_history_window` AttributeError → config.py CongestionPredictor 섹션에 재추가
- GRU의 count_ratio: `.get(k, 0.0)` fallback으로 안전하게 처리됨

### 작업 재개 위치
- run_test.py 재실행으로 정상 기동 확인
- 경부선 CCTV 영상으로 역주행 오탐 감소 실측 확인

---

## 2026-04-07 (77~80차 — 상행선 서행 탐지 공정성 개선) [대원]

### 오늘 한 작업

**bbox_coverage 원근 편향 제거 — cell occupancy 방식으로 교체**
- 기존: `Σbbox면적 / road_area` → 근거리 bbox가 4~5배 과대 계산
- 신규: 차량 footpoint(cx, cy)가 점유한 셀 수 / 방향별 유효 셀 수 (원근 무관)
- `feature_extractor.py`: `set_valid_cell_count(n)` 메서드 추가
- `traffic_analyzer.py`: `set_valid_cell_count(n)` 포워딩 추가

**방향별 유효 셀 수 분리 계산**
- `detector.py`: `_valid_cells_a`, `_valid_cells_b` 필드 + `_compute_direction_cell_counts()` 추가
- 학습 완료/재학습 완료/detect_only 로드 3곳에서 호출

**velocity 기반 방향 분류 조기 적용 (3프레임~)**
- 기존: flow_map 기반 → 상단 미학습 셀은 'a' fallback → 상행 차량 오분류
- 신규: trajectory 3프레임 이상이면 속도 벡터 코사인으로 A/B 판정
- velocity_window(20) 미만 구간도 조기 분류 가능 → 원거리 차량 tracks_b 포함률 향상

**jam_score 가중치 재보정 (slow_upper_nm=2.5 기준)**
- 기존: `0.80×slow + 0.70×stop + 0.35×sqrt(bbox)` → slow_ratio≈1.0 시 CONGESTED 오판
- 신규: `0.29×slow + 0.25×stop + 0.20×sqrt(bbox) + 0.08×count`
- 설계 목표: 원활→0.08, 서행→0.39, 정체→0.61

**slow_upper_nm 임계값 상향 조정**
- 실측 nm 분포: 서행(20~40 km/h) → nm ≈ 0.8~2.5 (avg=1.15)
- `config.py`: `slow_upper_nm` 0.50 → 2.5, `nm_cy_correction_k` 0.6 → 0.0 (이중보정 비활성화)
- `test_nm_live.py` 상수도 동기화

### 수정 파일
`src/config.py`, `src/congestion_judge.py`, `src/detector.py`,
`src/feature_extractor.py`, `src/traffic_analyzer.py`, `test_nm_live.py`

### 발생 오류 / 확인 사항
- 하행 SLOW→CONGESTED: slow_upper_nm 1.0 적용 시 slow_ratio≈1.0 → 구 공식 가중치 과도 → 가중치 재보정으로 해결
- .pyc 캐시 오염 → `__pycache__` 전체 삭제로 해결

### 작업 재개 위치
- run_test.py 재실행 후 상행(B) jam_score SLOW(≥0.30) 확인
- 상행 slow_ratio ≥ 0.6 달성 여부 실측 확인

---

## 2026-04-07 (69차 — 웹 교통 정체 모니터링 UI 고도화) [수빈]

### 오늘 한 작업

**경부고속도로 실제 도로 형상 시각화 (Overpass OSM)**
- VWorld 타일 제거 → CartoDB Positron 단일 레이어 (OSM 좌표 정합)
- `/api/its/highway_line`: Overpass API 3-미러 fallback, 1312 세그먼트 캐시
- simCenter 반경(0.35°) 내 세그먼트만 정체 색상, 나머지 초록 유지

**CCTV 스트림 아키텍처 변경**
- Nimble Streamer nimblesessionid IP-bound → 서버가 토큰 URL 반환, 브라우저 직접 연결
- `/api/its/stream_cctv`: 전체 ITS 목록에서 name 검색 (20개 샘플 제거)

**UI 정리**
- 미사용 탭 전부 제거, "교통흐름모니터링" 단일 탭만 유지

**YOLO 모델 연동**
- `C:\final_pj\runs\yolo11n_v1\weights\best.pt` 연동 (reverse_detector.py)
- CCTV 팝업: MJPEG 탐지 스트림 전폭 표시, HLS 제거

**상행/하행 분리 jam_score**
- `reverse_detector.py`: dx 부호로 상행(서울)/하행(부산) 분리, 방향별 EMA 계산
- `shared/state.py`: `jam_up`, `jam_down`, `level_up`, `level_down`, `count_up`, `count_down` 추가
- `carbon.py /status`: 6개 신규 필드 반환
- `index.jsx`: `DirectionJamCard` 컴포넌트 — 상행/하행 카드 나란히 표시 (전체 jam 바·차량 대수 제거)
- 지도 높이 600px, 팝업 오버레이 스크롤 가능

### 수정 파일
`C:\finalPj_웹` — state.py, carbon.py, its.py, reverse_detector.py, index.jsx, api.js, traffic/index.jsx, Sidebar.jsx

### 발생 오류 / 확인 사항
- Overpass Gateway Timeout → 3-mirror fallback으로 해결
- nimblesessionid 400 → 브라우저 직접 연결로 해결
- dx 기반 방향 판단: CCTV 촬영 방향에 따라 상행/하행 라벨 반전 가능 → 실차 테스트 필요

### 작업 재개 위치
- 상행/하행 라벨이 실제 촬영 방향과 맞는지 확인
- `C:\finalPj_웹` 깃 업로드 예정 (현재 미업로드)

---
