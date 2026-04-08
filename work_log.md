# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-08 (78차 — 대원 81~90차 pull 반영: 웹 config·feature 동기화) [수빈]

### 오늘 한 작업

**대원 81~90차 변경사항 웹 동기화**
- `reverse_modules/config.py`: `velocity_window` 20→10, `wrong_count_threshold` 12→15 (src 동기화)
- `reverse_modules/config.py`: `min_wrongway_track_age: int = 30` 추가 (역주행 판정 최소 추적 프레임)
- `reverse_modules/config.py`: `smooth_jam_threshold` 0.30→0.25 (src 동기화)
- `reverse_modules/config.py`: `slow_upper_nm` 2.5→0.70 (src 동기화: EMA smoothing 후 원거리 정상차량 오판 방지)
- `reverse_detector.py`: `_make_x_t()` 소표본 보정 _MIN_RELIABLE 3→5, 선형→제곱 감쇠 (src/feature_extractor.py 동기화)
- `congestion_judge.py`(src/ 직접 import): 가중치 0.29/0.25/0.20/0.08 → 0.80/0.70/0.45, count 제거 자동 반영

**work_log 충돌 해결**
- HEAD(수빈 75차) + remote(대원 81~90차) 충돌 → 두 항목 모두 유지

### 수정 파일
`C:\finalPj_웹` —
- `backend_flask/modules/traffic/detectors/reverse_detector.py`
- `backend_flask/modules/traffic/detectors/reverse_modules/config.py`

### 작업 재개 위치
- 플라스크 재시작 → jam_score 정상 범위 확인

---

## 2026-04-08 (77차 — jam_score 0.4 false positive + 경부선 fallback 좌표 수정) [수빈]

### 오늘 한 작업

**jam_score 0.4 false positive 수정 (서버 재시작 후 원활 도로)**
- `reverse_detector.py`: `cong_judge.set_baseline()` → `reset()`으로 교체 (load_flow_map, 초기학습완료, 재학습완료 3곳)
- 원인: `set_baseline()`은 EMA=0.5로 초기화 → alpha_down=0.04 감소율로 30-40초간 false SLOW
- 수정: `reset()`으로 EMA=0 시작 → 실제 도로 상태로 빠르게 수렴 (원활이면 5-10초내 0.1대)

**경부선 fallback 좌표 수정**
- `its.py`: GYEONGBU_FALLBACK 좌표 전면 교체 — 마지막 점 경도 오류(126.9→128.98 직선 점프)로 이상한 직선 표시됐던 것 수정
- 수정: 서울TG→수원→오산→천안→대전→옥천→황간→김천→구미→칠곡→대구→경산→언양→부산TG 실제 경로 좌표

### 수정 파일
`C:\finalPj_웹` —
- `backend_flask/modules/traffic/detectors/reverse_detector.py`
- `backend_flask/modules/traffic/its.py`

### 발생 오류 / 확인 사항
- fallback 좌표 마지막 점 [35.1775, **128.9835**] — 앞 점들이 경도 126.9대인데 갑자기 128.98 → 직선 jump
- EMA 0.5 시작 + alpha_down=0.04: 원활도로에서도 40초간 SLOW 오분류

### 작업 재개 위치
- 플라스크 재시작 → CCTV 탐지 → jam_score 원활도로 0.1대 확인 + 경부선 경로 정상 확인

---

## 2026-04-08 (76차 — bbox_coverage·count_ratio 분모 버그 수정) [수빈]

### 오늘 한 작업

**bbox_coverage 분모 수정 (src/detector.py 동기화)**
- `reverse_detector.py`: bbox_coverage 분모를 전체 그리드(225셀) → 방향별 유효 셀 수로 변경
- `reverse_detector.py`: `_compute_direction_cell_counts()` 메서드 추가 (flow_map 유효 셀을 up/down으로 분류)
- `reverse_detector.py`: `_valid_cells_up`, `_valid_cells_down` 필드 추가 (기본값 1)
- `reverse_detector.py`: `_compute_ref_direction()` 호출 3곳에 `_compute_direction_cell_counts()` 추가 (load_flow_map, 초기학습완료, 재학습완료)

**count_ratio 분모 수정 (src/feature_extractor.py 동기화)**
- 기존: `n_known`(궤적≥20프레임 차량만) → 신규차량 무시로 count_ratio 과소평가
- 수정: `n_total`(전체 활성 차량) — src/feature_extractor.py 동일 방식
- `_make_x_t()` 시그니처 `(n_known, n_total, stop_c, slow_c, bcov)`로 변경

### 수정 파일
`C:\finalPj_웹` —
- `backend_flask/modules/traffic/detectors/reverse_detector.py`

### 발생 오류 / 확인 사항
- bbox_coverage 원근 편향: 전체 225셀 분모 → up/down 실제 유효 셀 수(flow_map count>0) 기준으로 정규화
- count_ratio 저평가: 신규 진입 차량(traj<20f)이 많을 때 count 기여 0 → 전체 차량 수 반영

### 작업 재개 위치
- 플라스크 재시작 후 CCTV 탐지 → jam_score 정상 범위 확인

---

## 2026-04-08 (75차 — jam_score 과도 상승 원인 수정 + 파일 정리) [수빈]

### 오늘 한 작업

**jam_score 근본 수정 — src/ 동기화**
- `reverse_detector.py`: `src/congestion_judge.py` 직접 import (웹 복사본 제거) → 대원 수정사항 자동 반영
- `reverse_detector.py`: `n==0` 시 세 CongestionJudge 모두 `reset()` 호출 → 차량 0대 시 즉시 0.00
- `reverse_modules/config.py`: `slow_upper_nm` 0.50→2.5, `nm_cy_correction_k` 추가 0.0 (src/config.py 동기화)
- `reverse_detector.py`: nm 계산을 `velocity_window`(20프레임) 기반으로 수정 — 2프레임 gap → 고속 차량 slow 오분류 해소
- `reverse_detector.py`: `bbox_coverage` → cell occupancy 방식으로 교체 (src/feature_extractor.py 동기화) — 원근 편향 제거
- `reverse_detector.py`: 방향별 bbox_coverage_up/down 분리 계산

**경부선 초록선 복구**
- `its.py` `highway_line()`: Overpass 실패 시 정적 fallback 좌표 반환 → 서버 재시작 후에도 선 표시

**파일 정리 (미니프로젝트 잔재 제거)**
- 삭제: `plate/`, `raspi/`, `streaming.py`, `simulation.py`, `fire_detector.py`
- 삭제: 프론트 `plate/`, `raspi/`, `stats/`, `traffic/components/`, `traffic/hooks/`, `traffic/api.js`
- `app.py`: 제거 모듈 import/register 정리
- `its.py`: `fire_feed` 라우트 제거
- `models.py`: `FireResult`, `ManualResult` 제거

**DB 초기화**
- `detection_results`, `reverse_results`, `fire_results`, `manual_results` 전부 TRUNCATE

### 수정 파일
`C:\finalPj_웹` —
- `backend_flask/modules/traffic/detectors/reverse_detector.py`
- `backend_flask/modules/traffic/detectors/reverse_modules/config.py`
- `backend_flask/modules/traffic/its.py`
- `backend_flask/app.py`
- `backend_flask/models.py`

### 발생 오류 / 확인 사항
- jam_score 과도 상승 원인: ①웹 congestion_judge가 구버전 복사본 ②nm 2프레임 기반 계산 → 모든 차량 slow 분류 ③slow_upper_nm 웹 config 미동기화

### 작업 재개 위치
- 플라스크 재시작 후 CCTV 탐지 → jam_score 정상 범위 확인

---

## 2026-04-08 (74차 — cctv_state 키 불일치 버그픽스) [수빈]

### 오늘 한 작업

**CCTV 팝업 상태 미표시 버그 수정**
- `reverse_detector.py`: `self.display_name` 추가 — cctv_name이 `{명칭}_reverse` 형태일 때 `_reverse` 접미사 제거
- `its.py` `cctv_state()`: `{name}_reverse` 키 우선 조회, 없으면 `{name}` 시도

### 수정 파일
`C:\finalPj_웹` —
- `backend_flask/modules/traffic/detectors/reverse_detector.py`
- `backend_flask/modules/traffic/its.py`

### 발생 오류 / 확인 사항
- CCTV 팝업 열어도 방향별 jam 카드가 전부 0 → 키 불일치가 원인

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

**jam_score 가중치 재보정 (slow_upper_nm=2.5 기준)**
- 기존: `0.80×slow + 0.70×stop + 0.35×sqrt(bbox)` → slow_ratio≈1.0 시 CONGESTED 오판
- 신규: `0.29×slow + 0.25×stop + 0.20×sqrt(bbox) + 0.08×count`

**slow_upper_nm 임계값 상향 조정**
- `config.py`: `slow_upper_nm` 0.50 → 2.5, `nm_cy_correction_k` 0.6 → 0.0

### 수정 파일
`src/config.py`, `src/congestion_judge.py`, `src/detector.py`,
`src/feature_extractor.py`, `src/traffic_analyzer.py`, `test_nm_live.py`

---
