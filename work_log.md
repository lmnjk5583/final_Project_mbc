# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-08 (75차 — jam_score 과도 상승 원인 수정 + 파일 정리) [수빈]

### 오늘 한 작업

**jam_score 근본 수정**
- `reverse_detector.py`: `src/congestion_judge.py` 직접 import (웹 복사본 제거) → 대원 수정사항 자동 반영
- `reverse_detector.py`: `n==0` 시 세 CongestionJudge 모두 `reset()` 호출 → 차량 0대 시 즉시 0.00
- `reverse_modules/config.py`: `src/config.py` 와 jam_score 관련 파라미터 전체 동기화 (slow_upper_nm 등)
- `reverse_detector.py`: nm 계산을 velocity_window 기반으로 수정 (traj[-3] → traj[-1×velocity_window])

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
