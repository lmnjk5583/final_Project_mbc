# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-24 (130차 — 플로우맵 스냅샷 영속성 + 다중 도로 격리) [대원]

### 오늘 한 작업

#### ① 플로우맵 스냅샷 영속성 시스템 신규 구현

**목적:** 학습을 마칠 때마다 타임스탬프 파일로 저장하고, 재실행 시 유사한 스냅샷을 자동 로드해 학습을 완전히 스킵.

**`src/flow_map_matcher.py`:**
- `save_flow_snapshot(frame, flow_map_obj, save_dir)` 추가
  - `flow_map_YYYYMMDD_HHMMSS.npy` + `ref_frame_YYYYMMDD_HHMMSS.jpg` 쌍으로 저장
  - 기존 `save_ref_frame()` + 단일 `flow_map.npy` 저장 방식 대체
- `find_best_snapshot(current_frame, save_dir, min_score=0.35)` 추가
  - 타임스탬프 쌍 + 레거시 단일 파일 모두 검색
  - ORB(0.7) + 히스토그램(0.3) 혼합 유사도 비교 → 최고 점수 npy 경로 반환

**`src/detector.py`:**
- `run()` 시작 시 스냅샷 자동 매칭 블록 추가 (`init_grid` 직후)
  - 첫 프레임 peek → `find_best_snapshot()` → 매칭 시 로드 + 학습 스킵
  - 파일 입력이면 cap을 첫 프레임으로 되감기
  - 스냅샷 없으면 기존 학습 경로로 진행
- 초기 학습 / 재학습 완료 저장: `save_flow_snapshot()` 으로 교체

#### ② camera_id 기반 스냅샷 격리

**문제:** 다른 고속도로 CCTV가 동일 상위 폴더의 스냅샷을 오매칭하는 현상.

**`src/config.py`:** `camera_id: str = ""` 필드 추가
- 설정 시 `flow_map_path.parent/camera_id/` 서브폴더에만 저장·검색
- 미설정 시 기존처럼 `flow_map_path.parent` 사용

**`src/detector.py`:** `_snapshot_dir()` 헬퍼 메서드 추가
- `camera_id` 유무에 따라 격리 디렉터리 반환
- 저장·검색·출력 로그 모두 이 메서드 경유

#### ③ `run_its_live.py` 구조 정리

**기존 문제:** `_try_match_flow_map()` 이 `flow_maps/` 전체를 스캔해 다른 도로 npy를 복사 → 교차 오매칭 발생.

**수정:**
- `_try_match_flow_map()` 전면 제거 (Detector 내부 스냅샷 시스템으로 완전 대체)
- `make_config(cctv_url)` → `make_config()`: URL 파라미터 불필요
- `detect_only=False` 고정: 학습 스킵 여부는 Detector.run() 내부에서 결정
- `FORCE_RELEARN=True` 시 타임스탬프 스냅샷(`flow_map_*.npy`, `ref_frame_*.jpg`)도 함께 삭제

**다중 고속도로 동시 탐지 격리:**
- 각 프로세스의 `ROAD_DIR = flow_maps/[경부선] 양재/` 처럼 이미 분리됨
- `find_best_snapshot()` 가 해당 ROAD_DIR 안에서만 검색 → 교차 오매칭 원천 차단
- 처음 학습하는 도로는 스냅샷 없음 → 즉시 학습 시작 (다른 도로 스냅샷 무시)

### 수정 파일
`src/flow_map_matcher.py`, `src/detector.py`, `src/config.py`, `run_its_live.py`

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

> 이전 항목(123차 / 2026-04-22)은 `work_log_archive.md`로 이관됨.
