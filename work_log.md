# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

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

## 2026-04-06 (73~76차 — normal mode 제거 + jam_score 밀도 보정 + 방향 분류 nearest-neighbor) [대원]

### 오늘 한 작업 [대원] — 추가

**B방향(상행) jam_score 저평가 원인 분석 및 수정**
- 원인: flow_map 상단 rows 0~6 (y < 252px) 전체 미학습 → `get_interpolated` = None → 기본값 'a' → 상행 차량 전부 A방향으로 오분류
- track_log + flow_map 분석으로 확인: B방향 speed-known 차량 2대, 상단 미학습 115/400 셀
- `flow_map.py`: `get_nearest_direction(x, y)` 추가 — count>0 셀 중 그리드 거리 최소 셀 벡터 반환 (20×20 브루트포스)
- `detector.py._classify_direction`: `flow_v=None` 시 `get_nearest_direction` 호출 → 최종 None이면 'a' fallback

**jam_score 공식 sqrt(bbox_coverage) 적용**
- bbox weight: 0.25 → 0.35, 변환: linear → sqrt
- count_ratio 항 추가 후 제거 (count_ref=15 > 실탐지 10대로 효과 미미)
- 최종: `0.80×slow + 0.70×stop + 0.35×sqrt(bbox_coverage)`

### 수정 파일 [대원]
`src/congestion_judge.py`, `src/flow_map.py`, `src/detector.py`

### 발생 오류 / 확인 사항
- B방향 jam 0.07 → nearest-neighbor 적용 후 0.12~0.18 예상

### 작업 재개 위치 [대원]
- run_test.py 재실행 후 B방향 jam_score 실측 확인

---

## 2026-04-06 (73~74차 — normal mode 제거 + jam_score 밀도 보정) [대원]

### 오늘 한 작업 [대원]

**congestion_judge.py normal mode 전면 제거**
- `compute_jam_score()` 함수 삭제 (LCS 기반 정상 모드)
- `from baseline_stats import BaselineStats` import 제거
- `self.baseline` → `self._baseline_set: bool` 단순화
- `set_baseline()` 파라미터 무시 (`_baseline=None`)
- `compute_jam()`: if/else 분기 → 항상 `compute_jam_score_fallback()` 호출
- `get_smooth_threshold()` / `_get_slow_threshold()`: LCS 보정 제거 → 고정 임계값 반환

**jam_score smooth 구간 바닥 점수 개선**
- bbox_coverage 가중치: 0.25 → 0.35
- count_ratio 항 추가: `+0.10 × min(count_ratio, 1.0)`

### 수정 파일 [대원]
`src/congestion_judge.py`

### 작업 재개 위치 [대원]
- run_test.py 실행 후 smooth/congested jam_score 실측 확인

---

## 2026-04-06 (68~72차 — 역주행 오탐 근본 재설계 + bbox_coverage + nm 테스트) [대원]

### 오늘 한 작업 [대원]

**역주행 오탐 가드 — edge detection 방식으로 전면 재설계**
- 기존: last_correct_frame 기준 경과 시간 체크 → 신규 등장 차량 무력화 문제
- 신규: 방향 급변 순간(edge) 감지 후 guard_frames 동안 보호
- `config.py`: `direction_change_cos_threshold = 0.0` 추가

**BBoxStabilizer 제거 + footpoint 중앙(cy) + 3프레임 warmup**

**bbox_coverage — density_score 대체**
- `Σbbox면적 / (flow_map 유효 셀 수 × 셀 면적)` — 차선 수 독립

**fallback jam_score 공식 교체**
- 신규: `0.60×slow + 0.60×stop + 0.25×bbox_coverage`

**nm 테스트 코드 작성**
- `tests/test_nm_measurement.py`, `test_nm_live.py`

### 수정 파일 [대원]
`src/config.py`, `src/state.py`, `src/judge.py`, `src/id_manager.py`,
`src/detector.py`, `src/feature_extractor.py`, `src/congestion_judge.py`,
`tests/test_nm_measurement.py` (신규), `test_nm_live.py` (신규)

### 발생 오류 / 확인 사항
- `AttributeError: NoneType.count` → 수정 완료
- `UnboundLocalError: _learn_min_mag` → 수정 완료
- flow_map.npy 재학습 필요 (footpoint y2 → cy 변경)

### 작업 재개 위치 [대원]
- 역주행 오탐 edge detection 가드 실제 영상 검증
- flow_map 재학습 후 jam_score(bbox_coverage 포함) 실제 영상 테스트

---
