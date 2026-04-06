# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

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
- 기존 문제: 차량 많아도 smooth jam=0.01~0.05 (bbox_coverage 분자 작아 단독 효과 미미)
- bbox_coverage 가중치: 0.25 → 0.35
- count_ratio 항 추가: `+0.10 × min(count_ratio, 1.0)`
- smooth 차량 많음 기대치: 0.04 → 0.15

### 수정 파일 [대원]
`src/congestion_judge.py`

### 발생 오류 / 확인 사항
- 없음

### 작업 재개 위치 [대원]
- run_test.py 실행 후 smooth/congested jam_score 실측 확인

---

## 2026-04-06 (68~72차 — 역주행 오탐 근본 재설계 + bbox_coverage + nm 테스트) [대원]

### 오늘 한 작업 [대원]

**역주행 오탐 가드 — edge detection 방식으로 전면 재설계**
- 기존: 마지막 정상 판정 프레임(`last_correct_frame`) 기준 경과 시간 체크 → 신규 등장 차량 무력화 문제
- 신규: 방향이 "급변하는 순간(edge)"을 감지해 그 시점부터 guard_frames 동안 보호
  - `stable_velocity`: 정상 투표 통과 시 방향 벡터 저장
  - `direction_change_frame`: cos < cos_threshold(0.0) 이탈 순간 기록
  - `direction_was_stable`: 직전 프레임 안정 여부 → edge 감지용
  - early-exit 가드: 투표 루프 진입 전 guard 확인 → cos 연산 스킵
- `config.py`: `direction_change_cos_threshold = 0.0` 추가

**BBoxStabilizer 제거 + footpoint 중앙(cy)으로 변경 + 3프레임 warmup**
- BBoxStabilizer(alpha=0.5 EMA): 초기 이상 bbox가 후속 프레임에 오염 → 제거
- footpoint: `fy = (y1+y2)//2` (하단 y2 → 중앙 cy)
- 3프레임 warmup: `_age = frame_num - first_seen_frame < 3` 이면 궤적 추가 건너뜀

**bbox_coverage — density_score 대체**
- 기존 density_score: `vehicle_count / density_max_vehicles(40)` → 실탐지 20대도 드문 환경에서 실효 없음
- bbox_coverage: `Σbbox면적 / (flow_map 유효 셀 수 × 셀 면적)` — 차선 수 독립
- feature_extractor.py에 `"bbox_coverage"` 추가, density_score는 alias로 유지

**fallback jam_score 공식 교체**
- 기존: `0.50×slow + 0.30×stop + 0.20×density`
- 신규: `0.60×slow + 0.60×stop + 0.25×bbox_coverage` (가중치 합 1.45, 정체 시 1.0 초과 가능 → clip)

**nm 테스트 코드 작성**
- `tests/test_nm_measurement.py`: nm 계산 단위 테스트 8케이스 (pytest)
- `test_nm_live.py`: 실제 영상에서 차량별 nm 실시간 표시 (빨강=stop, 주황=slow, 초록=normal)

**버그 수정**
- `AttributeError: NoneType.count`: detector.py에서 `TrafficAnalyzer(flow_map=self.flow)` 미전달 → 수정
- `UnboundLocalError: _learn_min_mag`: if 블록 안에서만 정의 → 블록 바깥(공통 위치)으로 이동

### 수정 파일 [대원]
`src/config.py`, `src/state.py`, `src/judge.py`, `src/id_manager.py`,
`src/detector.py`, `src/feature_extractor.py`, `src/congestion_judge.py`,
`tests/test_nm_measurement.py` (신규), `test_nm_live.py` (신규)

### 발생 오류 / 확인 사항
- `AttributeError: 'NoneType' object has no attribute 'count'` → 수정 완료
- `UnboundLocalError: local variable '_learn_min_mag' referenced before assignment` → 수정 완료
- flow_map.npy는 footpoint가 y2 → cy로 변경되었으므로 **재학습 필요** (detect_only=False 실행)

### 작업 재개 위치 [대원]
- `run_test.py` detect_only=True 재실행으로 오류 수정 확인
- 역주행 오탐 edge detection 가드 실제 영상 검증
- flow_map 재학습 후 jam_score(bbox_coverage 포함) 실제 영상 테스트

---

## 2026-04-03 (68차 — 웹 정체 모니터링 탭 구현) [수빈]

### 오늘 한 작업 [수빈]

**웹 정체 탭 (탄소배출/정체 탭) 구현**
- 탄소 제외 결정 (팀원 중복 + 별도 개발 필요) → 정체 전용(안 A)으로 확정
- Phase 3 미래 예측 구현 + 의존성 파일 동기화 (plan.md, 프로그램설계서, 인터페이스명세서)
- CLAUDE.md 동기화 체크리스트 추가 (누락 방지)

**구현 파일 (C:\finalPj_웹)**
- `backend_flask/shared/state.py`: congestion_state, congestion_events, jam_score_history 추가
- `backend_flask/modules/carbon/carbon.py`: `/status` `/jam_history` `/history` `/stats` API 4개 구현
- `frontend_js/src/modules/carbon/api.js`: 신규 — API 호출 함수 4개
- `frontend_js/src/modules/carbon/index.jsx`: 전체 재작성
  - 현재 상태 카드 (레벨 배지 + jam_score 게이지)
  - KPI 3개 (오늘 이벤트, 최대 지속시간, GRU 예측)
  - jam_score 실시간 SVG Sparkline 차트
  - GRU 확률 바 (p_smooth/p_slow/p_congested)
  - 정체 이벤트 로그 테이블

### 수정 파일 [수빈]
`C:\finalPj_웹` 4개 파일, `c:\final_pj` src/gru_module.py, src/config.py, dev_guide_phase2.md, plan.md, 산출물 2개, CLAUDE.md

### 발생 오류 / 확인 사항
- 차트 라이브러리 미설치 (recharts/chart.js 없음) → SVG polyline으로 직접 구현

### 작업 재개 위치 [수빈]
- 백엔드 실제 탐지기 연동: CongestionDetector가 shared.congestion_state를 갱신하는 연결 작업 필요
- 현재는 API 구조만 완성, 탐지기 미연결 상태 (초기값 SMOOTH 반환)

---

