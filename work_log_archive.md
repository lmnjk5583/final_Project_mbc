# Work Log Archive

> work_log.md에서 이관된 과거 작업 기록. 수정하지 않음.

---

## 2026-03-28 (19차 — detector.py PassageTracker 연동 + 속도 계산 수정)

### 오늘 한 작업
1. **`src/detector.py` 전면 수정** (PassageTracker 연동 + 속도 계산 정상화)
   - `PassageTracker` import 추가
   - `__init__`: `self.passage_tracker = PassageTracker(cfg, self.state)` 생성
   - `flow.load()` 튜플 반환값 언패킹: `loaded, saved_baseline = self.flow.load(...)` 수정
   - `run()`: `TrafficAnalyzer` 생성 시 `passage_tracker` 연결
   - `run()`: `traffic_analyzer.set_state(self.state)` 호출 추가
   - `run()`: 저장된 baseline 있으면 즉시 `set_baseline()` 호출
   - per-track: `fx = (x1+x2)/2, fy = y2` footpoint 계산 추가
   - per-track: 첫 등장 시 `passage_tracker.on_entry()` 호출
   - 루프 후: 사라진 ID 감지 → `passage_tracker.on_exit()` 호출
   - 루프 후: `passage_tracker.record_frame_stats()` 매 프레임 호출
   - 학습/재학습 완료 시: `finalize_baseline()` → `set_baseline()` 연동
   - 카메라 전환 시: `passage_tracker.reset()` 추가
2. **`tests/conftest.py` 신규 생성** — `test_traffic_analyzer_legacy.py` pytest 수집 제외
3. **pytest 30/30 PASS 유지 확인**

### 수정/생성 파일
- `src/detector.py` (전면 수정), `tests/conftest.py` (신규)

---

## 2026-03-28 (20차 — 버그 2건 수정: relearning 초기값 + camera_switch NoneType 오류)

### 오늘 한 작업
1. **`src/state.py`** — `relearning = True` → `False` 수정
2. **`src/camera_switch.py`** — `set_reference()`에 `prev_small` 초기화 추가
3. **`src/detector.py`** — 초기 학습 진행률 화면 표시 추가

### 수정 파일
- `src/state.py`, `src/camera_switch.py`, `src/detector.py`

---

## 2026-03-28 (21차 — flow_map 학습 임계값 완화 + Phase 2 방향 결정)

### 오늘 한 작업
1. **`src/config.py`** — `min_move_distance`: 20.0→8.0, `min_move_per_frame`: 1.5→0.5

### 수정 파일
- `src/config.py`

---

## 2026-03-28 (22차 — 하행 중앙 차선 버스 오탐 수정)

### 오늘 한 작업
1. **`src/config.py`** — `cos_threshold`: -0.5→-0.65, `wrong_count_threshold`: 4→6, `vote_threshold`: 0.6→0.65, `grid_size`: 15→20

### 수정 파일
- `src/config.py`

---

## 2026-03-28 (23차 — 중앙 분리대 경계 보간 오탐 근본 수정)

### 오늘 한 작업
1. **`src/flow_map.py`** — `get_interpolated()` 방향 충돌 감지 로직 추가

### 수정 파일
- `src/flow_map.py`

---

## 2026-03-28 (24차 — grid_size 복원 + 충돌 임계값 수정으로 과탐 회귀 수정)

### 오늘 한 작업
1. **`src/config.py`** — `grid_size`: 20→15 복원, `learning_frames`: 500→700
2. **`src/flow_map.py`** — 충돌 감지 임계값 -0.3→-0.5

### 수정 파일
- `src/config.py`, `src/flow_map.py`

---

## 2026-03-28 (25차 — id_manager.py 구조적 오탐 버그 2건 수정)

### 오늘 한 작업
1. **`src/id_manager.py`** — `check_reappear()` flow_v=None 시 재매칭 허용 버그 수정
2. **`src/id_manager.py`** — `cleanup()` wrong_way_ids 미정리 버그 수정

### 수정 파일
- `src/id_manager.py`

---

## 2026-03-28 (26차 — flow_map 충돌 감지 수정 + id_manager 연쇄 오탐 추가 수정)

### 오늘 한 작업
1. **`src/flow_map.py`** — `get_interpolated()` 충돌 감지 임계값 -0.5→-0.4 + 폴백 None 반환
2. **`src/id_manager.py`** — 재등장 연쇄 오탐 추가 수정

### 수정 파일
- `src/flow_map.py`, `src/id_manager.py`

---

## 2026-03-28 (27차 — 중앙선 blank 구역 역주행 미탐지 수정)

### 오늘 한 작업
1. **`src/judge.py`** — `total_checked < 3` → `< 2` 수정

### 수정 파일
- `src/judge.py`

---

## 2026-03-28 (28차 — 오탐 폭증 원인 파악 및 flow_map 재학습 준비)

### 오늘 한 작업
1. 27차 롤백: `total_checked < 2` → `< 3` 복원
2. flow_map 오염 확인 → 정상 영상으로 재학습 준비
3. `run_wrongway.py` VIDEO_FILE 변경, `임시/flow_map.npy` 삭제

### 수정 파일
- `src/judge.py`, `run_wrongway.py`

---

## 2026-03-28 (29차 — 원본 방식 복원: 이중선형보간 + 공간평활화 + 임계값)

### 오늘 한 작업
1. **`src/flow_map.py`** — `get_interpolated()` 순수 이중선형보간 복원, `apply_spatial_smoothing()` 원본+보호 방식
2. **`src/config.py`** — `cos_threshold`: -0.7→-0.5, `vote_threshold`: 0.65→0.6 (원본 복원)

### 수정 파일
- `src/flow_map.py`, `src/config.py`

---

## 2026-03-28 (30차 — learn_step 방향 게이팅으로 경계 셀 오염 근본 차단)

### 오늘 한 작업
1. **`src/flow_map.py`** — `learn_step()` 방향 게이팅 추가 (cos<-0.4이면 확립 셀 갱신 거부)

### 수정 파일
- `src/flow_map.py`

---

## 2026-03-28 (31차 — 경계 셀 erosion + CCTV 글자 오탐 필터 추가)

### 오늘 한 작업
1. **`src/flow_map.py`** — `apply_boundary_erosion()` 메서드 추가
2. **`src/detector.py`** — `apply_boundary_erosion()` 호출 추가
3. **`src/judge.py`** — 궤적 점프 필터 추가
4. **`src/config.py`** — `max_jump_ratio=4.0` 추가

### 수정 파일
- `src/flow_map.py`, `src/detector.py`, `src/judge.py`, `src/config.py`

---

## 2026-03-28 (32차 — eroded_mask로 재학습 차단 + 방향 급변 필터 교체)

### 오늘 한 작업
1. **`src/flow_map.py`** — `eroded_mask` 추가
2. **`src/state.py`** — `last_velocity` 딕셔너리 추가
3. **`src/judge.py`** — 방향 급변 필터로 교체
4. **`src/config.py`** — `max_jump_ratio` 제거

### 수정 파일
- `src/flow_map.py`, `src/state.py`, `src/judge.py`, `src/config.py`

---

## 2026-03-28 (33차 — 정상→역방향 급전환 가드 필터 추가)

### 오늘 한 작업
1. **`src/state.py`** — `last_correct_frame` 딕셔너리 추가
2. **`src/judge.py`** — `direction_change_guard_frames` 가드 추가
3. **`src/config.py`** — `direction_change_guard_frames=8` 추가

### 수정 파일
- `src/state.py`, `src/judge.py`, `src/config.py`

---

## 2026-03-27 (18차 — 폴더 분리 + run_wrongway.py import 오류 수정)

### 오늘 한 작업
1. **폴더 분리**
   - `mini_project_0313-feature-data_team/` → `mini_project/` 복사 완료 (원본은 Notebook 열려있어 수동 삭제 필요)
   - `src/run_wrongway.py` → 프로젝트 루트 `run_wrongway.py`로 이동
2. **import 오류 수정** (`ModuleNotFoundError: No module named 'wrongway'`)
   - 원인: `from wrongway import ...` (존재하지 않는 패키지)
   - 수정: `from src import Detector, DetectorConfig` (`run_test.py`와 동일 방식)
3. **Phase 1 flat import 호환 수정** (`run_wrongway.py`, `run_test.py`)
   - `sys.path.insert(0, str(PROJECT_ROOT / "src"))` 추가
   - 이유: `traffic_analyzer.py`의 `from feature_extractor import ...` (flat)이 패키지 import 시 실패하는 문제 해결
4. `CLAUDE.md` 폴더 역할 표 업데이트 (`mini_project/`, `run_wrongway.py`, `run_test.py` 추가)
5. `FILE_INDEX.md` 실행 스크립트 항목 추가
6. **pytest 30/30 PASS 유지 확인**

### 수정/생성 파일
- `run_wrongway.py` (루트 신규, 기존 src/run_wrongway.py 대체)
- `run_test.py` (sys.path 수정)
- `mini_project/` (폴더 생성, 파일 19개 복사 완료)
- `CLAUDE.md`, `FILE_INDEX.md`

### 미완료 (수동 작업 필요)
- `mini_project_0313-feature-data_team/` 폴더 삭제 — Notebook 열린 채로 삭제 불가
  → Jupyter Notebook 모두 닫은 후 폴더 수동 삭제

---

## 2026-03-27 (17차 — Phase 1 TDD Step 11~12 완료, pytest 30/30 PASS)

### 오늘 한 작업
1. **Step 11** `src/traffic_analyzer.py` 내부 전면 교체
   - 기존: `_mag_to_kmh()` + `free_flow_speed` 비율 판정 (절대 km/h)
   - 신규: `FeatureExtractor` + `CongestionJudge` 조합 (baseline 대비 비율)
   - 신규 public 메서드: `set_state()`, `set_baseline()`, `get_jam_score()`
   - `CongestionPredictor` 클래스는 기존과 동일하게 유지
   - 빈 트랙(empty tracks) 처리 — early return으로 SMOOTH 유지 (TA-02 대응)
2. **Step 12** `tests/test_traffic_analyzer.py` 재작성 (TA-01~10 + CP-01~02, 12개 케이스)
   - 기존 TC-01~13 → `tests/test_traffic_analyzer_legacy.py`로 이동(비활성화)
   - 신규 테스트: `_MockState`, `_MockPassageTracker`, `_make_baseline()`, `_make_tracks()` 헬퍼 작성
3. **전체 pytest** 30/30 PASS (test_passage_tracker·test_congestion_judge·test_traffic_analyzer)

### 수정/생성 파일
- `src/traffic_analyzer.py` (내부 전면 교체, 인터페이스 유지)
- `tests/test_traffic_analyzer.py` (TA-01~10 + CP-01~02 신규)
- `tests/test_traffic_analyzer_legacy.py` (기존 TC-01~13 보존용)

---

## 2026-03-27 (16차 — dev_guide.md 분리 + CLAUDE.md 팀 공유 지침 추가)

### 오늘 한 작업
1. `CLAUDE.md` — "산출물 폴더 먼저 검토" 지침 추가 (팀 공유, 메모리 대체)
2. `Docs/dev_guide.md` — Phase 2 섹션(GRU·기술스택·알림등급·WebSocket) 분리
3. `Docs/dev_guide_phase2.md` — 신규 생성 (Phase 2 이후 참조용)
4. MD 파일 전체 간결화 진행 중

### 수정 파일
- `CLAUDE.md`, `Docs/dev_guide.md`, `Docs/dev_guide_phase2.md`, `FILE_INDEX.md`, `work_log.md`

---

## 2026-03-26 (15차 — 정체 탐지 알고리즘 설계 완료 및 MD 전면 업데이트)

### 오늘 한 작업
1. 정체 탐지 핵심 설계 확정 (pixels_per_meter 완전 제거 — ITS API CCTV 방향/각도 미제공 확인)
   - normalized_mag(mag/bbox_h) + dwell_time 기반 상대 지표 설계
   - jam_score 가중치: 0.50×속도 + 0.30×dwell + 0.20×밀도 / feature 벡터 7차원 확정
   - footpoint(y2) 사용, baseline freeze(SMOOTH 구간), LCS(학습 품질) 0~1
   - Phase 1(규칙 기반) → Phase 2(GRU 40%) → Phase 3(GRU 65%) 로드맵 확정
2. `Docs/dev_guide.md` 전면 재작성 (19개 섹션)
3. `Docs/plan.md` §3.0 모듈 목록 업데이트 / `FILE_INDEX.md` 신규 파일(4개) 추가

### 수정 파일
- `Docs/dev_guide.md` (전면 재작성), `Docs/plan.md`, `FILE_INDEX.md`

---

## 2026-03-26 (14차 — 개발 준비 완료 및 Opus 프롬프트 작성)

### 오늘 한 작업
1. `CLAUDE.md`: 프로젝트 개요/폴더역할/개발단계 업데이트, dev_guide.md 먼저 읽기 지침 추가
2. `FILE_INDEX.md`: 개발 파일 역할 표 최신화
3. `src/` 폴더 생성 + 미니프로젝트 py 전체 복사 (13개 파일)
4. `tests/` 폴더 생성
5. 기존 파일 분석 (tracks: [{id,x1,y1,x2,y2,cx,cy}], speeds: {tid:mag})
6. Opus 프롬프트 작성 (Phase 1 TDD용)

### 수정 파일
- `CLAUDE.md`, `FILE_INDEX.md`, `Docs/dev_guide.md`, `Docs/plan.md`, `src/` (신규), `tests/` (신규)

---

## 2026-03-26 (13차 — 초기화 완료 및 설계 검토)

### 오늘 한 작업
1. 미니프로젝트 원본 복구 확인 (event_publisher.py 삭제, traffic_analyzer.py 없음 확인)
2. `Docs/plan.md`: §2 정체 탐지 항목 제거, §3.0 미개발 명시
3. `수빈_노트/유스케이스_v1.0.md`: UC-06~09 상태 → 🔲 미개발
4. 메모리 정리, `개발/` 폴더 삭제

### 수정 파일
- `Docs/plan.md`, `수빈_노트/교통흐름모니터링_유스케이스_v1.0.md`

---

## 2026-03-26 (12차 — 개발 방향 재정립 및 정리)

### 오늘 한 작업
1. 팀원 Flask+React 웹에 교통흐름모니터링 탭 이미 구현 확인 → 우리 역할: 탐지코드 개발 후 깃허브 공유
2. `개발/상세설계서.md`, `개발/테스트계획서.md` 삭제
3. `Docs/plan.md` 간소화 (완성된 것 목록 + 연동 방향 + 수정 원칙만 유지)

### 수정 파일
- `개발/상세설계서.md` (삭제), `개발/테스트계획서.md` (삭제), `Docs/plan.md`

---

## 2026-03-26 (11차 — 산출물 보강 + Phase 1 TDD + 산출물 전면 일관성 검토)

### 오늘 한 작업
1. 요구사항정의서: FR-017~020 추가(관제센터 웹), API 명세 4건 추가(WS 2건+REST 2건)
2. pytest 환경 수정 (Python 3.13 호환) — pytest 70/70 PASS
3. event_publisher.py: CongestionEvent + EventPublisher 구현 (TC-31~37)
4. plan.md 기술스택: Flask+Flask-SocketIO+MySQL+React+Docker로 수정, Phase 4 추가
5. 기획서 슬라이드 6·9·10: density.py → traffic_analyzer.py 수정
6. 산출물 전체 일관성 최종 검토 완료

### 수정 파일
- `산출물/요구사항정의서_v2.0.xlsx`, `산출물/기획서_v1.0.pptx`, `Docs/plan.md`
- `wrongway/__init__.py`, `wrongway/pytest.ini`, `wrongway/event_publisher.py` (신규)

---

## 2026-03-25 (10차 — 프로젝트 방향 재정립 + research.md/plan.md 재설계)

### 오늘 한 작업
1. 산출물 불일치 파악 및 수정 (요구사항정의서 시트8 추적매트릭스, 시트5 신규모듈명, 유스케이스 UC-06~09)
2. 프로젝트 방향 재정립 (역주행 로직 불변 원칙 확립, 새 목표: 정체 탐지 → 관제센터 웹 알림 → 조치 권고)
3. 정체 해결 전략 논문 조사 (VMS, VSL, ALINEA, 사고관리, 우회경로, 서울 FTMS)
4. `research.md` §11 신규 추가 (정체 해결 전략 논문 7종 + 조치 매핑 표)
5. `plan.md` 전면 재설계 (버그수정 계획서 → 관제센터 웹 개발 계획서)

### 수정 파일
- `산출물/교통흐름모니터링_요구사항정의서_v2.0.xlsx`, `산출물/update_artifacts.py` (신규)
- `수빈_노트/교통흐름모니터링_유스케이스_v1.0.md`, `Docs/research.md`, `Docs/plan.md`

---

## 2026-03-25 (9차 — Phase 2 리팩토링: VirtualCountingLine → 궤적 기반 방향 분류)

### 오늘 한 작업
1. 라인 선 방식 → cos_sim 기반 상행/하행 분류 전면 리팩토링
2. traffic_analyzer.py: VirtualCountingLine 제거, cos_sim 방향 판정, get_direction_counts() 추가
3. visualizer.py: draw_counting_lines() 제거 → draw_direction_count() 추가
4. config.py: counting_lines 필드 제거 / run_wrongway.py: COUNTING_LINES 관련 제거
5. tests/test_phase2.py 완전 재작성 (TC-21~26) — pytest 63/63 PASS

### 수정 파일
- `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/run_wrongway.py`, `src/detector.py`, `tests/test_phase2.py`

---

## 2026-03-25 (8차 — Phase 3: 단기 정체 예측)

### 오늘 한 작업
1. `테스트계획서.md` TC-27~30 추가
2. `tests/test_phase3.py` 신규 작성 (6개)
3. `traffic_analyzer.py` CongestionPredictor 클래스 추가 (polyfit 선형 회귀 기반)
4. `visualizer.py`: draw_prediction_status() 추가 (p 키 토글)
5. `config.py`: prediction_history_window, prediction_horizon 추가 / `detector.py` 연동
   — pytest 63/63 PASS

### 수정 파일
- `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/detector.py`, `tests/test_phase3.py`, `개발/테스트계획서.md`

---

## 2026-03-25 (7차 — 카운팅 라인 사용자 설정 인터페이스)

### 오늘 한 작업
1. `config.py`: counting_lines 필드 추가
2. `visualizer.py`: show_counting_lines, l 키 토글, draw_counting_lines() 추가
3. `detector.py`: cfg.counting_lines → add_counting_line() 자동 등록
4. `run_wrongway.py`: COUNTING_LINES 변수 추가 — pytest 57/57 PASS

### 수정 파일
- `src/config.py`, `src/visualizer.py`, `src/detector.py`, `src/run_wrongway.py`

---

## 2026-03-25 (6차 — Phase 2 기능 개발)

### 오늘 한 작업
1. `상세설계서.md` §11 추가 (Phase 2 설계)
2. `테스트계획서.md` TC-21~26 추가 / `tests/test_phase2.py` 신규 작성 (6개)
3. `traffic_analyzer.py` Phase 2 확장: VirtualCountingLine, EMA, 트렌드 판정
4. `visualizer.py`: draw_congestion_heatmap() 추가 (h 키 토글)
5. `config.py`: speed_ema_alpha, speed_decline_threshold 추가 / `detector.py` 연동
   — pytest 57/57 PASS

### 수정 파일
- `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/detector.py`, `tests/test_phase2.py`

---

## 2026-03-25 (5차 — BUG-01, BUG-07 수정 + UC-06 전체 개발)

### 오늘 한 작업
1. BUG-01: detector.py cleanup 매 프레임 호출로 변경
2. BUG-07: detector.py mode 동적 결정 (LEARNING/RELEARNING/DETECTING)
3. traffic_analyzer.py 신규 구현 + 연동 (config.py, logger.py, detector.py)
4. visualizer.py: draw_congestion_status() 추가 (c 키 토글)
5. 나머지 버그 전체 수정 (BUG-02~04, 06, 08~09, ISSUE-01~05)
6. tests/test_bug_fixes.py (TC-15~20) + tests/test_integration.py (TC-11~12) 신규
   — pytest 63/63 PASS

### 수정 파일
- `src/detector.py`, `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/logger.py`, `tests/test_bug_fixes.py`, `tests/test_integration.py`

---

## 2026-03-25 (4차 — 화면설계서 검수 후 피드백 반영)

### 오늘 한 작업
1. 기획서 제목 수정 / SCR-04 헤딩 밑줄 제거 / SCR-05 예시 데이터 수정
2. SCR-09 정탐률 → 처리율 수정 / SCR-07 모바일 알림 탭 추가

### 수정 파일
- `산출물/generate_screen_design_docx.py`, `산출물/교통흐름모니터링_화면설계서.docx` (재생성)

---
