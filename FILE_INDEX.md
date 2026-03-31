# FILE_INDEX.md
# 요청 유형별 최소 읽기 파일 지도
# 이 파일은 CLAUDE.md를 짧게 유지하기 위해 분리된 파일 안내서입니다.

## 폴더 역할

| 폴더/파일 | 내용 |
|-----------|------|
| `Docs/` | 개발 기준서(dev_guide.md), 계획서(plan.md), 논문 근거(research.md) |
| `src/` | 탐지 시스템 Python 코드 전체. 미니프로젝트 py 복사본(역주행) + Phase 1 신규 파일(정체). 완성 후 팀원 웹 reverse_modules/에 이식 |
| `tests/` | pytest 테스트 파일 (TDD — 코드 작성 전 먼저 작성) |
| `mini_project/` | 미니프로젝트 Jupyter Notebook — 데이터 전처리·라벨링·YOLO 학습 (참조용) |
| `run_wrongway.py` | 역주행 탐지 실행 진입점 (루트에서 `python run_wrongway.py`) |
| `run_test.py` | 정체 탐지 연동 테스트 실행 (루트에서 `python run_test.py`) |
| `산출물/` | 기획서(PPT), WBS, 요구사항 정의서(Excel) |
| `대원_노트/` | 코드 분석 노트 |
| `수빈_노트/` | 산출물 체크리스트 |

## 요청 유형 → 읽을 파일

| 요청 유형 | 읽을 파일 (순서대로) |
|-----------|----------------------|
| 개발 관련 작업 (코드, 테스트, 설계) | **`Docs/dev_guide.md`** (Phase 1 기준 전체 포함) |
| Phase 2 이후 작업 (GRU·기술스택·알림등급) | **`Docs/dev_guide_phase2.md`** |
| 현재 상태 파악 / 어디까지 했는지 | `work_log.md` |
| 전체 계획 / 마일스톤 | `Docs/plan.md` |
| 논문 근거 / 조치 권고 상세 | `Docs/research.md` |
| 산출물 확인 (기획서, WBS 등) | `산출물/` 폴더 내 해당 파일 |
| 산출물 체크리스트 | `수빈_노트/` 폴더 내 해당 파일 |
| 전체 프로젝트 구조 파악 | CLAUDE.md (이미 로드됨) → `work_log.md` |

## 개발 파일 역할

### 기존 파일 (최소 수정)

| 파일 | 경로 | 역할 | Phase 1 변경 내용 |
|------|------|------|-------------------|
| `config.py` | `src/` | DetectorConfig 파라미터 | 신규 파라미터 추가 |
| `detector.py` | `src/` | 메인 오케스트레이터 | footpoint·baseline freeze·feature 추출 연동 |
| `state.py` | `src/` | 런타임 상태 관리 | `entry_positions` 필드 추가 |
| `id_manager.py` | `src/` | ID 관리·재매칭 | `cleanup()`에 passage 기록 추가 |
| `flow_map.py` | `src/` | 15×15 흐름 벡터 학습 | `speed_ref` 배열·`learn_baseline()` 추가 |
| `visualizer.py` | `src/` | 시각화 | 정체 시각화 메서드 추가 |
| `logger.py` | `src/` | CSV 로거 | jam_score·정체레벨 컬럼 추가 |
| `traffic_analyzer.py` | `src/` | 정체 탐지 (기존) | 내부 교체 (인터페이스 유지) |

### 신규 파일 (Phase 1)

| 파일 | 경로 | 역할 |
|------|------|------|
| `baseline_stats.py` | `src/` | PassageRecord·BaselineStats 데이터 클래스 |
| `passage_tracker.py` | `src/` | 차량 진입/퇴장 기록·dwell 집계·LCS 산출 |
| `feature_extractor.py` | `src/` | 매 프레임 7차원 feature 벡터 계산 |
| `congestion_judge.py` | `src/` | jam_score 계산·레벨 판정·히스테리시스 |

### 신규 파일 (Phase 2)

| 파일 | 경로 | 역할 |
|------|------|------|
| `gru_module.py` | `src/` | GRU 추론·온라인 학습·camera_switch 대응 |

### 실행 스크립트 (프로젝트 루트)

| 파일 | 경로 | 역할 |
|------|------|------|
| `run_wrongway.py` | 루트 | 역주행 탐지 실행 진입점 (`detect_only=True`) |
| `run_test.py` | 루트 | 정체 탐지 연동 확인용 테스트 실행 (`detect_only=False`) |

> 두 파일 모두 `src/` 패키지 + Phase 1 flat import를 모두 지원하도록 sys.path 이중 설정.

### 테스트 파일

| 파일 | 경로 | 역할 |
|------|------|------|
| `test_traffic_analyzer.py` | `tests/` | TrafficAnalyzer 통합 테스트 (Phase 1 기준 재작성) |
| `test_passage_tracker.py` | `tests/` | PassageTracker 단위 테스트 |
| `test_congestion_judge.py` | `tests/` | jam_score·레벨 판정 테스트 |

### 문서 파일

| 파일 | 경로 | 역할 |
|------|------|------|
| `dev_guide.md` | `Docs/` | **개발 기준서 Phase 1** — 알고리즘 설계·모듈 명세·인터페이스 |
| `dev_guide_phase2.md` | `Docs/` | **개발 기준서 Phase 2** — GRU·기술스택·알림등급·WebSocket |
| `plan.md` | `Docs/` | 전체 개발 계획, 마일스톤 |
| `research.md` | `Docs/` | 논문 근거, 조치 권고 매핑 |
| `교통흐름모니터링_유스케이스_v1.0.md` | `수빈_노트/` | 유스케이스 명세서 — **§7에 UC↔SCR 대응표 포함** (팀원 연동 시 참조) |

## 파일 추가 시 이 파일만 수정하면 됩니다.
