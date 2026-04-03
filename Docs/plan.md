# 교통 이상징후 탐지 시스템 — 개발 계획서

> 기준 문서: `research.md`
> 최종 수정: 2026-03-26
> 핵심 원칙: **역주행 탐지 로직 불변. 정체 탐지 + 정체 이벤트 외부 전달 및 관제 대응 시스템만 추가.**

---

## 1. 프로젝트 목표

```
[기존 — 완료]                        [추가 목표]
역주행 탐지 (로컬 영상)    +    정체 탐지 → 정체 이벤트 외부 전달 → 관제 대응 시스템 → 정체 해결 조치 권고
```

역주행 탐지 시스템은 완성된 상태. 이 위에 정체 탐지 결과를 외부 관제 시스템으로 전달하고,
정체 심각도에 따라 논문 기반 해결 조치를 운영자에게 제시하는 것이 목표.

---

## 2. 현재 완성된 것 (수정 금지)

| 구분 | 모듈 | 상태 |
|------|------|------|
| 역주행 탐지 | detector.py, judge.py, id_manager.py, flow_map.py, tracker.py, bbox_stabilizer.py, camera_switch.py, state.py | ✅ 완료 — §5 수정 원칙 적용 |

---

## 3. 새로 추가할 기능

### 3.0 정체 탐지 모듈 (Phase 1·2 완료 — 2026-04-02)

**설계 원칙**: 절대 km/h 없이 학습 기준 대비 상대 비율로 정체 판단.
상세 설계는 `Docs/dev_guide.md` §3~10 참조.

| 구분 | 모듈 | 내용 |
|------|------|------|
| feature 추출 | `feature_extractor.py` (신규) | 7차원 feature 벡터 (normalized_mag, dwell_ratio 등) |
| 통과 기록 | `passage_tracker.py` (신규) | 차량 진입/퇴장 dwell 집계 → BaselineStats 산출 |
| 정체 판정 | `congestion_judge.py` (신규) | jam_score 계산·레벨 판정·히스테리시스 |
| 정체 탐지 (래퍼) | `traffic_analyzer.py` (개조) | 기존 인터페이스 유지, 내부 교체 |
| GRU 모듈 | `gru_module.py` (완료, Phase 2) | 시계열 패턴 학습·추론 |
| baseline 자료구조 | `baseline_stats.py` (신규) | PassageRecord·BaselineStats 데이터 클래스 |
| 기존 파일 확장 | `state.py`, `id_manager.py`, `flow_map.py` | 최소 수정 (§15 원칙 적용) |
| 정체 시각화 | `visualizer.py` 확장 | draw_congestion_status, draw_congestion_heatmap |
| 로깅 | `logger.py` 확장 | frame_log에 jam_score·정체레벨 컬럼 추가 |
| 설정 | `config.py` 확장 | 신규 파라미터 추가 |

**개발 순서 (Phase 1 → 2 → 3)**:
```
Phase 1: 규칙 기반 jam_score (normalized_mag + dwell + count)   ✅ 완료
Phase 2: GRU 병렬 추가 (40% 혼합)                              ✅ 완료
Phase 3: GRU primary (65%) + 미래 예측 — dev_guide_phase2.md §13 참조  🔧 구현 중
```

### 3.1 정체 해결 조치 권고 (research.md §11 기반)

| 정체 레벨 | 조치 | 논문 근거 |
|-----------|------|-----------|
| SLOW | VSL 하향 권고 (100→80→60 km/h) | Papageorgiou et al. 2006 |
| SLOW | VMS 서행 안내 | MDPI Infrastructures 2024 |
| CONGESTED | VMS 우회 경로 안내 | MDPI Infrastructures 2024 (우회율 +18%) |
| CONGESTED | 램프 미터링 가동 권고 | ALINEA (Papageorgiou 1991, TTS -10~50%) |
| CONGESTED 지속 (5분+) | 순찰대 출동 요청 | FHWA CHART (처리시간 -11분) |

### 3.2 개발 방향 및 연동 구조

**개발 위치**: `최종 프로젝트/src/` (미니프로젝트 py 전체 복사 + 신규/수정 파일 개발)

**수정 대상 파일**: `dev_guide.md` §0 체크리스트 및 §2 개발 파일 위치 참조.

**팀원 웹 통합 구조 (Flask + React + MySQL)**:
```
backend_flask/modules/traffic/detectors/
    ├── reverse_modules/          ← 우리 src/ 전체 이식
    └── reverse_detector.py       ← traffic_analyzer 호출 지점
backend_flask/modules/carbon/
    └── carbon.py                 ← 팀원이 정체 API 라우트 추가
frontend_js/src/modules/carbon/
    └── index.jsx                 ← 팀원이 정체 UI 개발
```

**역할 분담**:
- 대원: `src/` 개발 완료 → 깃허브 공유
- 수빈: 화면설계서 기반 React 웹 화면 구성 (뼈대) 담당
- 팀원: carbon.py API 라우트 + WebSocket + MySQL 연동

**DB**: 역주행 + 정체 이벤트 모두 저장 (MySQL, 기존 DetectionResult 모델 활용 또는 신규 모델 추가)

**인터페이스** (어떤 데이터를 넘길지): 개발하면서 결정

---

## 4. 차량 감지 모델 고도화 (추가학습)

### 목적
현재 YOLO11n은 일반 사전학습 모델을 사용 중. 고속도로 CCTV 환경에 특화된 데이터로 추가학습하여 오탐/미탐 감소.

### 방법
```
데이터 수집 (임시/ 폴더)
    └── 고속도로 CCTV 차량 이미지 + 라벨
         ↓
    YOLO11n 추가학습 (Fine-tuning)
         ↓
    기존 best.pt 교체 → 탐지 정확도 향상
```

### 현재 상태
- 데이터 수집 중 (`임시/` 폴더)
- 학습 완료 후 `config.py`의 `model_path` 경로만 교체하면 연동 완료

---

## 5. 역주행 로직 수정 원칙

> 상세 파일별 가이드: **`Docs/dev_guide.md §13`** 참조.

- 역주행 로직은 원칙적으로 수정하지 않는다.
- 새 기능 구현에 직접 영향을 주는 버그 발견 시만 수정 가능:
  - 대원_노트/ 폴더에 버그 내용 먼저 기록
  - 수정 후 기존 테스트 전체 PASS 필수
  - judge.py 핵심 알고리즘은 수정 범위 제외
