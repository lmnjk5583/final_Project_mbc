# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-09 (91차 — Cell Dwell EMA 기반 jam_score 전면 재설계) [대원]

### 오늘 한 작업

**Cell Dwell EMA 도입 — jam_score 신호 전면 교체**
- `feature_extractor.py`: `_cell_dwell_ema` (20×20 ndarray) 추가 — 셀 점유 지속 시간 EMA 누적
  - 점유 중: `ema += 0.05×(1-ema)` / 빈 셀: `ema *= 0.98`
  - 정상 차량(2~5프레임/셀): peak ema≈0.10~0.23 / 정체 차량(30f+): ema→0.78+
- `feature_extractor.py`: `cell_dwell_score` 재설계 — 강도(점유셀 평균 ema) × 밀도(점유셀/차선셀) 조합
- `feature_extractor.py`: `dwell_cell_ratio` 추가 — tid별 체류(같은 셀 15f+) 셀 비율
- `feature_extractor.py`: `cell_persistence` 추가 — 2×2 코어스 그리드 Jaccard 유사도(30프레임 창) EMA
- `feature_extractor.py`: `flow_occupancy` 추가 (`bbox_coverage` 별칭 통합)
- `feature_extractor.py`: `occupied_cell_count` feature 추가 — 저규모(≤2) 차단용

**congestion_judge.py 재설계**
- 기존 `slow_cell_density + stop_cell_density + nm_variance + bbox` 수식 폐기
- 신규: `cell_dwell_score × 0.90 + flow_occupancy × 0.30` 기반
- 저규모 가드: `known_cnt≤2 or occupied_cnt≤2 or flow_occ<0.06` → 최대 0.10 반환

**config.py 파라미터 조정**
- `slow_jam_threshold`: 0.60 → 0.55
- `jam_ema_alpha_up`: 0.15 → 0.70 (정체 진입 빠른 반응)
- `gru_blend_ratio`: 0.40 → 0.0 (rule-only 모드, GRU 영향 제거)
- 신규: `dwell_threshold_frames=15`, `cell_dwell_ema_up=0.05`, `cell_dwell_ema_down=0.02`

**run_test.py 영상·모델 경로 갱신**
- 모델: `yolo11n_v1` → `yolo11n_v5`
- 영상: `임시/정체_완화_테스트.mp4` → `임시/2026-04-02_10-05-59/videos/record_2026-04-02_10-05-59.mp4`

**신규 flow_map 파일 추가**
- `flow_maps/flow_map-서행.npy`, `flow_maps/flow_map-정체.npy` (테스트용 케이스별 맵)
- `flow_maps/gru_a.pt`, `flow_maps/gru_b.pt` (방향별 GRU 체크포인트)

### 수정 파일
`src/config.py`, `src/congestion_judge.py`, `src/feature_extractor.py`, `src/traffic_analyzer.py`, `run_test.py`
추가: `flow_maps/flow_map-서행.npy`, `flow_maps/flow_map-정체.npy`, `flow_maps/gru_a.pt`, `flow_maps/gru_b.pt`

### 발생 오류 / 확인 사항
- GRU blend 0.40 시 rule 신호 희석 → blend 0.0으로 rule-only 확인 후 재조정 예정
- DEBUG print 잔존 (`congestion_judge.py` 상단 `os.path.abspath`) — 프로덕션 전 제거 필요

### 작업 재개 위치
- `run_test.py` 실행 → cell_dwell_score / jam_score 범위 확인 (원활<0.25, 서행<0.55, 정체≥0.55)
- DEBUG print 제거 후 웹 동기화 (`reverse_detector.py` + `congestion_judge.py` import 경로 확인)

---

## 2026-04-08 (79차 — 모델 고정·conf 버그·학습중 jam 차단·전체CCTV 자동시작) [수빈]

### 오늘 한 작업

**모델 경로 통일 (항상 yolo11n_v1 사용)**
- `reverse_detector.py`: GPU/CPU 분기 제거 → `C:\final_pj\runs\yolo11n_v1\weights\best.pt` 고정

**conf=0 적용 안되는 버그 수정**
- `reverse_detector.py`: `os.getenv(...) or conf or 0.35` → Python falsy로 0.0이 0.35로 치환됨
- 수정: `if _env_conf / elif conf / else 0.35` 명시적 분기로 교체

**학습 중 jam_score 계산 차단**
- `reverse_detector.py`: congestion 블록에 `not st.is_learning and not st.relearning` 가드 추가
- 이유: 학습 중엔 _ref_direction·cell_count 미설정 → bbox_coverage 오산 → jam 오염

**전체 CCTV 자동 시작 (서버 부팅 시)**
- `its.py`: `_fetch_gyeongbu_cctvs(all_cameras=True)` 파라미터 추가 — 20개 랜덤 제한 제거
- `app.py`: `_auto_start_all_detectors()` 백그라운드 스레드 추가
  - 서버 시작 8초 후 전체 경부선 CCTV detector 자동 생성
  - 3분마다 ITS URL 갱신 (TTL=4분 만료 전 토큰 갱신)
  - URL 변경 감지 시 해당 detector만 재시작

### 수정 파일
`C:\finalPj_웹` —
- `backend_flask/app.py`
- `backend_flask/modules/traffic/its.py`
- `backend_flask/modules/traffic/detectors/reverse_detector.py`

### 발생 오류 / 확인 사항
- conf `or` 체인 falsy 버그: Python에서 0.0 or 0.35 = 0.35 → 항상 0.35로 overwrite
- 학습 중 _valid_cells_up=1(기본값) → bbox_coverage = n/1 → 최대 1.0 → jam 오염

### 작업 재개 위치
- 플라스크 재시작 → 자동시작 로그 확인 ("🚀 [Auto-start] 전체 경부선 CCTV 자동 탐지 시작...")
- 내가 클릭 안 한 CCTV도 학습 진행 여부 확인

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
