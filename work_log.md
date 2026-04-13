# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-13 (93차 — 역주행 오탐 수정 + W키 단축키) [대원]

### 오늘 한 작업

**역주행 오탐 수정 (순간이동 오감지 근본 원인 제거)**
- `judge.py`: `dir_jump_filtered` 조기 리턴 시 `direction_change_frame` 즉시 세트 + `wrong_way_count=0`
  - 기존: 방향 급변 감지 후 1프레임 공백 → 후속 프레임에서 wrong_count 누적 가능
  - 수정: 급변 감지와 동시에 120프레임 guard 발동 → 완전 차단
- `detector.py`: solo_jump 기준 배수 `1.5 → 1.2` (단독 차량 순간이동 탐지 범위 확대)

**역주행 패널 W키 토글**
- `visualizer.py`: `show_wrongway=True` 플래그 추가, `W`키 핸들러 등록
- `detector.py`: `_show_as_wrong` 변수로 역주행 시각화 통합 게이팅
  - W키 OFF → 역주행 차량도 일반 박스(초록)로만 표시, 경고 텍스트/붉은 궤적 숨김

### 수정 파일
`src/judge.py`, `src/detector.py`, `src/visualizer.py`

### 발생 오류 / 해결
- dir_jump_filtered가 direction_change_frame을 세트하지 않아 guard 1프레임 공백 → 즉시 세트로 해결

### 작업 재개 위치
- `python run_its_live.py` 실행 후 역주행 오탐 빈도 확인
- 여전히 오탐 발생 시: `wrong_count_threshold` 15→20 상향 검토

---

## 2026-04-12 (92차 — GRU 직접예측·연속학습·ITS 실시간 실행기) [대원]

### 오늘 한 작업

**구조적 버그 수정**
- `congestion_judge.py`: 모듈 상단 DEBUG print(`import os`, `print(f"[DEBUG]...")`) 제거
- `congestion_judge.py`: `set_baseline()` 내 `print(f"[CJ] alpha_up=...")` 제거
- `feature_extractor.py`: `_ema_flat` 미사용 dead code 제거
- `feature_extractor.py`: `_lane_cell_count // 2` 버그 수정 — `_valid_cell_count_override` 지정 시 이미 단방향값인데 다시 /2 해서 cds 2배 과대평가 → 조건부 분기로 수정
- `congestion_judge.py`: `compute_jam_score_fallback()` cds 가중치 1.10→1.90 (위 버그 보정치 교정)

**파라미터 개선**
- `config.py`: `jam_ema_alpha_up` 0.70→0.40 (0.70은 EMA≈raw값으로 비대칭 설계 무의미)
- `config.py`: `dwell_threshold_frames=15` → `dwell_threshold_sec=0.5` (FPS 독립적 설계)
- `feature_extractor.py`: `__init__(fps)` 파라미터 추가 → `_dwell_thr_frames = int(0.5 * fps)` 변환
- `config.py`: `congestion_hysteresis_sec` 3.0→7.0 (순간 변동으로 레벨 오락가락 방지)

**초기 확정 구간 추가**
- `config.py`: `initial_confirm_sec=5.0`, `initial_hysteresis_sec=2.0` 추가
- `congestion_judge.py`: `apply_level()` — 학습 완료 후 5초간 2초 히스테리시스 적용, 이후 정규 7초 전환
- `congestion_judge.py`: `reset()` 시 `_baseline_frame=None` 초기화

**GRU Direct Prediction Head (1분·3분·5분 예측)**
- `gru_module.py`: `_FEATURE_KEYS` 수정 — 실제 feature_extractor 출력키와 불일치하던 3개 키 교정
  - 제거: `count_ratio`, `exit_rate_ratio`, `dwell_ratio`
  - 추가: `norm_speed_ratio`, `slow_ratio`, `flow_occupancy`, `cell_dwell_score`, `cell_persistence`, `rule_jam_score`
- `gru_module.py`: `_GRUNet.direct_heads` — horizon별 독립 FC 헤드(hidden→32→3) ModuleList 추가
- `gru_module.py`: `pretrain()` 확장 — self-supervised MSE 후 direct head CrossEntropy 학습
- `gru_module.py`: `predict_direct()` 추가 — 현재 hidden state → horizon별 level+confidence 반환
- `config.py`: `gru_predict_horizons_sec=(60,180,300)`, `gru_pretrain_min_sec=600.0`, `gru_direct_epochs=10` 추가

**GRU 연속 학습 (세션 간 데이터 누적)**
- `gru_module.py`: `append_feature_log(features, path)` — pickle append로 디스크 누적
- `gru_module.py`: `load_feature_log(path)` — 누적 pickle 로드
- `gru_module.py`: `retrain_from_log(path)` — 전체 누적 데이터로 재학습
- `config.py`: `gru_log_interval=3`, `gru_retrain_interval_sec=3600.0` 추가
- `detector.py`: `_log_path_a/b`, 1시간마다 `retrain_from_log()` 자동 실행
- `detector.py`: `online_step` SMOOTH 전용 → 3레벨(SMOOTH/SLOW/CONGESTED) 모두 학습
- `config.py`: `gru_blend_ratio` 0.0→0.20 (GRU 활성화)

**스트림 실행 지원**
- `detector.py`: `run(video_name, max_seconds=None)` — HTTP/RTSP 스트림 감지, 읽기 실패 시 sleep+continue
- `detector.py`: `result_dir=None` 시 VideoWriter 생성 건너뜀, write/release None 체크 추가

**ITS 실시간 실행기 신규 생성**
- `run_its_live.py`: ITS API로 CCTV 스트림 URL 조회 → 200초마다 자동 갱신 무한 루프
- `.env`의 `ITS_API_KEY` 사용, CCTV별 독립 저장 폴더 (`flow_maps/[CCTV_NAME]/`)
- HLS(M3U8) 응답 자동 감지 → URL 직접 스트림으로 사용
- `DIRECT_STREAM_URL` 옵션 — API 없이 스트림 URL 직접 지정 가능

### 수정 파일
`src/config.py`, `src/congestion_judge.py`, `src/feature_extractor.py`,
`src/traffic_analyzer.py`, `src/gru_module.py`, `src/detector.py`
신규: `run_its_live.py`

### 발생 오류 / 해결
- `ITS_CCTV_API_URL`에 HLS 스트림 URL 입력 시 JSON 파싱 실패 → Content-Type 감지로 자동 처리
- `result_dir=None`일 때 `_get_next_filename()` TypeError → VideoWriter 조건부 생성으로 수정

### 작업 재개 위치
- `python run_its_live.py` 실행 → 1800프레임 flow_map 학습 완료 후 GRU feature 누적 시작 확인
- 10분 후 pretrain 로그(`🧠 GRU-A pretrain 완료`) 확인

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

### 수정 파일
`src/config.py`, `src/congestion_judge.py`, `src/feature_extractor.py`, `src/traffic_analyzer.py`, `run_test.py`
추가: `flow_maps/flow_map-서행.npy`, `flow_maps/flow_map-정체.npy`, `flow_maps/gru_a.pt`, `flow_maps/gru_b.pt`

### 발생 오류 / 확인 사항
- GRU blend 0.40 시 rule 신호 희석 → blend 0.0으로 rule-only 확인 후 재조정 예정
- DEBUG print 잔존 (`congestion_judge.py` 상단 `os.path.abspath`) — 92차에서 제거 완료

### 작업 재개 위치
- 92차에서 전면 개선 완료 (GRU 활성화, 예측 헤드 추가, ITS 실행기 생성)

---
