# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-03-31 (59차 — 프로젝트 경로 C드라이브 이전 및 CLAUDE.md 정리)

- **이전 내용**: 프로젝트 전체를 `N:\개인\대원&수빈\최종 프로젝트` → `C:\final_pj`로 이동 (데이터 폴더 `임시`·`capture`는 N드라이브에 유지)
- **CLAUDE.md**: 시스템 아키텍처·다음 개발 단계 섹션 삭제(완료된 내용), 작업 환경 경로 `C:\final_pj`로 수정
- **경로 주석 수정**: `src/` 8개 파일, `tests/` 4개 파일, `run_wrongway.py`, `src/train/train_yolo11n.py` (PROJECT_ROOT), `수빈_노트/pdf캡처.py`
- **재개**: 다음 개발 작업 진행

---

## 2026-03-30 (58차 — 신규/정지 차량 stops_ratio 분리 재설계)

- `detector.py`: 신규 차량(궤적 < velocity_window) speeds 미등록, `mag_val is not None`만 등록
- `feature_extractor.py`: 신규 제외(`tid not in speeds` → continue), `mag <= 0` → 정지 카운트, `mag > 0` → nm 계산, 분모=`speed_known_count`
- **재개**: 재학습 → 상행 0~0.1 / 하행 0.7+ 확인

---

## 2026-03-30 (57차 — feature_extractor speeds=0 오염 수정)

- `feature_extractor.py`: `mag <= 0` 차량 완전 제외(nm·정지 모두 스킵), 분모=`len(norm_mags)`
- 원인: `speeds[tid]=0` 기본값 → 신규 차량 전부 정지 카운트됨
- **재개**: 재학습 → 상행 0~0.1, 하행 0.7+ 확인

---

## 2026-03-30 (56차 — 47~55차 롤백: 46차 상태 복원)

- `feature_extractor.py`: IQR/markers/effective_active 제거, 단순 nm+상위50% 중앙값 복원
- `congestion_judge.py`: warmup 30프레임 제거
- `config.py`: `jitter_speed_multiplier` 제거, `base_speed_threshold=7.0` 복원
- `judge.py`: jitter 보정 제거, `cos_threshold` 직접 사용 복원
- `detector.py`: speeds 기본값 `0` 복원, fallback `return 'a'` 복원
- **재개**: 46차 동작 확인 후 새 접근으로 정체 개선

---

## 2026-03-30 (55차 — 정지 차량 stop_ratio 누락 수정 + flow_map 롤백)

- `detector.py`: speeds 마커 3단계(`-1`신규/`0`정지/`>0`이동)
- `feature_extractor.py`: 3단계 기반 stop_ratio(`-1`제외, `0`정지카운트, `>0` nm계산)
- `flow_map.py`: 52차 2칸 거리 제한 롤백(하행 flow 미채움 문제)
- **재개**: 재학습 → DOWN CONGESTED, UP SMOOTH 확인

---

## 2026-03-30 (54차 — speeds=0 차량 nm 계산 버그 수정)

- `feature_extractor.py`: `mag > 0.0`만 nm 계산(기본값 0이 nm=0으로 stopped 폭등하던 버그)
- **재개**: 상행 jam_score 0.1 이하 복구 확인

---

## 2026-03-30 (53차 — flow=None fallback 개선)

- `detector.py`: `_classify_direction()` flow_v=None 시 같은 x열 가장 가까운 유효 셀로 판단(기존: 무조건 'a')
- 영향: 정체 차량이 전부 A로 편입 → UP도 CONGESTED 오판 해소
- **재개**: UP SMOOTH, DOWN CONGESTED 분리 확인

---

## 2026-03-30 (52차 — norm_speed_ratio 통일 + flow_map 2칸 확장)

- `feature_extractor.py`: 현재·기준 모두 전체 중앙값 사용(기존: 상위50% vs 전체 불일치)
- `flow_map.py`: count=0 셀 채움을 Chebyshev 거리 ≤ 2로 제한(50차 과도 차단 조정)
- **재개**: 재학습 → 원활 상행 jam 0.1~0.2 이하 확인

---

## 2026-03-30 (51차 — stop_ratio 과대 버그 수정)

- `feature_extractor.py`: 신규 차량 stopped_count 완전 제외, nm 집계 차량만으로 산출
- 신뢰도 가중 혼합 로직 제거 → 단순화
- `congestion_judge.py`: warmup EMA 정리
- **재개**: 원활 상행 jam 0.1 이하 확인

---

## 2026-03-30 (50차 — jam 급등 억제 + flow_map 불필요 영역 차단)

- `congestion_judge.py`: set_baseline 후 warmup 30프레임(양방향 alpha_up 빠른 수렴)
- `flow_map.py`: 이웃 count>0이 4개 이상일 때만 채움(하늘·갓길 차단)
- **재개**: 재학습 → flow_map 확인, 원활 jam 0.5→0.1 빠른 하강 확인

---

## 2026-03-30 (49차 — bbox jitter 역주행 오탐 수정)

- `judge.py`: 속도 게이트~`jitter_speed_multiplier`배 구간에서 cos_threshold 선형 강화(최대 30%)
- `config.py`: `jitter_speed_multiplier=2.5` 추가
- **재개**: 정체 구간 역주행 오탐 억제 + 실제 역주행 탐지율 확인

---

## 2026-03-30 (48차 — EMA 초기값 0.5 + jam 계산식 재보정)

- `congestion_judge.py`: EMA 초기값 0→0.5, synergy항 추가(`speed×stop×0.10`), 가중치 `0.55/0.35/0.10`
- `config.py`: `jam_ema_alpha_up` 0.15→0.10
- **재개**: 원활 0.1 이하, 극심 정체 0.85+ 확인

---

## 2026-03-30 (47차 — IQR outlier 필터)

- `feature_extractor.py`: nm에 IQR fence(`Q3+1.5×IQR`) 적용, 초과 차량 제외
- **재개**: DOWN 정체 시 버스 필터링, UP 원활 시 전부 유효 확인

---

## 2026-03-30 (46차 — dwell 제거, speed+stop 가중치 재설계)

- `congestion_judge.py`: dwell 삭제, `speed×0.60+stop×0.25+density×0.15`, 보너스 조건 `speed>0.9`
- fallback도 `stop×0.50` 중심으로 변경
- **재개**: UP jam≈0, DOWN jam≈0.4+ 확인

---

## 2026-03-30 (45차 — 비대칭 EMA + 카메라 전환 reset)

- `config.py`: `jam_ema_alpha_up=0.15`, `jam_ema_alpha_down=0.04`
- `congestion_judge.py`: 비대칭 EMA + `reset()` 메서드
- `detector.py`: 카메라 전환 시 CJ reset
- **재개**: jam_score 부드러운 변화 확인

---

## 2026-03-30 (44차 — UP SLOW 근본 수정 + 원근 보정)

- `detector.py`: fallback lcs 0.5→0.0(smooth_thr 0.24→0.30 복원)
- `feature_extractor.py`: nm 대표값 → 상위 50% 중앙값, stop_ratio 신뢰도 가중 혼합
- **재개**: UP SMOOTH, DOWN SLOW/CONGESTED 확인

---

## 2026-03-30 (43차 — 한글→ASCII + SLOW 임계값 + 궤적 미생성 처리)

- `visualizer.py`: 한글 6곳 → 영문 전환
- `config.py`: `smooth_jam_threshold` 0.25→0.30
- `feature_extractor.py`: 신규 차량 nm 평균 제외, stopped_count에만 포함
- **재개**: UP SMOOTH 확인

---

## 2026-03-30 (42차 — 속도 기준값 보정 + bbox_h 클램프)

- `config.py`: `norm_stop_threshold` 0.08→0.05, `min_bbox_h=30.0` 추가
- `feature_extractor.py`: `bbox_h = max(raw, 30)` 클램프
- `detector.py`: fallback `norm_speed_ref` 0.5→0.15
- **재개**: UP SMOOTH / DOWN CONGESTED 확인

---

## 2026-03-30 (41차 — UP/DOWN 판별 수정 + flow_map 다수결 erosion)

- `detector.py`: `ref_vy < 0` → UP(화면 위=UP 통일)
- `flow_map.py`: 2단계 다수결 erosion(3×3 반대방향 ≥40% → 제거)
- **재개**: 재학습 → UP/DOWN 레이블·역주행 오탐 확인

---

## 2026-03-30 (40차 — 정지 판단 원근 보정 + 히스테리시스 완화)

- `config.py`: `norm_stop_threshold=0.08`, `congestion_hysteresis_sec` 15→5
- `feature_extractor.py`: `mag/bbox_h < 0.08` 기준으로 교체
- **재개**: DOWN CONGESTED, UP SMOOTH 확인

---

## 2026-03-29 (39차 — 방향별 차선 분리 구현)

- `config.py`: `lane_cos_threshold=0.0`
- `visualizer.py`: 듀얼 패널(A/B 방향별 레벨+bar+지속시간+조치권고)
- `detector.py`: 듀얼 PT/GRU/TA/Predictor, `_compute_ref_direction()`, `_classify_direction()`, footpoint 기반 분류, 방향별 독립 갱신
- **재개**: DOWN SLOW/CONGESTED 표시 확인

---

## 2026-03-29 (38차 — visualizer·logger 정체 연동)

- `visualizer.py`: `draw_congestion_status()` 좌하단 패널(레벨+bar+조치권고)
- `logger.py`: `jam_score`, `정체레벨` 컬럼 추가
- `detector.py`: 시각화·logger 연동
- **재개**: 정체 패널 화면 표시 확인

---

## 2026-03-29 (37차 — GRU 버그 2건 수정)

- `gru_module.py`: predict() hidden state 미갱신 → `self._hidden = ...` 수정
- `detector.py`: pretrain 데이터 경로 수정(`_gru_feature_history` 별도 리스트)
- `test_gru_module.py`: GRU-05 수정, GRU-10 복원, GRU-11 추가
- **재개**: GRU pretrain 실행 확인

---

## 2026-03-29 (36차 — Phase 2 GRU 모듈 개발 완료)

- `test_gru_module.py`: GRU-01~10 TDD
- `gru_module.py`: GRUNet(2층 hidden=64), push/predict/pretrain/online_step/reset, PyTorch fallback
- `config.py`: GRU 파라미터 8개
- `congestion_judge.py`: `compute_jam()`/`apply_level()` 분리
- `traffic_analyzer.py`: GRU 블렌딩
- `detector.py`: GRU 생명주기 연동
- **재개**: 실영상 GRU 동작 확인

---

## 2026-03-29 (35차 — sudden_change_rejected + global_ok 다중 위치 조회)

- `judge.py`: wrong_count 도달 시 `fsf-lcf ≤ guard` → 확정 거부+리셋, global_ok 현재→시작→중간 순 조회
- 12000프레임 오탐 0건 확인
- **재개**: Phase 2 설계 또는 웹 이식

---

## 2026-03-29 (34차 — last_correct_frame 설정 경로 정리)

- `judge.py`: slow/total_checked<3에서 lcf 설정 제거, global_traj_ok에 추가, 가드 발동 시 wrong_count=0 리셋
- **재개**: CCTV 글자 오탐 제거 + 역주행 탐지 확인