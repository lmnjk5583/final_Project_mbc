# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-01 (63차 — flow_map 개선 + 시각화 개편 + 파라미터 튜닝)

### 오늘 한 작업

**flow_map 학습 품질 개선 (3개 파일)**
- `flow_map.py`: smoothed_mask 추가 — 보간 채움 셀 추적, 실 데이터 유입 시 자동 해제, save/load 포함
- `detector.py`: 중간 평활화 150프레임 주기 → 80%/95% 시점 플래그 방식으로 변경 (정확히 1회씩)
- `judge.py`: smoothed_mask 셀 cos_threshold 완화 — 단기 -0.50, 장기 -0.60, 전체 궤적 -0.75 유지

**시각화 개편**
- `visualizer.py`: 정체 패널 1개(380px 좌하단) → 2개(Down 좌하단, Up 우하단, 190px씩) 분리
- 조치 권고 텍스트: 반투명 배경 추가, 폰트 크기 0.38→0.52, 두께 1→2
- `C`키 단축키 추가 — 정체 패널 ON/OFF 토글
- 단축키 안내에 `[C] 정체 패널` 추가

**파라미터 튜닝**
- `norm_stop_threshold`: 0.10 → 0.06 (원거리 bbox_h 클램프 오판 방지)
- `density_max_vehicles`: 20 → 40 (방향별 10대 원활 시 density 과대평가 방지)
- `density_score` 정규화 로직 추가 (`feature_extractor.py`)
- `direction_change_guard_frames`: 45 → 90 (점진적 방향 전환 오탐 대응)
- `wrong_count_threshold`: 5 → 8 (사용자 직접 수정)
- `vote_threshold`: 0.60 → 0.70 (사용자 직접 수정)
- `slow_jam_threshold`: 0.55 → 0.42 → 0.55 복원 → 0.60 (사용자 직접 수정)

### 수정 파일
`src/flow_map.py`, `src/detector.py`, `src/judge.py`, `src/visualizer.py`, `src/feature_extractor.py`, `src/config.py`

### 발생 오류 / 확인 사항
- 정체 jam_score 최대 0.53 → density 정규화로 0.9+ 도달 가능하게 수정
- UP 원활 → SLOW 오탐: norm_stop_threshold + density_max_vehicles 조정으로 해결
- 역주행 오탐 지속: wrong_count_threshold/vote_threshold 강화로 완화 중

### 작업 재개 위치
- 역주행 오탐 추가 검증 (wrong_count_threshold=8, vote_threshold=0.70 적용 후)
- 다양한 영상으로 임계값 검증 필요
- Passage 중복 카운트 문제 (낮은 우선순위)

---

## 2026-04-01 (62차 — 정체 탐지 baseline 설계 확정 + 코드 수정)

- **설계 확정**: flow_map(역주행 전용) / 정체 판정(fallback + LCS=0.36) 분리
  - flow_map.npy → 방향 학습·역주행 탐지 전용. 카메라 전환 시 재학습
  - 정체 판정 → 항상 fallback 모드 (stop_ratio + density). 카메라 무관
  - LCS=0.36 (한강 측정값) → 임계값 보정에만 사용. baseline.npy 별도 불필요
- **수정 파일**:
  - `src/config.py`: `default_lcs=0.36` 파라미터 추가
  - `src/detector.py`: 3곳 수정
    - run() 시작: 항상 fallback baseline (LCS=0.36) 설정
    - 학습 완료: set_baseline 제거, flow_map만 저장
    - 재학습 완료: 동일
- **다음 작업**: run_test.py로 실행 확인 (detect_only=False → 학습 후 정체 탐지)

---

## 2026-04-01 (61차 — LCS 수정 완료 확인 + 설계 방향 논의)

- 베이스라인 재학습 결과: `passage=8404, lcs=0.36` 확인 (0.96→0.58→0.36)
- LCS 0.36 = 품질 점수, 실제 판정 기준은 `norm_speed_ref` 임을 정리
- 설계 이슈 논의: 임의 카메라 대응 문제 → 62차에서 해결

---

## 2026-03-31 (60차 — LCS compute_lcs() 3단계 수정)

- `passage_tracker.py`: `compute_lcs()` 총 3차례 수정 (0.96 → 0.58 → 0.36)
  - signal_A: `percentile(5)/5` → `percentile(10)/8` — 자연 편차 흡수 + 범위 보정
  - signal_B: `max` → `percentile(95)` — bbox 튐 이상치 방지
  - signal_C: `exit/active(≈0.977 고정)` → `std(dwells)/median/2` — 변동계수 기반 교체
- **수정 파일**: `src/passage_tracker.py`

---

## 2026-03-31 (59차 — 프로젝트 경로 C드라이브 이전 및 CLAUDE.md 정리)

- 프로젝트 전체 `N:\개인\대원&수빈\최종 프로젝트` → `C:\final_pj` 이동
- `src/` 8개, `tests/` 4개, 실행 스크립트 2개 경로 주석 수정
- CLAUDE.md 완료 섹션 정리

---
