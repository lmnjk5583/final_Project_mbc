# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-02 (64차 — 문서 체계 정비 + 화면설계서 v1.1 재생성)

### 오늘 한 작업

**문서 체계 정비**
- `CLAUDE.md`: guide.md 운영 규칙 확립 (산출물↔guide 동기화 원칙 명시), 수빈_노트/대원_노트 참조 제외, N드라이브 산출물 경로·읽기 방법·충돌 보고 형식 고정
- `FILE_INDEX.md`: N드라이브 산출물 경로 명시, 인터페이스명세서 v1.2 경로 업데이트
- `Docs/dev_guide.md`: §0 체크리스트 전체 ✅ 업데이트 (Phase 1·2 완료 반영), guide 역할 명시, 파라미터 기준값 실제값과 동기화
- `Docs/dev_guide_phase2.md`: Phase 1·2 완료 상태 명시, **Phase 3 설계 명세 신규 추가** (§13)
- `Docs/plan.md`: Phase 1·2 완료 / Phase 3 미구현 상태 반영, 역할 분담 업데이트

**화면설계서 v1.1 재생성**
- `generate_screen_design_docx.py` 수정: km/h → 정상 대비 속도(%) 전환 (7곳), LSTM → GRU, 정체 판정 기준 jam_score 기반으로 변경, 개정이력 v1.1 추가
- 재생성 후 `N:\개인\대원&수빈\최종 프로젝트\산출물\교통흐름모니터링_화면설계서.docx` 덮어쓰기 완료

**산출물 충돌 잔존 확인** (수빈이 직접 수정 필요)
- 인터페이스 명세서 v1.2: 표지 버전 표기·§1.1 Phase 상태·§4.4.2 임계값·§2.2 C키 누락 (4곳)
- 프로그램설계서: §4.1·§8.1·§8.2 파라미터값 (3곳)

### 수정 파일
`CLAUDE.md`, `FILE_INDEX.md`, `Docs/dev_guide.md`, `Docs/dev_guide_phase2.md`, `Docs/plan.md`, `work_log.md`, `work_log_archive.md`
`N드라이브: generate_screen_design_docx.py`, `교통흐름모니터링_화면설계서.docx`

### 발생 오류
- python-docx 미설치 → cv 환경에 설치 완료

### 작업 재개 위치
- 웹 화면 구성(React 뼈대) 작업 시작 예정
- 산출물 인터페이스명세서·프로그램설계서 수동 수정 대기 중 (수빈)

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

