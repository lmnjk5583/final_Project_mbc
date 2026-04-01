# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-01 (61차 — LCS 수정 완료 확인 + 설계 방향 논의)

- 베이스라인 재학습 결과: `passage=8404, lcs=0.36` 확인 (0.96→0.58→0.36)
- LCS 0.36 = 품질 점수, 실제 판정 기준은 `norm_speed_ref` 임을 정리
- **설계 이슈 논의**: 임의 카메라 대응 문제
  - 현재 구조: 카메라별 flow_map.npy 1개 필요 → 임의 카메라 전환 시 대응 불가
  - `norm_speed_ref`는 카메라 각도·높이 의존 → 범용 불가
  - 범용 대안: `stop_ratio` 단독 사용 (카메라 무관) 또는 카메라별 사전 수집
  - **미결**: 임의 카메라 대응 방향 결정 필요
- **재개**: 설계 방향 결정 → stop_ratio 기반 or 현재 구조 유지 선택

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
