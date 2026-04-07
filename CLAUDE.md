# CLAUDE.md

## 프로젝트 개요

고속도로 CCTV 영상 기반 **교통 흐름 모니터링 시스템**
- **박대원**: 정체 탐지 모듈 개발·백엔드 | **이수빈**: 기획·산출물·프론트·QA
- 작업 브랜치: `feature/Traffic_Flow_Monitoring` | 환경: Windows, `C:\final_pj`

---

## ⛔ 절대 규칙 (어떤 요청과도 상충 시 이 규칙 우선)

1. **`feature/Traffic_Flow_Monitoring` 브랜치에서만 작업** — main 직접 push 금지
2. **세션 시작**: `git checkout feature/Traffic_Flow_Monitoring` → `git pull` → `CLAUDE.md` → `work_log.md` 순으로 실행
3. **세션 종료**: `work_log.md 업데이트` → `git add .` → `git commit` → `git push`
4. **한 단계씩 진행** — 다음 스텝까지 한꺼번에 알려주지 않는다
5. **개발 작업 시 `dev_guide.md` + `dev_guide_phase2.md` 먼저 읽는다**

---

## 금지 사항

| 금지 | 이유 |
|------|------|
| `수빈_노트/`, `대원_노트/` 읽기 | 작업 자료 없는 개인 메모 |
| Glob / Grep 전수 조사 | 불필요한 탐색 비용 |
| `src/` 내부에서 run 스크립트 실행 | import 경로 오류 발생 |
| `.docx` / `.xlsx` 직접 수정 시도 | 수정 불가 포맷 — `Docs/산출물/*.md` 사용 |
| main 브랜치 직접 push | 완성 후 PR로만 병합 |

---

## 파일 읽기 규칙 (조건부 분기)

```
IF 개발 작업 (src/)          → Docs/dev_guide.md + dev_guide_phase2.md 먼저 읽기
IF 웹 개발 작업 (finalPj_웹) → C:\finalPj_웹\dev_guide_web.md 먼저 읽기
IF 산출물 참고/수정          → Docs/산출물/ 해당 파일 직접 편집 (10개 파일 — FILE_INDEX.md 참조)
IF 파일 미지정 요청          → FILE_INDEX.md 먼저 읽고 최소 파일만 선택
IF 명시적 요청 파일          → 그 파일만 읽는다
```

> **Good** ✅ 웹 작업 시 `dev_guide_web.md` 읽고 §3 현황표 확인 후 코드 수정
> **Good** ✅ `dev_guide.md` + `dev_guide_phase2.md` 읽고 인터페이스 확인 후 코드 수정
> **Bad** ❌ FILE_INDEX.md 없이 Glob 전체 탐색하거나 `.docx`를 직접 읽으려 시도

---

## 문서 동기화 원칙

**3개 계층이 항상 동기화 상태를 유지해야 한다:**

```
plan.md + research.md
        ↓
dev_guide.md (Phase 1)    dev_guide_phase2.md (Phase 2)
        ↕                           ↕
 인터페이스명세서.md          화면설계서.md
 프로그램설계서.md     ←———— (공통 영향)
        ↕
src/config.py  →  파라미터 변경 시 dev_guide.md §A 역방향 동기화

plan.md ↔ 기획서.md  (WBS·일정 동기화)
```
- 기획서.md는 개발 참조 대상 아님 (배경·일정 전용)
- 상세 의존성 → [Docs/conventions.md](Docs/conventions.md)

- `.md` 파일 충돌 발견 시 **그 자리에서 즉시 수정**
- 상세 의존성 맵 → [Docs/conventions.md](Docs/conventions.md)

### ⚡ 동기화 체크리스트 — 개발 작업 완료 시 필수

코드 또는 dev_guide 수정 후 아래 표에서 수정한 파일 행을 찾아 **우측 파일을 즉시 동기화**한다:

| 수정한 파일 | 반드시 동기화할 파일 |
|-------------|----------------------|
| `src/config.py` | `dev_guide.md §A`, `인터페이스명세서.md §4`, `프로그램설계서.md §8` |
| `dev_guide.md` | `plan.md`, `인터페이스명세서.md`, `프로그램설계서.md` |
| `dev_guide_phase2.md` | `인터페이스명세서.md`, `프로그램설계서.md`, `화면설계서.md`, `plan.md` |
| `src/*.py` (신규 메서드·클래스) | `프로그램설계서.md §3`, `인터페이스명세서.md` 해당 섹션 |
| **`finalPj_웹` 코드 변경** | **`dev_guide_web.md` §3 현황표 + 해당 섹션** |
| `finalPj_웹/shared/state.py` | `dev_guide_web.md §5` |
| `finalPj_웹/carbon.py` API 추가 | `dev_guide_web.md §4-A` |
| `finalPj_웹/its.py` API 추가 | `dev_guide_web.md §4-B` |

> 이 표는 conventions.md의 요약본. 동기화 누락 시 산출물과 코드가 불일치 상태가 됨.

---

## 개발 방식

- 모든 개발은 `src/`에서 진행. `run_wrongway.py`·`run_test.py`는 **프로젝트 루트**에서 실행
- **TDD**: 코드 작성 전 테스트 먼저 설계
- 복잡도 높은 작업(여러 파일 수정)은 **Opus 적합성 먼저 알리고 프롬프트 작성**
- 코드 제공 시: **파일명+전체 경로 명시**, **모든 줄에 주석**

---

## 응답 방식

- 시각화 중요시 — 흐름·구조는 다이어그램/표로 표현
- 코드 실행 시 저장 파일이 있으면 **저장 경로와 파일명** 반드시 기재
- 답변 전 빠진 내용·수정할 것 한 번 더 검토
- 프롬프트 요청 시 대화 전체를 정리해 다음 세션용 프롬프트 작성

---

## work_log.md 관리

- 구조: `오늘 한 작업 / 수정 파일 / 발생 오류 / 작업 재개 위치` — 간결하게
- **당일 + 전날** 항목만 유지 — 2일 이전 항목은 `work_log_archive.md` **하단에 append** (archive는 읽지 않고 추가만 — 토큰 절약)
- 커밋 형식 예시: `[대원] 정체탐지 threshold 로직 구현` / `[수빈] QA 체크리스트 업데이트`
