# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-15 (107차 — FlowMap apply_direction_repair 추가) [대원]

### 오늘 한 작업

**프레임 스킵 방향 오류 교정 로직 추가**

**문제**: 카메라 끊김 시 차량 순간이동 → velocity_window 기반 방향 벡터가 반대 방향으로 계산 → 셀이 반대 방향으로 학습됨
(`_is_frame_skip`으로 대부분 차단되지만 미탐지 끊김·일부 프레임 누락이 남음)

**수정 (flow_map.py)**: `apply_direction_repair()` 메서드 추가
- 학습 완료 후 각 셀을 3×3 이웃 평균 방향과 비교
- 경계 구역(이웃끼리 방향 불일치) 여부를 이웃 일관성으로 판별 → 건너뜀
  - avg_mag ≈ 0: 이웃 방향이 상쇄됨 → 중앙선 경계 → 교정하지 않음
  - consistent_count < min_consistent_neighbors: 이웃 불일치 → 경계 → 교정하지 않음
- 이웃 일관성 충분 + 이 셀만 반대(cos_vs_avg < -0.3) → 이웃 평균으로 덮어씌움
- smoothed_mask=True로 표시 (판정 완화 대상)

**수정 (detector.py)**: 초기학습·재학습 완료 시퀀스에 ③ 추가
```
① apply_spatial_smoothing(verbose=True)
② apply_overlap_erosion()
③ apply_direction_repair()   ← NEW
④ apply_spatial_smoothing()
```

### 수정 파일
`src/flow_map.py`, `src/detector.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- flow_map.npy 삭제 후 `python run_its_live.py` 재시작
- 학습 완료 로그에서 `🔧 direction_repair: N셀 교정` 확인
- N이 과도하게 크면(50+ 셀) `repair_cos_threshold` -0.3 → -0.5로 강화 검토

---

## 2026-04-15 (106차 — FlowMap 중앙점 즉시 덮어씌움 + 궤적 방향 학습) [대원]

### 오늘 한 작업

**FlowMap learn_step 전면 재설계 — 중앙점 게이팅 제거 + 궤적 방향 학습**

**근본 원인:**
- `_bbox_contra_count`가 dist=1 셀에서도 쌓임
  - 반대 차선 bbox edge가 살짝 넘어온 것만으로도 정상 셀이 침식 대상에 포함
  - 1800프레임 × 많은 차량 = dist=1 셀도 contra_count >= 8 도달 → 상행선 하단 대량 삭제
- 중앙점(dist=0)에 방향 게이팅이 적용 → 오염 셀 위를 지나가도 방향이 갱신되지 않음

**수정 (flow_map.py):**
```
dist=0 (중앙점): 게이팅 완전 제거 → 항상 방향 갱신
                 궤적 방향(traj[0]→traj[-1]) 우선 사용 (velocity_window 노이즈 차단)
                 _bbox_contra_count: 중앙점에서만 추적 (실제 중심 침범만 카운트)
dist=1 (인접):   게이팅 유지, contra·count 감소 없음 (정상 셀 보호)
dist≥2:          게이팅 없음, count 미증가 (soft 유지)
```

**수정 (detector.py):**
- `learn_step` 호출 전 궤적 방향 계산: `traj_ndx = (fx - traj[0][0]) / tmag`
- 충분한 이동거리(>= min_move_distance)일 때만 궤적 방향 전달

**기대 효과:**
- 상행선 하단 셀 공백: dist=1에서 contra 미추적 → overlap_erosion 대상 급감
- 중앙점 즉시 덮어씌움: 오염 셀 위로 정상 차량이 지나가면 바로 방향 반영
- 중앙선 경계: 실제 중심이 침범한 차량만 contra 카운트 → 정밀한 경계 검출

### 수정 파일
`src/flow_map.py`, `src/detector.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- flow_map.npy 삭제 후 `python run_its_live.py` 재시작 → 재학습
- `overlap_erosion: N셀 제거` N이 대폭 줄었는지 확인
- 상행선 하단 셀이 채워지는지, 중앙선 근처 오탐 감소 확인

---

## 2026-04-15 (105차 — FlowMap count 감소 범위 수정 + 이관) [대원]

### 오늘 한 작업

**FlowMap count 감소 로직 버그 수정**

**문제 1 — 상행선 하단 셀 공백**
- 원인: `count -= 1`이 dist==1 셀에도 적용 → 학습 중 경계 인접 셀 count가 하행 bbox 방문마다 감소
  - dist=1 셀: 상행 bbox 방문 count+1 vs 하행 bbox 방문 count-1 → net ≈ 0
  - count≈0 셀은 smoothing에서 중앙선 경계 충돌(이웃 방향 반대)로 채우기 거부 → 공백
  - 부가: count < min_samples 상태에서 `_bbox_contra_count`도 미누적 → 오염 셀 침식 안 됨 → 오탐 원인

**수정**: `flow_map.py` — count 감소를 `dist == 0` (bbox 중앙점 셀)에만 제한
```python
if bbox is not None and dist == 0:
    self.count[r, c] = max(0, self.count[r, c] - 1)
```
- dist=1 셀: count 감소 없음 → 학습 중 count 안정 → 셀 유지 + _bbox_contra_count 정상 누적
- dist=0 셀: 반대방향 중앙점 충분히 방문 → count→0 → 게이팅 해제 → 방향 전환 허용

**이관**: 93차(2026-04-13) 이전 항목 → `work_log_archive.md`

### 수정 파일
`src/flow_map.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- flow_map.npy 삭제 후 `python run_its_live.py` 재시작 → 재학습
- 학습 완료 로그에서 `overlap_erosion: N셀 제거` N이 이전보다 줄었는지 확인
- 상행선 하단 셀이 채워지는지, 중앙선 근처 오탐이 사라졌는지 확인

---

## 2026-04-14 (104차 — FlowMap 침식 과다 + 중앙점 방향 전환 불가 수정) [대원]

### 오늘 한 작업

**FlowMap 두 가지 잔여 문제 수정**

**문제 1 — 너무 많은 셀 침식**
- 원인: `bbox_contra_threshold=3` — bbox 풋프린트 확대 후 중앙선 인접 셀이 3번만 반대방향 방문해도 삭제
- 수정: `config.py` `bbox_contra_threshold: 3 → 8`

**문제 2 — 중앙점 셀 방향 전환 불가**
- 원인: `count >= min_samples`인 셀은 방향 게이팅이 EMA를 hard-block. 반대방향 중앙점이 아무리 지나가도 count가 줄지 않아 영구 잠금
- 수정: `flow_map.py` 방향 게이팅 차단 시 `count = max(0, count - 1)` 추가
  - min_samples(=5)번 반대방향 중앙점 방문 → count→0 → 게이팅 해제 → 다음 방문에서 방향 갱신

### 수정 파일
`src/config.py`, `src/flow_map.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- `python run_its_live.py` 재시작 → flow map 재학습 후 중앙선 셀 분리 확인
- 침식 셀 수가 적절히 줄었는지, 중앙선 근처 셀 방향이 차선별로 올바르게 전환되는지 확인

---

## 2026-04-15 (103차 — FlowMap bbox 거리 기반 alpha 감쇠 학습) [대원]

### 오늘 한 작업

**FlowMap 중앙점 우선 학습 — 거리 기반 alpha 감쇠**

**문제**: 중앙선 근처 edge 셀이 bbox로 채워진 후 방향 게이팅에 잠겨 반대 차선이 지나가도 방향을 못 바꿈

**설계:**
```
bbox 중심(dist=0) → alpha=0.10, 게이팅 적용, count 증가  (강하게 학습, 잠금 가능)
dist=1            → alpha=0.05, 게이팅 적용, count 증가
dist=2            → alpha=0.025, 게이팅 없음, count 증가 없음  (부드럽게, 잠금 불가)
dist=3+           → alpha<0.025, 게이팅 없음, count 증가 없음
```

**수정 내용:**
- `config.py`: `bbox_alpha_decay=0.5`, `bbox_gating_alpha_ratio=0.3` 추가
- `flow_map.py`: FlowMap.__init__에 두 파라미터 추가
- `flow_map.py`: learn_step — bbox 모드에서 Chebyshev 거리 계산 후:
  - `alpha_cell = alpha × decay^dist`
  - `alpha_ratio < gating_ratio` 셀: 게이팅 없이 EMA 갱신, count 미증가 (항상 soft 유지)
- `detector.py`: FlowMap 생성 시 두 파라미터 전달

**효과:**
- 중앙선 edge 셀: 낮은 alpha로 부드럽게 학습 → 반대 차선 차량 지나가면 방향 교체 가능
- 차선 중앙 셀: 강한 alpha, 게이팅으로 안정적으로 잠김

### 수정 파일
`src/config.py`, `src/flow_map.py`, `src/detector.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- `python run_its_live.py` 재시작 → 1800프레임 학습 후 flow map 시각화 확인
- 중앙선 근처 셀이 깔끔하게 분리되는지 확인
- 오탐 재발 시: `bbox_gating_alpha_ratio` 0.3→0.5로 상향 (더 강하게 잠금) 검토

---

## 2026-04-15 (102차 — global trajectory bypass 버그 수정) [대원]

### 오늘 한 작업

**fast-track & normal path global trajectory bypass 버그 수정**
- 원인: `_ft_ok = True`, `global_ok = True` 기본값 → eroded 셀이 많아 flow=None이면 전체 궤적 검증 skip → 무조건 확정
- live 오탐 W1373~1375 원인 = global_cos=None(bypassed) + flow_map.npy 재학습 직후 불안정 셀

**judge.py 수정:**
- fast-track: `_ft_ok = True` → `_ft_ok = False` (flow 없으면 확정 불가)
- normal path: `global_ok = True` → `global_ok = False` (flow 없으면 확정 불가)
- 단, 궤적 이동이 `min_move_distance` 미만이면 global_ok=True (정지 차량은 단기 투표에 위임)

**live 오탐 원인 추가 파악:**
- `[경부선] 양재/flow_map.npy` 없음 → 매 세션 re-learn → 학습 직후 셀 방향 불안정 상태에서 탐지
- 해결: run_its_live.py 재시작 → 자동 재학습 → 학습 완료 후 global_cos=-0.XX 값 확인

### 수정 파일
`src/judge.py`

### 발생 오류 / 해결
global trajectory bypass → 기본값 False로 수정, 이동 부족 시만 True

### 작업 재개 위치
- `python run_its_live.py` 재시작 → 1800프레임 학습 완료 후 탐지 재개
- 역주행 확정 시 `global_cos=-0.XX` 출력 확인 (bypass 없음)
- 학습 완료 후에도 오탐 지속 시 global_cos 양수면 flow map 방향 오류 → 재학습

---

## 2026-04-15 (101차 — velocity_window 전환 노이즈 기반 경계값 수정) [대원]

### 오늘 한 작업

**fast-track & SCR_CHECK 경계값 수정 — velocity_window 전환 노이즈 반영**

**진단 로그 분석 결과:**
```
f=302~306: d_ratio=0.00, lcf가 매 프레임 갱신 → lcf=306
f=307: d_ratio=1.00, nm=0.746 → FT_CHECK: lcf(306)<=age_end(296) = FALSE → fast-track 차단
SCR_CHECK: lcf(306) > age_end(296) = TRUE, gap=1 → 무한 리셋
```
**원인**: age gate 해제 직후 velocity_window(10프레임) 동안 방향 벡터가 전환 중
→ d_ratio=0.00 프레임에서 lcf가 age_gate_end(296)+velocity_window(10)=306까지 갱신
→ `lcf(306) <= age_end(296)` = FALSE → fast-track 차단
→ `lcf(306) > age_end(296)` = TRUE, `gap=1` → SCR_CHECK 무한 리셋

**수정 내용 (judge.py):**
- 경계값 공식: `first_seen + min_age + velocity_window`
- fast-track: `_lcf_ft <= _age_gate_end_ft + velocity_window` (= 276+20+10=306)
  - `lcf(306) <= 306` → TRUE → fast-track 허용 ✓
- SCR_CHECK: `lcf > _scr_boundary` where `_scr_boundary = first_seen + min_age + velocity_window`
  - `lcf(306) > 306` → FALSE → 리셋 없음 ✓

**wrong_count_threshold 복구 (config.py):**
- 12 → 20 (live 오탐 1300건 대응: fast-track이 명확한 역주행 처리, normal path는 보수적 유지)

**진단 prints 제거**

### 수정 파일
`src/judge.py`, `src/config.py`

### 발생 오류 / 해결
velocity_window 전환 구간 lcf 노이즈 → `+ velocity_window` 경계 추가로 해결

### 작업 재개 위치
- `python run_test.py` → `🚨 ID:30 역주행 즉시 확정 (fast-track, frame=307~)` 확인
- `python run_its_live.py` 재시작 후 역주행 오탐 건수 정상화 확인

---

## 2026-04-14 (100차 — sudden_change_rejected 무한 리셋 근본 수정) [대원]

### 오늘 한 작업

**역주행 miss 근본 원인 수정 — age gate 중 기록된 lcf 노이즈 처리**

**버그 재현:**
- ID:30 트럭이 하단에서 일자로 역주행, frame=276 등장 → frame=307 의심 시작 → 미확정
- 원인: age gate 기간(276~296) 중 방향 벡터 노이즈로 `last_correct_frame(lcf) = 290` 기록
- `sudden_change_rejected` 체크: `lcf(290) > 0` AND `fsf(307) - lcf(290) = 17 ≤ 120` → wrong_count 강제 리셋 0
- wrong_count가 12 도달할 때마다 계속 리셋 → 영구 미확정

**수정 내용 (judge.py 2곳):**

1. `sudden_change_rejected` 조건:
   - 변경 전: `lcf > 0 and (fsf - lcf) <= 120`
   - 변경 후: `lcf > _age_gate_end and (fsf - lcf) <= 120`
   - age gate 이후(first_seen + min_age) 기록된 lcf만 "정상 주행 확인"으로 인정

2. fast-track 조건:
   - 변경 전: `_lcf_ft == 0`
   - 변경 후: `_lcf_ft <= _age_gate_end_ft`
   - age gate 기간 중 lcf가 노이즈로 기록돼도 fast-track 허용

**오탐 방어 유지:**
- 진짜 정방향→역주행 급변 차량: age gate 이후에도 lcf가 기록됨 → 급변 필터 정상 동작
- fast-track: disagree_ratio≥0.95 + nm_speed≥0.20 + 전체 궤적 검증 유지

### 수정 파일
`src/judge.py`

### 발생 오류 / 해결
age gate 중 lcf 노이즈 기록 → sudden_change_rejected 무한 리셋 → age gate 해제 시점 기준으로 분기

### 작업 재개 위치
- `python run_test.py` (역주행 테스트 영상) 실행 → `🚨 ID:30 역주행 즉시 확정 (fast-track)` 또는 normal path 확정 확인

---

## 2026-04-14 (99차 — 역주행 탐지 속도 파라미터 3종 조정) [대원]

### 오늘 한 작업

**역주행 miss 방지 — 3파라미터 동시 조정**
- 원인: `⚠️ ID:30 역주행 의심 시작 (frame=321, 첫등장=276)` → age gate가 276+45=321에서 해제,
  normal path로 25 wrong_count 더 필요(→346) → 차량이 프레임을 이미 벗어나 miss

| 파라미터 | 변경 전 | 변경 후 | 효과 |
|---|---|---|---|
| `min_wrongway_track_age` | 45 | 20 | age gate 단축 (30fps 기준 0.83초 절약) |
| `fast_confirm_speed` | 0.40 | 0.20 | 중속 역주행도 fast-track 진입 가능 |
| `wrong_count_threshold` | 25 | 12 | normal path 확정까지 0.83초→0.4초 |

- ID:30 케이스 시뮬레이션: 276 등장 → **296 age gate** → fast-track ~300 즉시 확정 / normal path ~308 확정
- 오탐 방어 유지: `disagree_ratio≥0.95`(fast-track), `vote_threshold=0.7`, `long_window`, `global_traj`, `direction_change_guard=120f` 모두 유지

### 수정 파일
`src/config.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- `python run_test.py` (역주행 테스트 영상) 실행 후 miss 없이 탐지되는지 확인
- 오탐 발생 시: `min_wrongway_track_age` 25~30으로 상향, `wrong_count_threshold` 15로 재조정

---

## 2026-04-14 (98차 — FlowMap bbox 풋프린트 학습 + 겹침 기반 경계 제거) [대원]

### 오늘 한 작업

**FlowMap 학습 방식 개선**
- `flow_map.py`: `learn_step()` — bbox=None 파라미터 추가
  - 기존: 이동 경로 중간점 1셀만 EMA 갱신
  - 개선: bbox 전체 영역에 포함된 모든 셀에 EMA 갱신
  - bbox 모드에서 방향 게이팅 거부 시 `_bbox_contra_count[r,c] += 1` 누적
- `flow_map.py`: `_get_bbox_cells(bx1,by1,bx2,by2)` 헬퍼 추가
- `flow_map.py`: `apply_overlap_erosion(contra_threshold=3)` 추가
  - 반대 차선 차량 bbox가 3회 이상 밟은 셀 → 중앙선 경계로 판정 → eroded_mask 설정
  - 기존 `apply_boundary_erosion` (옆 1칸 단순 침식) 대체
- `config.py`: `bbox_contra_threshold=3` 추가

**학습 완료 후 처리 순서 변경**
```
기존: smoothing → boundary_erosion
개선: smoothing(①) → overlap_erosion(②) → smoothing(③ 중앙선 불가침 재채움)
```

**detector.py**: `learn_step` 호출에 `bbox=(x1,y1,x2,y2)` 추가 (초기학습·재학습 모두)

### 수정 파일
`src/flow_map.py`, `src/detector.py`, `src/config.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- `FORCE_RELEARN=True` 후 `python run_its_live.py` 실행
- 학습 완료 로그에서 `overlap_erosion: N셀 제거` 확인
- 오탐 시: `bbox_contra_threshold` 값 상향 (4~5)

---

## 2026-04-14 (97차 — 역주행 fast-track 즉시 확정) [대원]

### 오늘 한 작업

**역주행 고신뢰 즉시 확정 (fast-track)**
- `config.py`: `fast_confirm_ratio=0.95`, `fast_confirm_speed=0.40` 추가
- `judge.py`: long_suspect 통과 직후 fast-track 분기 추가
  - 조건: disagree_ratio≥0.95 AND nm_speed≥0.40 AND lcf==0(처음부터 역방향)
  - wrong_count 25회 누적 없이 전체 궤적 검증 → 즉시 확정
  - 갑자기 방향 바뀐 차량(lcf>0)은 fast-track 제외 → 기존 경로 유지

**탐지 시간 비교 (6fps 기준)**

| 경로 | 조건 | 탐지 시간 |
|------|------|-----------|
| 기존 | 모든 역주행 | age(7.5s) + count(4.2s) ≈ 11.7s |
| fast-track | 압도적 고속 역주행 | age(7.5s) + 1프레임 ≈ 7.5s |
| 기존 유지 | 애매한/서행 역주행 | 11.7s 유지 |

### 수정 파일
`src/config.py`, `src/judge.py`

### 발생 오류 / 해결
walrus operator 오타 즉시 수정

### 작업 재개 위치
- `python run_its_live.py` 실행 후 역주행 차량 등장 시 로그에서
  `🚨 역주행 즉시 확정 (fast-track)` 출력 확인
- fast-track 오탐 발생 시: `fast_confirm_ratio` 0.95→1.0 또는 `fast_confirm_speed` 상향

---

## 2026-04-14 (96차 — URL 선제 갱신 백그라운드 cap 교체) [대원]

### 오늘 한 작업

**화면 멈춤 없는 선제 URL 갱신 구현**
- `detector.py`: `import threading` 추가
- `detector.py`: `run()` 에 `url_refresh_interval` 파라미터 추가
  - 갱신 20초 전(`_PREFETCH_AHEAD=20.0`): 백그라운드 스레드로 새 URL 발급 + `cv2.VideoCapture` 미리 열기
  - 갱신 시점: 미리 열어둔 cap으로 교체 → **루프 중단 없음, 화면 멈춤 없음**
  - 사전 준비 실패 시 동기 fallback 유지
- `run_its_live.py`: `max_seconds` 제거, `url_refresh_interval=180` 으로 전환
  - 외부 while 루프 제거 → `detector.run()` 단일 호출로 무한 실행
  - `session_count` 미사용 변수 제거

**갱신 흐름**
```
t=0s   : 스트림 시작
t=160s : 백그라운드 스레드 → 새 URL 발급 + 새 VideoCapture 열기
t=180s : 미리 열어둔 cap으로 교체 (1프레임 이하 끊김)
t=180s~: 루프 계속 (화면 유지)
```

### 수정 파일
`src/detector.py`, `run_its_live.py`

### 발생 오류 / 해결
`session_count` unused 힌트 → 변수 제거

### 작업 재개 위치
- `python run_its_live.py` 실행 후 160초 시점에 `새 스트림 사전 준비 시작` 로그 확인
- 180초 시점에 `✅ 스트림 교체 완료 (끊김 없음)` 로그 확인

---

## 2026-04-14 (95차 — AI 모델 학습 보고서 작성) [대원]

### 오늘 한 작업

**URL 선제 갱신 (run_its_live.py)**
- `URL_REFRESH_INTERVAL`: `None` → `180` (초)
  - ITS cctvurl 만료(~200초) 20초 전에 선제 교체 → 토큰 만료로 인한 끊김 제거
- `Detector` 인스턴스를 루프 밖에서 **1회만 생성** (기존: 매 세션마다 재생성)
  - URL 갱신 시 궤적·flow_map·GRU 상태 유지
- `_fetch_url_with_retry()` 헬퍼 추가 — 지수 백오프(10s/20s/30s…) 재시도
- 단절 fallback(`_get_fresh_url`) 유지 — 50프레임 연속 실패 시 즉시 재발급

**동작 흐름**
```
시작 → Detector 1회 생성
while True:
  새 URL 발급 → detector.run(url, max_seconds=180)
  180초 경과 → 루프 종료 → 새 URL 발급 → run() 재호출 (상태 유지)
  (중간 단절 시) url_refresher 콜백 → 즉시 URL 교체 후 계속
```

### 수정 파일
`run_its_live.py`

### 발생 오류 / 해결
없음

### 작업 재개 위치
- `python run_its_live.py` 재시작 후 180초 간격으로 `[세션 N] URL 갱신` 로그 확인

---

## 2026-04-14 (95차 — AI 모델 학습 보고서 작성) [대원]

### 오늘 한 작업

**AI 모델 학습 보고서 신규 작성**
- `Docs/산출물/교통흐름모니터링_AI모델학습보고서_v1_0.md` 생성
  - YOLO11n 베이스 모델 아키텍처 개요
  - v1~v6 버전별 학습 이력 및 설정 기록 (학습일, 데이터셋, 에폭, 옵티마이저, 증강 전략)
  - results.csv 기반 성능 비교표 (Precision/Recall/mAP50/mAP50-95/F1)
  - 최종 배포 모델 v6 선정 근거 및 학습 설정 코드 첨부
  - 실사용 통합 파이프라인 구조도, 향후 개선 방향

### 수정 파일
`Docs/산출물/교통흐름모니터링_AI모델학습보고서_v1_0.md` (신규)

### 발생 오류 / 해결
없음

### 작업 재개 위치
- 보고서 내용 검토 후 추가 수정 필요 시 해당 파일 편집

---

## 2026-04-14 (94차 — 실시간 테스트 3개 버그 수정) [대원]

### 오늘 한 작업

**버그 1: GRU 오염 차단 (gru_blend_ratio 비활성)**
- `config.py`: `gru_blend_ratio` 0.20→0.0
  - 원인: 하루+ 실행으로 데이터 대부분 SMOOTH → 클래스 불균형 → GRU가 SMOOTH만 예측
  - 오염된 GRU 예측이 rule_jam에 20% 섞여 정체/원활 양방향 오탐
  - 재활성 조건: retrain 후 precision/recall 검증 완료 시 0.10~0.20 점진 적용

**버그 2: 카메라 끊김 시 jam_score 급변 방지**
- `detector.py`: TA update 조건에 `and not _is_frame_skip` 추가
  - 원인: 프레임 스킵 감지(`_is_frame_skip=True`)여도 `traffic_analyzer.update()`가 호출되어
  - 순간이동 좌표로 feature 계산 → jam_score 오염
  - 수정 후: 스킵 프레임에선 jam_score가 이전 EMA값 유지 (변화 없음)

**버그 3: 서행에서 jam_score 1.0 도달 억제**
- `config.py`: `cell_dwell_ema_up` 0.05→0.03
  - 6fps 환경 서행 2초 체류(12프레임) ema: 0.46→0.31 (cds 과누적 억제)
- `congestion_judge.py`: `1.30 * cds` → `1.10 * cds`
  - 서행 cds(0.30~0.50)에서 1.30 계수가 JAM 임계(0.60) 초과 유발

### 수정 파일
`src/config.py`, `src/congestion_judge.py`, `src/detector.py`

### 발생 오류 / 해결
없음 (파라미터·로직 수정)

### 작업 재개 위치
- `python run_its_live.py` 재시작 후:
  1. 서행 구간에서 jam_score가 0.25~0.55 범위 유지 확인
  2. 카메라 끊김 후 jam_score 안정 확인
  3. GRU blend=0으로 rule_jam만 표시되는지 확인
- GRU 재활성화: 1주일 이상 누적 후 클래스 균형 확인하고 `gru_blend_ratio=0.10`으로 시험

---

