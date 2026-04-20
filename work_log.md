# Work Log

> 오래된 항목은 `work_log_archive.md`로 이관. work_log.md는 당일 + 전날 항목만 유지.

---

## 2026-04-19 (122차 — 궤적 방향 vs 기준 방향 직접 비교 최종 안전망) [대원]

### 오늘 한 작업

**13만 프레임 5건 오탐 잔존 → 근본 원인 분석 및 수정**

**원인 분석:**

flow_map 오염 or 채널 오분류로 정상 'a' 차량이 `track_dir='b'`로 잘못 분류될 경우:
1. `flow_b` 채널 조회 → 역방향 벡터와 비교 → disagree 투표 20건 누적
2. `global_ok` 검증(line 436)도 `direction=None`(글로벌 맵) → 오염된 'b'방향 벡터 → `global_cos < -0.75` → 확정!
3. 이 경우 **차량의 실제 궤적(traj[0]→traj[-1])은 정상 방향**인데 확정됨

**최종 안전망 추가: 궤적 방향 vs ref_direction 직접 비교**

```
_traj_ref_cos = gndx * _ref_dx + gndy * _ref_dy

정상 차량: cos ≈ +1.0 → > -0.3 → 취소 ✓
진짜 역주행: cos ≈ -1.0 → < -0.3 → 확정 허용 ✓
```

어떤 flow map 오염·채널 오분류가 있어도, 차량이 실제로 역방향 궤적을 그리지 않으면 최종 확정 불가.

#### 수정 내용

**`src/judge.py`:**
- normal-path global_ok 이후: traj_ref_cos 체크 추가
- fast-track _ft_ok 이후: 동일 체크 추가
- diagnostic print에 `traj_ref_cos` 포함

### 수정 파일
`src/judge.py`

---

## 2026-04-19 (121차 — 채널 2-cell 최소 요건 + 진단 출력 추가) [대원]

### 오늘 한 작업
- `flow_map.py` 채널 조회 시 4개 인접 셀 중 **2개 이상** 채널 데이터 있어야 채널 보간 사용 (기존: 1개)
- `judge.py` normal-path 확정 시 `🚨 [normal]` 진단 출력 추가

### 수정 파일
`src/flow_map.py`, `src/judge.py`

---

## 2026-04-19 (120차 — 구역 확정 쿨다운 wrong_zone_cooldown) [대원]

### 오늘 한 작업

**15만 프레임 테스트 후 잔존 FP 3건 근본 분석 및 수정**

순차 통과 패턴(1대씩 서로 다른 시점) → 이웃 가드 미발동 → 구역 쿨다운으로 에코 차단.

#### 수정 내용

- `src/state.py`: `wrong_zone_confirmed: dict = {}` 추가
- `src/config.py`: `wrong_zone_cooldown_frames: int = 900` 추가
- `src/detector.py`: 이웃 가드 이후 구역 쿨다운 체크 삽입

### 수정 파일
`src/state.py`, `src/config.py`, `src/detector.py`
