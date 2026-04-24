# Work Log Archive

> work_log.md에서 이관된 과거 작업 기록. 수정하지 않음.

---

## 2026-04-22 (123차 — GRU 전면 제거 + HistoricalPredictor 도입 + 재연결 잼스코어 안정화) [대원]

### 오늘 한 작업

#### ① GRU 모듈 전면 제거

**배경:** GRU는 90프레임(15초) 입력으로 5분 후를 예측 — 구조적 한계로 예측 품질 불신.
API 끊김으로 인한 jam_score 변동에도 GRU가 노이즈를 증폭하는 부작용 존재.

- `src/gru_module.py` 삭제
- `tests/test_gru_module.py` 삭제
- `src/traffic_analyzer.py`: `gru_module` 파라미터 및 블렌딩 로직 제거, `final_jam = rule_jam` 직결
- `src/detector.py`: GRU import, 초기화, 피처 수집, pretrain/online_step, 세션 flush 전부 제거
- `src/config.py`: GRU 파라미터 섹션 전부 제거 (gru_hidden, gru_layers, gru_seq_len, gru_blend_ratio 등)
- `tests/test_traffic_analyzer.py`: GRU Mock 클래스 및 TestAnchoring(TA-09/TA-10) 제거

#### ② HistoricalPredictor 신규 도입

**설계:** 시각별(hour × 5분 슬롯) jam_score 이력을 CSV에 누적 → 5분 후 정체 수준 예측.

- 288슬롯/일 (`slot_id = hour * 12 + minute // 5`)
- 매 5분 창마다 jam_score 중앙값을 flush (API 순간 블립 내성)
- `predict()` → 해당 슬롯 데이터 없으면 `None` → 패널 "Training..." 표시
- CSV 저장 경로: `flow_map_path.parent/hist_jam_a.csv`, `hist_jam_b.csv`
- 신뢰도: `min_conf_samples=14` (약 70분 누적 시 100%)
- 종료 시 `flush_current()` 호출 → 마지막 미완성 창 저장

#### ③ 재연결 후 잼스코어 안정화 (2단계)

**문제:** API 끊김 → 재연결 시 jam_score가 순간 급락 (두 가지 원인).

**1단계 — post_skip_grace_frames (TA 업데이트 차단):**
- 재연결 직후 IDManager 속도 이력 미구성 → nm=0 차량이 stop_count 증가
- `config.py`: `post_skip_grace_frames = 30` 추가 (6fps 기준 5초)
- `detector.py`: displacement skip / timestamp gap / freeze reconnect 3개 지점에서 `_last_skip_frame` 갱신
- `_in_grace` 조건 충족 시 `traffic_analyzer.update()` 차단

**2단계 — 체류 시간 소급 부여 (dwell retroactive credit):**
- grace 기간 후 새 tid가 등록될 때 `dwell_cell_ratio`가 0으로 리셋되는 문제
- `feature_extractor.py`: 새 tid 첫 등장 시 해당 셀의 `cell_dwell_ema > 0.3`이면
  `first_frame = frame_num - dwell_thr_frames`으로 소급 설정 → 즉시 체류 셀로 인정
- `cell_dwell_ema`는 ID 무관 셀 기반 → 재연결 후에도 이전 혼잡도 보존됨

### 수정 파일
`src/gru_module.py` (삭제), `tests/test_gru_module.py` (삭제),
`src/traffic_analyzer.py`, `src/detector.py`, `src/config.py`,
`src/feature_extractor.py`, `src/historical_predictor.py` (신규),
`tests/test_traffic_analyzer.py`

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

---

## 2026-04-18 (119차 — v3 npy 채널 자동 재구성 + 이웃 가드 임계값 완화 + cascade reset) [대원]

### 오늘 한 작업

**118차 가드가 W1-W4 오탐 시 발동되지 않는 문제 수정**

**원인 분석:**
1. `neighbor_guard_min_total=3` 인데 W1 확정 시 이웃(W2, W3)이 2대뿐 → 가드 미발동
2. 기존 flow_map.npy가 v3 형식이면 `_ref_dx=None` → 채널 미구축 → 117/118차 효과 없음

---

#### ① v3 npy 로드 후 채널 자동 재구성 (`detector.py`)

```python
# detect_only 로드 직후
self._compute_ref_direction()
if self._ref_direction is not None:
    self.flow.build_directional_channels(*self._ref_direction)
```

v4 미만 파일을 로드해도 즉시 양방향 채널을 재구성 → npy 삭제 불필요.

---

#### ② 이웃 가드 임계값 완화 (`config.py`)

```
neighbor_guard_min_total: 3 → 2   # W2+W3 2대만 있어도 발동
neighbor_guard_agree:     2 → 1   # 1대만 동방향이어도 취소
```

---

#### ③ cascade reset — 연쇄 확정 방지 (`detector.py`)

W1 가드 취소 시, 의심 누적 중인 이웃(W2·W3)도 `wrong_way_count=0` 리셋.
→ W1 취소 직후 W2·W3이 연속 확정되는 패턴 차단.

```python
for _ov2 in (same-dir neighbors with wrong_way_count > 0 going same direction):
    st.wrong_way_count[_ov2] = 0
    st.first_suspect_frame.pop(_ov2, None)
```

### 수정 파일
`src/detector.py`, `src/config.py`

### 주의사항
- 기존 npy 삭제 불필요 — 로드 직후 자동 채널 재구성
- cascade reset은 이미 `wrong_way_ids`에 있는 확정 차량은 건드리지 않음

---

## 2026-04-18 (118차 — 오염-인식 글로벌 fallback + 이웃 차량 방향 가드) [대원]

### 오늘 한 작업

**39f/19f 균일 오탐 패턴 근본 수정**

**문제 패턴 분석:**
```
W1 │  39f (1.30s) │  19f (0.63s)   ← 의심 시작 = age gate 해제 순간(20f)
W3 │  39f (1.30s) │  19f (0.63s)   ← wrong_count가 20프레임 연속 누적 → 확정
```
normal path 확정: age gate 해제 직후부터 **단 한 프레임도 agree 없이** 20f 연속 disagree.
= flow map이 해당 차량의 진행 경로에서 반대 방향을 가리키고 있음.

---

#### ① 오염-인식 글로벌 fallback (`flow_map.py`)

**기존 문제:**
양방향 채널 구축 후, 채널에 데이터 없는 셀은 글로벌 맵으로 fallback.
글로벌 맵이 반대 차선 오염 벡터면 그대로 사용 → 오탐 유발.

**수정:**
채널 데이터 없는 셀에서 글로벌 방향이 쿼리 방향과 반대이면 `None` 반환.
```
A차량 → flow_a 조회 → 없음 → 글로벌=B방향(오염) → None → vote skip
A차량 → flow_a 조회 → 없음 → 글로벌=A방향(clean) → 반환 → vote 참여
```
`_ref_dx, _ref_dy` 를 FlowMap에 저장, `build_directional_channels` 시 세팅.

---

#### ② 이웃 차량 방향 일치 가드 (`detector.py` + `config.py`)

**기존 문제:**
flow map 오류 시 해당 구역의 모든 정상 차량이 일괄 flagging됨.
flow map에만 의존하므로 "현재 프레임에서 무슨 일이 벌어지는지" 모름.

**수정:**
역주행 확정 직전, 같은 방향분류(A/B) 이웃 차량 중 같은 방향으로 이동 중인 차량 수 확인.
```
진짜 역주행: 이 차량만 반대방향 → 같은방향 이웃 0~1대 → 가드 비발동
flow map 오류: 여러 정상차량 일괄 flagging → 이웃도 같은방향 → 취소
```

```python
for _ov, _ovv in st.last_velocity.items():
    if track_direction[_ov] != _sus_dir: continue  # 같은 분류만
    if cos(ndx,ndy, _ovv) > 0.5: _same_dir += 1

if _total_nbr >= 3 and _same_dir >= 2:
    wrong_way_ids.discard(tid)  # 역주행 취소
```

**config 추가:**
- `neighbor_guard_min_total: int = 3`
- `neighbor_guard_agree: int = 2`

### 수정 파일
`src/flow_map.py`, `src/detector.py`, `src/config.py`

---

## 2026-04-18 (117차 — 중앙값 속도 벡터 + 양방향 flow map) [대원]

### 오늘 한 작업

**수치 조정 없는 근본 해결 2종 동시 적용**

---

#### ① 중앙값(Median) 속도 벡터 (`detector.py`, `judge.py`)

**기존 방식 문제:**
```
velocity = traj[-1] - traj[-velocity_window]  # endpoint-to-endpoint
```
velocity_window(10f) 안에 신호 끊김·서버 지연이 1프레임이라도 있으면
시작점 또는 끝점이 오염 → 전체 방향 벡터가 뒤집힘.

**수정:**
```python
_pfx = [traj[si+i+1][0] - traj[si+i][0] for i in range(w-1)]
_pfy = [traj[si+i+1][1] - traj[si+i][1] for i in range(w-1)]
vdx = median(_pfx) * (w-1)
vdy = median(_pfy) * (w-1)
```

10프레임 중 1~2프레임이 끊겨도 중앙값에는 영향 없음.

#### ② 양방향(Dual-Channel) flow map (`flow_map.py`, `detector.py`, `judge.py`)

학습 완료 후 A/B 채널 분리. A차량은 flow_a만, B차량은 flow_b만 조회 → 중앙선 오염 구조적 차단.

### 수정 파일
`src/flow_map.py`, `src/detector.py`, `src/judge.py`

---

## 2026-04-15 (116차 — 서행→가속 fast-track 오탐 방지) [대원]

서행 구간(nm<0.15) 중 lcf 미갱신 → 가속 직후 fast-track 즉시 확정 오탐 수정.
이중 방어: ① 서행 중에도 방향 일치 시 lcf 갱신 / ② post_slow_guard_frames=30 추가.

### 수정 파일
`src/config.py`, `src/judge.py`

---

## 2026-04-15 (115차 — fast-track 최소 나이 요건 추가) [대원]

fast-track 오탐 방지용 `fast_confirm_min_age=45` 추가. 새 씬·오염 셀에서 정상 차량 28f만에 확정되는 패턴 차단.

### 수정 파일
`src/config.py`, `src/judge.py`

---

## 2026-04-15 (114차 — bbox 수평 폭 제한 + dist≥2 게이팅 추가) [대원]

bbox 학습 폭을 `bbox_height × 0.8`로 제한해 중앙선 침범 방지. dist≥2 셀 게이팅 추가.

### 수정 파일
`src/detector.py`, `src/flow_map.py`, `src/config.py`

---

## 2026-04-15 (113차 — fleet cos 재감지 버그 수정 + 중앙선 침범 방지 강화) [대원]

fleet cos 재감지 안 되는 버그(prev=-0.02 고정) 수정. dist=1 게이팅 임계값 강화(-0.4→-0.2).

### 수정 파일
`src/detector.py`, `src/flow_map.py`

---

## 2026-04-15 (112차 — fleet cosine 기반 끊김 감지 추가) [대원]

전차량 fleet cosine 급락 감지 3번째 끊김 감지 계층 추가. 발동 조건: 유효 차량 ≥5, avg<-0.1 또는 0.5 급락.

### 수정 파일
`src/detector.py`

---

## 2026-04-15 (111차 — freeze 감지 임계값 상대값으로 교체) [대원]

freeze 감지 고정 임계값 → 상대값(avg × 10%) 교체. 정체 구간 adj≈0.4 오감지 방지.
- `_adj_diff_history` 90프레임 롤링 평균, 임계값 `max(avg×0.10, 0.05)`
- `config.py`: `freeze_diff_threshold` 제거, `min_freeze_frames` 3→10
- `judge.py`: normal-path `_reconnect_guard` 제거 (direction_change_frame + min_age로 충분)

수정 파일: `src/detector.py`, `src/judge.py`, `src/config.py`

---

## 2026-04-15 (110차 — 프레임 freeze 기반 끊김 재연결 감지) [대원]

프레임 내용(adj_diff) 기반 끊김 재연결 감지 추가.
- `state.py`: `post_reconnect_frame = 0` 추가
- `config.py`: `freeze_diff_threshold=0.5`, `min_freeze_frames=3` 추가
- `detector.py`: freeze 카운터, 재연결 이벤트 시 전차량 궤적 초기화 + direction_change_frame
- `judge.py`: fast-track 조건에 `not _reconnect_guard` 추가

수정 파일: `src/state.py`, `src/config.py`, `src/detector.py`, `src/judge.py`

---

## 2026-04-15 (109차 — fast-track 궤적 일관성 체크 + FlowMap 가장자리 마진) [대원]

- `judge.py`: fast-track에서 traj[0]→[-1] 전체방향 vs traj[-window]→[-1] 최근방향 cos < 0.5 → 차단
- `config.py`: `flow_map_edge_margin: int = 1` 추가
- `flow_map.py` + `detector.py`: 그리드 외곽 1줄 학습 제외

수정 파일: `src/judge.py`, `src/flow_map.py`, `src/config.py`, `src/detector.py`

---

## 2026-04-15 (108차 — dist==0 trajectory 기반 게이팅 복원) [대원]

dist==0 게이팅을 velocity_window 방향 → trajectory 방향으로 교체. `continue` 추가(EMA 갱신 거부 핵심). `count -= 1 → count -= 2` (2배 빠른 잠금 해제).

수정 파일: `src/flow_map.py`

---

## 2026-04-15 (107차 — FlowMap apply_direction_repair 추가) [대원]

학습 완료 후 셀 방향 교정: 이웃 일관성 충분 + 해당 셀만 반대(cos<-0.3) → 이웃 평균으로 교정.
처리 순서: ①smoothing → ②overlap_erosion → ③direction_repair → ④smoothing

수정 파일: `src/flow_map.py`, `src/detector.py`

---

## 2026-04-15 (106차 — FlowMap 중앙점 즉시 덮어씌움 + 궤적 방향 학습) [대원]

learn_step 전면 재설계. dist=0 게이팅 제거(항상 갱신) + traj 방향 우선. `_bbox_contra_count`를 dist=0에서만 추적.
⚠️ 106차 자체가 역효과(오염 심화) → 108차에서 게이팅 복원.

수정 파일: `src/flow_map.py`, `src/detector.py`

---

## 2026-04-15 (105차 — FlowMap count 감소 범위 수정) [대원]

count 감소를 dist==0에만 제한 (dist=1 count 감소 제거 → 상행선 하단 셀 공백 해소).

수정 파일: `src/flow_map.py`

---

## 2026-04-14 (104차 — FlowMap 침식 과다 + 중앙점 방향 전환 불가 수정) [대원]

`bbox_contra_threshold` 3→8. dist=0 게이팅 차단 시 `count = max(0, count-1)` 추가 (영구 잠금 해제).

수정 파일: `src/config.py`, `src/flow_map.py`

---

## 2026-04-15 (103차 — FlowMap bbox 거리 기반 alpha 감쇠 학습) [대원]

`bbox_alpha_decay=0.5`, `bbox_gating_alpha_ratio=0.3` 도입. dist당 alpha 감쇠, dist≥2는 soft 유지.

수정 파일: `src/config.py`, `src/flow_map.py`, `src/detector.py`

---

## 2026-04-15 (102차 — global trajectory bypass 버그 수정) [대원]

`_ft_ok = True` → `False`, `global_ok = True` → `False` (flow 없으면 확정 불가). 이동 부족 시만 True.

수정 파일: `src/judge.py`

---

## 2026-04-15 (101차 — velocity_window 전환 노이즈 기반 경계값 수정) [대원]

fast-track `_lcf_ft <= _age_gate_end_ft` 경계: `first_seen + min_age + velocity_window` 공식 적용. `wrong_count_threshold` 12→20.

수정 파일: `src/judge.py`, `src/config.py`

---

## 2026-04-14 (100차 — sudden_change_rejected 무한 리셋 근본 수정) [대원]

SCR 체크 경계: `lcf > 0` → `lcf > _age_gate_end`. fast-track: `lcf == 0` → `lcf <= _age_gate_end`.

수정 파일: `src/judge.py`

---

## 2026-04-14 (99차 — 역주행 탐지 속도 파라미터 3종 조정) [대원]

`min_wrongway_track_age` 45→20, `fast_confirm_speed` 0.40→0.20, `wrong_count_threshold` 25→12.

수정 파일: `src/config.py`

---

## 2026-04-14 (98차 — FlowMap bbox 풋프린트 학습 + 겹침 기반 경계 제거) [대원]

`learn_step()` bbox 모드 추가 (bbox 전체 셀 EMA). `apply_overlap_erosion()` 추가 (contra_count≥3 → eroded). `apply_boundary_erosion` 대체.

수정 파일: `src/flow_map.py`, `src/detector.py`, `src/config.py`

---

## 2026-04-06 (73~76차 — normal mode 제거 + jam_score 밀도 보정 + 방향 분류 nearest-neighbor) [대원]

### 오늘 한 작업 [대원] — 추가

**B방향(상행) jam_score 저평가 원인 분석 및 수정**
- 원인: flow_map 상단 rows 0~6 (y < 252px) 전체 미학습 → `get_interpolated` = None → 기본값 'a' → 상행 차량 전부 A방향으로 오분류
- `flow_map.py`: `get_nearest_direction(x, y)` 추가 — count>0 셀 중 그리드 거리 최소 셀 벡터 반환 (20×20 브루트포스)
- `detector.py._classify_direction`: `flow_v=None` 시 `get_nearest_direction` 호출 → 최종 None이면 'a' fallback

**jam_score 공식 sqrt(bbox_coverage) 적용**
- 최종: `0.80×slow + 0.70×stop + 0.35×sqrt(bbox_coverage)`

### 수정 파일 [대원]
`src/congestion_judge.py`, `src/flow_map.py`, `src/detector.py`

---

## 2026-04-06 (73~74차 — normal mode 제거 + jam_score 밀도 보정) [대원]

**congestion_judge.py normal mode 전면 제거** (LCS 기반 → 항상 fallback 호출)
**bbox_coverage 가중치**: 0.25 → 0.35, count_ratio 항 추가

### 수정 파일 [대원]
`src/congestion_judge.py`

---

## 2026-04-06 (68~72차 — 역주행 오탐 근본 재설계 + bbox_coverage + nm 테스트) [대원]

**역주행 오탐 가드** — edge detection 방식, direction_change_cos_threshold=0.0
**bbox_coverage** — density_score 대체
**fallback jam_score**: `0.60×slow + 0.60×stop + 0.25×bbox_coverage`
**nm 테스트**: `tests/test_nm_measurement.py`, `test_nm_live.py`

### 수정 파일 [대원]
`src/config.py`, `src/state.py`, `src/judge.py`, `src/id_manager.py`, `src/detector.py`, `src/feature_extractor.py`, `src/congestion_judge.py`

---

## 2026-04-03 (67차 — 웹 탄소/정체 탭 설계 논의 + 벤치마킹)

### 오늘 한 작업 [수빈]

**웹 화면 구조 파악**
- `C:\finalPj_웹` 기존 팀원 웹 프로젝트 구조 분석 완료
  - React(Vite) + Flask + Socket.IO, 다크 테마, 인라인 스타일
  - `carbon` 탭: Sidebar·라우팅 이미 등록됨, `carbon/index.jsx`는 플레이스홀더
  - `backend_flask/modules/carbon/carbon.py`: `/health` 엔드포인트만 존재
- 데이터 소스 결정: **웹 백엔드가 `src/` 탐지 모듈 직접 import해서 실행**

**탄소/정체 탭 설계 논의**
- 탄소배출 계산: 신규 ML 개발 없이 배출계수 공식으로 추정 가능 확인
  - 정체 수준별 배출계수 (EMEP/EEA 기준): 정상 2.3 / 서행 4.1 / 정체 6.8 g/s/대
- 팀원 탄소 중복 여부 미확정 → 탄소 포함/제외 두 안 도출, 결정 보류

**벤치마킹 조사 완료**
- TOPIS, View-T, ITS 국가교통정보센터 (국내) + TomTom, INRIX, JamVis (해외) 참조
- 공통 구성: 정체 레벨 상태 카드 + 시계열 차트 + KPI 카드 + 이벤트 로그

### 수정 파일 [수빈]
- 없음 (설계·조사 단계)

### 발생 오류 / 확인 사항
- 없음

### 작업 재개 위치 [수빈]
- 탄소 포함/제외 결정 후 `carbon/index.jsx` UI 구현 시작
- 안 A (정체 전용): 상태 카드 + jam_score 차트 + KPI 3개 + 이벤트 로그
- 안 B (정체 + 간이 탄소): 안 A + 탄소 추정 사이드 카드 추가
- 백엔드 `carbon.py`에 정체 데이터 API 라우트 추가 필요

---

## 2026-04-02 (66차 — 역주행 오탐 파라미터 조정)

### 오늘 한 작업 [대원]

**역주행 오탐 감소를 위한 파라미터 조정 (`src/config.py`)**
- `wrong_count_threshold`: 8 → 12 (연속 의심 횟수 기준 강화 — 단발성 오탐 차단)
- `direction_change_guard_frames`: 90 → 120 (정상 판정 후 4초간 의심 카운트 차단 — 커브/차선변경 오탐 차단)

**파라미터 조정 범위 외 확인된 추가 오탐 원인** (코드 수정 필요 시 별도 진행)
- `judge.py` 하드코딩: `smoothed_mask` 역방향 판정 임계값 `-0.50` → `-0.65` (코사인 130°)
- 단기 투표 커브 오탐: 과거 궤적에 현재 방향벡터 적용하는 구조적 문제

### 수정 파일 [대원]
`src/config.py`

### 발생 오류 / 확인 사항
- jam_score 개선(slow_ratio 도입) 후 테스트 계속 중 (서행 0.3~0.4 → 0.5+ 목표)
- 역주행 오탐 파라미터 조정 후 검증 필요

### 작업 재개 위치 [대원]
- 역주행 오탐 추가 검증 (파라미터 조정 후)
- 필요 시 `judge.py` smoothed_mask 임계값 코드 수정 (사용자 요청 시)

---

## 2026-04-02 (65차 — fallback jam_score 근본 수정: slow_ratio 도입)

### 오늘 한 작업 [대원]

**jam_score 로직 전체 분석 및 근본 문제 발견**
- `norm_speed_ratio` fallback 적용 불가 이유 확인:
  - fallback baseline의 `norm_speed_ref=0.15` (고정값)
  - 실제 고속도로 nm = 0.5~2.0 → ratio가 항상 1.0으로 clip → speed_contribution = 0
  - `upper_half` 중앙값 방식: 혼합 방향 차량 존재 시 빠른 차량이 서행 신호 덮어버림
- `stop_ratio`만으로는 nm 0.06~0.15 구간 서행 차량 감지 불가

**slow_ratio 신규 feature 도입 (`src/feature_extractor.py`)**
- nm 구간별 분류 추가:
  - nm < 0.06 → `stopped_count` (정지, 기존)
  - 0.06 ≤ nm < 0.15 → `slow_count` (서행, 신규)
  - nm ≥ 0.15 → 정상 주행 (카운트 없음)
- `slow_ratio = slow_count / speed_known_count` (소표본 보정 동일 적용)
- feature 딕셔너리에 `"slow_ratio"` 추가 (8차원으로 확장)

**fallback 공식 최종 교체 (`src/congestion_judge.py`)**
- 기존: `0.40×density + 0.60×stop` (서행 감지 불가)
- 최종: `0.50×slow + 0.30×stop + 0.20×density`
- 설계 목표: 원활 jam=0.06, 서행 jam=0.40, 극심 jam=0.48

### 수정 파일 [대원]
`src/config.py`, `src/flow_map.py`, `src/judge.py`, `src/detector.py`,
`src/feature_extractor.py`, `src/traffic_analyzer.py`, `src/congestion_judge.py`

### 발생 오류 / 확인 사항
- norm_speed_ratio 기반 접근 4차례 시도 → 근본 원인(norm_speed_ref 고정값) 확인 후 포기
- slow_ratio 방식으로 전환: nm 기준값 의존 없이 직접 구간 카운트

### 작업 재개 위치 [대원]
- 서행/정체 영상에서 slow_ratio 도입 후 jam_score 확인 (목표: 서행 ≥ 0.30)

---

## 2026-04-02 (64차 — 문서 체계 정비 + 화면설계서 v1.1 재생성)

### 오늘 한 작업 [수빈]

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

### 수정 파일 [수빈]
`CLAUDE.md`, `FILE_INDEX.md`, `Docs/dev_guide.md`, `Docs/dev_guide_phase2.md`, `Docs/plan.md`
`N드라이브: generate_screen_design_docx.py`, `교통흐름모니터링_화면설계서.docx`

### 발생 오류
- python-docx 미설치 → cv 환경에 설치 완료

### 작업 재개 위치 [수빈]
- 웹 화면 구성(React 뼈대) 작업 시작 예정
- 산출물 인터페이스명세서·프로그램설계서 수동 수정 대기 중

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

**파라미터 튜닝**
- `norm_stop_threshold`: 0.10 → 0.06, `density_max_vehicles`: 20 → 40
- `direction_change_guard_frames`: 45 → 90, `wrong_count_threshold`: 5 → 8
- `vote_threshold`: 0.60 → 0.70, `slow_jam_threshold`: → 0.60

### 수정 파일
`src/flow_map.py`, `src/detector.py`, `src/judge.py`, `src/visualizer.py`, `src/feature_extractor.py`, `src/config.py`

---

## 2026-04-01 (62차 — 정체 탐지 baseline 설계 확정 + 코드 수정)

- **설계 확정**: flow_map(역주행 전용) / 정체 판정(fallback + LCS=0.36) 분리
- **수정 파일**: `src/config.py` (default_lcs=0.36), `src/detector.py` (3곳)

---

## 2026-04-01 (61차 — LCS 수정 완료 확인 + 설계 방향 논의)

- 베이스라인 재학습 결과: `passage=8404, lcs=0.36` 확인 (0.96→0.58→0.36)
- LCS 0.36 = 품질 점수, 실제 판정 기준은 `norm_speed_ref` 임을 정리

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

---

## 2026-03-28 (19차 — detector.py PassageTracker 연동 + 속도 계산 수정)

### 오늘 한 작업
1. **`src/detector.py` 전면 수정** (PassageTracker 연동 + 속도 계산 정상화)
   - `PassageTracker` import 추가
   - `__init__`: `self.passage_tracker = PassageTracker(cfg, self.state)` 생성
   - `flow.load()` 튜플 반환값 언패킹: `loaded, saved_baseline = self.flow.load(...)` 수정
   - `run()`: `TrafficAnalyzer` 생성 시 `passage_tracker` 연결
   - `run()`: `traffic_analyzer.set_state(self.state)` 호출 추가
   - `run()`: 저장된 baseline 있으면 즉시 `set_baseline()` 호출
   - per-track: `fx = (x1+x2)/2, fy = y2` footpoint 계산 추가
   - per-track: 첫 등장 시 `passage_tracker.on_entry()` 호출
   - 루프 후: 사라진 ID 감지 → `passage_tracker.on_exit()` 호출
   - 루프 후: `passage_tracker.record_frame_stats()` 매 프레임 호출
   - 학습/재학습 완료 시: `finalize_baseline()` → `set_baseline()` 연동
   - 카메라 전환 시: `passage_tracker.reset()` 추가
2. **`tests/conftest.py` 신규 생성** — `test_traffic_analyzer_legacy.py` pytest 수집 제외
3. **pytest 30/30 PASS 유지 확인**

### 수정/생성 파일
- `src/detector.py` (전면 수정), `tests/conftest.py` (신규)

---

## 2026-03-28 (20차 — 버그 2건 수정: relearning 초기값 + camera_switch NoneType 오류)

### 오늘 한 작업
1. **`src/state.py`** — `relearning = True` → `False` 수정
2. **`src/camera_switch.py`** — `set_reference()`에 `prev_small` 초기화 추가
3. **`src/detector.py`** — 초기 학습 진행률 화면 표시 추가

### 수정 파일
- `src/state.py`, `src/camera_switch.py`, `src/detector.py`

---

## 2026-03-28 (21차 — flow_map 학습 임계값 완화 + Phase 2 방향 결정)

### 오늘 한 작업
1. **`src/config.py`** — `min_move_distance`: 20.0→8.0, `min_move_per_frame`: 1.5→0.5

### 수정 파일
- `src/config.py`

---

## 2026-03-28 (22차 — 하행 중앙 차선 버스 오탐 수정)

### 오늘 한 작업
1. **`src/config.py`** — `cos_threshold`: -0.5→-0.65, `wrong_count_threshold`: 4→6, `vote_threshold`: 0.6→0.65, `grid_size`: 15→20

### 수정 파일
- `src/config.py`

---

## 2026-03-28 (23차 — 중앙 분리대 경계 보간 오탐 근본 수정)

### 오늘 한 작업
1. **`src/flow_map.py`** — `get_interpolated()` 방향 충돌 감지 로직 추가

### 수정 파일
- `src/flow_map.py`

---

## 2026-03-28 (24차 — grid_size 복원 + 충돌 임계값 수정으로 과탐 회귀 수정)

### 오늘 한 작업
1. **`src/config.py`** — `grid_size`: 20→15 복원, `learning_frames`: 500→700
2. **`src/flow_map.py`** — 충돌 감지 임계값 -0.3→-0.5

### 수정 파일
- `src/config.py`, `src/flow_map.py`

---

## 2026-03-28 (25차 — id_manager.py 구조적 오탐 버그 2건 수정)

### 오늘 한 작업
1. **`src/id_manager.py`** — `check_reappear()` flow_v=None 시 재매칭 허용 버그 수정
2. **`src/id_manager.py`** — `cleanup()` wrong_way_ids 미정리 버그 수정

### 수정 파일
- `src/id_manager.py`

---

## 2026-03-28 (26차 — flow_map 충돌 감지 수정 + id_manager 연쇄 오탐 추가 수정)

### 오늘 한 작업
1. **`src/flow_map.py`** — `get_interpolated()` 충돌 감지 임계값 -0.5→-0.4 + 폴백 None 반환
2. **`src/id_manager.py`** — 재등장 연쇄 오탐 추가 수정

### 수정 파일
- `src/flow_map.py`, `src/id_manager.py`

---

## 2026-03-28 (27차 — 중앙선 blank 구역 역주행 미탐지 수정)

### 오늘 한 작업
1. **`src/judge.py`** — `total_checked < 3` → `< 2` 수정

### 수정 파일
- `src/judge.py`

---

## 2026-03-28 (28차 — 오탐 폭증 원인 파악 및 flow_map 재학습 준비)

### 오늘 한 작업
1. 27차 롤백: `total_checked < 2` → `< 3` 복원
2. flow_map 오염 확인 → 정상 영상으로 재학습 준비
3. `run_wrongway.py` VIDEO_FILE 변경, `임시/flow_map.npy` 삭제

### 수정 파일
- `src/judge.py`, `run_wrongway.py`

---

## 2026-03-28 (29차 — 원본 방식 복원: 이중선형보간 + 공간평활화 + 임계값)

### 오늘 한 작업
1. **`src/flow_map.py`** — `get_interpolated()` 순수 이중선형보간 복원, `apply_spatial_smoothing()` 원본+보호 방식
2. **`src/config.py`** — `cos_threshold`: -0.7→-0.5, `vote_threshold`: 0.65→0.6 (원본 복원)

### 수정 파일
- `src/flow_map.py`, `src/config.py`

---

## 2026-03-28 (30차 — learn_step 방향 게이팅으로 경계 셀 오염 근본 차단)

### 오늘 한 작업
1. **`src/flow_map.py`** — `learn_step()` 방향 게이팅 추가 (cos<-0.4이면 확립 셀 갱신 거부)

### 수정 파일
- `src/flow_map.py`

---

## 2026-03-28 (31차 — 경계 셀 erosion + CCTV 글자 오탐 필터 추가)

### 오늘 한 작업
1. **`src/flow_map.py`** — `apply_boundary_erosion()` 메서드 추가
2. **`src/detector.py`** — `apply_boundary_erosion()` 호출 추가
3. **`src/judge.py`** — 궤적 점프 필터 추가
4. **`src/config.py`** — `max_jump_ratio=4.0` 추가

### 수정 파일
- `src/flow_map.py`, `src/detector.py`, `src/judge.py`, `src/config.py`

---

## 2026-03-28 (32차 — eroded_mask로 재학습 차단 + 방향 급변 필터 교체)

### 오늘 한 작업
1. **`src/flow_map.py`** — `eroded_mask` 추가
2. **`src/state.py`** — `last_velocity` 딕셔너리 추가
3. **`src/judge.py`** — 방향 급변 필터로 교체
4. **`src/config.py`** — `max_jump_ratio` 제거

### 수정 파일
- `src/flow_map.py`, `src/state.py`, `src/judge.py`, `src/config.py`

---

## 2026-03-28 (33차 — 정상→역방향 급전환 가드 필터 추가)

### 오늘 한 작업
1. **`src/state.py`** — `last_correct_frame` 딕셔너리 추가
2. **`src/judge.py`** — `direction_change_guard_frames` 가드 추가
3. **`src/config.py`** — `direction_change_guard_frames=8` 추가

### 수정 파일
- `src/state.py`, `src/judge.py`, `src/config.py`

---

## 2026-03-27 (18차 — 폴더 분리 + run_wrongway.py import 오류 수정)

### 오늘 한 작업
1. **폴더 분리**
   - `mini_project_0313-feature-data_team/` → `mini_project/` 복사 완료 (원본은 Notebook 열려있어 수동 삭제 필요)
   - `src/run_wrongway.py` → 프로젝트 루트 `run_wrongway.py`로 이동
2. **import 오류 수정** (`ModuleNotFoundError: No module named 'wrongway'`)
   - 원인: `from wrongway import ...` (존재하지 않는 패키지)
   - 수정: `from src import Detector, DetectorConfig` (`run_test.py`와 동일 방식)
3. **Phase 1 flat import 호환 수정** (`run_wrongway.py`, `run_test.py`)
   - `sys.path.insert(0, str(PROJECT_ROOT / "src"))` 추가
   - 이유: `traffic_analyzer.py`의 `from feature_extractor import ...` (flat)이 패키지 import 시 실패하는 문제 해결
4. `CLAUDE.md` 폴더 역할 표 업데이트 (`mini_project/`, `run_wrongway.py`, `run_test.py` 추가)
5. `FILE_INDEX.md` 실행 스크립트 항목 추가
6. **pytest 30/30 PASS 유지 확인**

### 수정/생성 파일
- `run_wrongway.py` (루트 신규, 기존 src/run_wrongway.py 대체)
- `run_test.py` (sys.path 수정)
- `mini_project/` (폴더 생성, 파일 19개 복사 완료)
- `CLAUDE.md`, `FILE_INDEX.md`

### 미완료 (수동 작업 필요)
- `mini_project_0313-feature-data_team/` 폴더 삭제 — Notebook 열린 채로 삭제 불가
  → Jupyter Notebook 모두 닫은 후 폴더 수동 삭제

---

## 2026-03-27 (17차 — Phase 1 TDD Step 11~12 완료, pytest 30/30 PASS)

### 오늘 한 작업
1. **Step 11** `src/traffic_analyzer.py` 내부 전면 교체
   - 기존: `_mag_to_kmh()` + `free_flow_speed` 비율 판정 (절대 km/h)
   - 신규: `FeatureExtractor` + `CongestionJudge` 조합 (baseline 대비 비율)
   - 신규 public 메서드: `set_state()`, `set_baseline()`, `get_jam_score()`
   - `CongestionPredictor` 클래스는 기존과 동일하게 유지
   - 빈 트랙(empty tracks) 처리 — early return으로 SMOOTH 유지 (TA-02 대응)
2. **Step 12** `tests/test_traffic_analyzer.py` 재작성 (TA-01~10 + CP-01~02, 12개 케이스)
   - 기존 TC-01~13 → `tests/test_traffic_analyzer_legacy.py`로 이동(비활성화)
   - 신규 테스트: `_MockState`, `_MockPassageTracker`, `_make_baseline()`, `_make_tracks()` 헬퍼 작성
3. **전체 pytest** 30/30 PASS (test_passage_tracker·test_congestion_judge·test_traffic_analyzer)

### 수정/생성 파일
- `src/traffic_analyzer.py` (내부 전면 교체, 인터페이스 유지)
- `tests/test_traffic_analyzer.py` (TA-01~10 + CP-01~02 신규)
- `tests/test_traffic_analyzer_legacy.py` (기존 TC-01~13 보존용)

---

## 2026-03-27 (16차 — dev_guide.md 분리 + CLAUDE.md 팀 공유 지침 추가)

### 오늘 한 작업
1. `CLAUDE.md` — "산출물 폴더 먼저 검토" 지침 추가 (팀 공유, 메모리 대체)
2. `Docs/dev_guide.md` — Phase 2 섹션(GRU·기술스택·알림등급·WebSocket) 분리
3. `Docs/dev_guide_phase2.md` — 신규 생성 (Phase 2 이후 참조용)
4. MD 파일 전체 간결화 진행 중

### 수정 파일
- `CLAUDE.md`, `Docs/dev_guide.md`, `Docs/dev_guide_phase2.md`, `FILE_INDEX.md`, `work_log.md`

---

## 2026-03-26 (15차 — 정체 탐지 알고리즘 설계 완료 및 MD 전면 업데이트)

### 오늘 한 작업
1. 정체 탐지 핵심 설계 확정 (pixels_per_meter 완전 제거 — ITS API CCTV 방향/각도 미제공 확인)
   - normalized_mag(mag/bbox_h) + dwell_time 기반 상대 지표 설계
   - jam_score 가중치: 0.50×속도 + 0.30×dwell + 0.20×밀도 / feature 벡터 7차원 확정
   - footpoint(y2) 사용, baseline freeze(SMOOTH 구간), LCS(학습 품질) 0~1
   - Phase 1(규칙 기반) → Phase 2(GRU 40%) → Phase 3(GRU 65%) 로드맵 확정
2. `Docs/dev_guide.md` 전면 재작성 (19개 섹션)
3. `Docs/plan.md` §3.0 모듈 목록 업데이트 / `FILE_INDEX.md` 신규 파일(4개) 추가

### 수정 파일
- `Docs/dev_guide.md` (전면 재작성), `Docs/plan.md`, `FILE_INDEX.md`

---

## 2026-03-26 (14차 — 개발 준비 완료 및 Opus 프롬프트 작성)

### 오늘 한 작업
1. `CLAUDE.md`: 프로젝트 개요/폴더역할/개발단계 업데이트, dev_guide.md 먼저 읽기 지침 추가
2. `FILE_INDEX.md`: 개발 파일 역할 표 최신화
3. `src/` 폴더 생성 + 미니프로젝트 py 전체 복사 (13개 파일)
4. `tests/` 폴더 생성
5. 기존 파일 분석 (tracks: [{id,x1,y1,x2,y2,cx,cy}], speeds: {tid:mag})
6. Opus 프롬프트 작성 (Phase 1 TDD용)

### 수정 파일
- `CLAUDE.md`, `FILE_INDEX.md`, `Docs/dev_guide.md`, `Docs/plan.md`, `src/` (신규), `tests/` (신규)

---

## 2026-03-26 (13차 — 초기화 완료 및 설계 검토)

### 오늘 한 작업
1. 미니프로젝트 원본 복구 확인 (event_publisher.py 삭제, traffic_analyzer.py 없음 확인)
2. `Docs/plan.md`: §2 정체 탐지 항목 제거, §3.0 미개발 명시
3. `수빈_노트/유스케이스_v1.0.md`: UC-06~09 상태 → 🔲 미개발
4. 메모리 정리, `개발/` 폴더 삭제

### 수정 파일
- `Docs/plan.md`, `수빈_노트/교통흐름모니터링_유스케이스_v1.0.md`

---

## 2026-03-26 (12차 — 개발 방향 재정립 및 정리)

### 오늘 한 작업
1. 팀원 Flask+React 웹에 교통흐름모니터링 탭 이미 구현 확인 → 우리 역할: 탐지코드 개발 후 깃허브 공유
2. `개발/상세설계서.md`, `개발/테스트계획서.md` 삭제
3. `Docs/plan.md` 간소화 (완성된 것 목록 + 연동 방향 + 수정 원칙만 유지)

### 수정 파일
- `개발/상세설계서.md` (삭제), `개발/테스트계획서.md` (삭제), `Docs/plan.md`

---

## 2026-03-26 (11차 — 산출물 보강 + Phase 1 TDD + 산출물 전면 일관성 검토)

### 오늘 한 작업
1. 요구사항정의서: FR-017~020 추가(관제센터 웹), API 명세 4건 추가(WS 2건+REST 2건)
2. pytest 환경 수정 (Python 3.13 호환) — pytest 70/70 PASS
3. event_publisher.py: CongestionEvent + EventPublisher 구현 (TC-31~37)
4. plan.md 기술스택: Flask+Flask-SocketIO+MySQL+React+Docker로 수정, Phase 4 추가
5. 기획서 슬라이드 6·9·10: density.py → traffic_analyzer.py 수정
6. 산출물 전체 일관성 최종 검토 완료

### 수정 파일
- `산출물/요구사항정의서_v2.0.xlsx`, `산출물/기획서_v1.0.pptx`, `Docs/plan.md`
- `wrongway/__init__.py`, `wrongway/pytest.ini`, `wrongway/event_publisher.py` (신규)

---

## 2026-03-25 (10차 — 프로젝트 방향 재정립 + research.md/plan.md 재설계)

### 오늘 한 작업
1. 산출물 불일치 파악 및 수정 (요구사항정의서 시트8 추적매트릭스, 시트5 신규모듈명, 유스케이스 UC-06~09)
2. 프로젝트 방향 재정립 (역주행 로직 불변 원칙 확립, 새 목표: 정체 탐지 → 관제센터 웹 알림 → 조치 권고)
3. 정체 해결 전략 논문 조사 (VMS, VSL, ALINEA, 사고관리, 우회경로, 서울 FTMS)
4. `research.md` §11 신규 추가 (정체 해결 전략 논문 7종 + 조치 매핑 표)
5. `plan.md` 전면 재설계 (버그수정 계획서 → 관제센터 웹 개발 계획서)

### 수정 파일
- `산출물/교통흐름모니터링_요구사항정의서_v2.0.xlsx`, `산출물/update_artifacts.py` (신규)
- `수빈_노트/교통흐름모니터링_유스케이스_v1.0.md`, `Docs/research.md`, `Docs/plan.md`

---

## 2026-03-25 (9차 — Phase 2 리팩토링: VirtualCountingLine → 궤적 기반 방향 분류)

### 오늘 한 작업
1. 라인 선 방식 → cos_sim 기반 상행/하행 분류 전면 리팩토링
2. traffic_analyzer.py: VirtualCountingLine 제거, cos_sim 방향 판정, get_direction_counts() 추가
3. visualizer.py: draw_counting_lines() 제거 → draw_direction_count() 추가
4. config.py: counting_lines 필드 제거 / run_wrongway.py: COUNTING_LINES 관련 제거
5. tests/test_phase2.py 완전 재작성 (TC-21~26) — pytest 63/63 PASS

### 수정 파일
- `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/run_wrongway.py`, `src/detector.py`, `tests/test_phase2.py`

---

## 2026-03-25 (8차 — Phase 3: 단기 정체 예측)

### 오늘 한 작업
1. `테스트계획서.md` TC-27~30 추가
2. `tests/test_phase3.py` 신규 작성 (6개)
3. `traffic_analyzer.py` CongestionPredictor 클래스 추가 (polyfit 선형 회귀 기반)
4. `visualizer.py`: draw_prediction_status() 추가 (p 키 토글)
5. `config.py`: prediction_history_window, prediction_horizon 추가 / `detector.py` 연동
   — pytest 63/63 PASS

### 수정 파일
- `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/detector.py`, `tests/test_phase3.py`, `개발/테스트계획서.md`

---

## 2026-03-25 (7차 — 카운팅 라인 사용자 설정 인터페이스)

### 오늘 한 작업
1. `config.py`: counting_lines 필드 추가
2. `visualizer.py`: show_counting_lines, l 키 토글, draw_counting_lines() 추가
3. `detector.py`: cfg.counting_lines → add_counting_line() 자동 등록
4. `run_wrongway.py`: COUNTING_LINES 변수 추가 — pytest 57/57 PASS

### 수정 파일
- `src/config.py`, `src/visualizer.py`, `src/detector.py`, `src/run_wrongway.py`

---

## 2026-03-25 (6차 — Phase 2 기능 개발)

### 오늘 한 작업
1. `상세설계서.md` §11 추가 (Phase 2 설계)
2. `테스트계획서.md` TC-21~26 추가 / `tests/test_phase2.py` 신규 작성 (6개)
3. `traffic_analyzer.py` Phase 2 확장: VirtualCountingLine, EMA, 트렌드 판정
4. `visualizer.py`: draw_congestion_heatmap() 추가 (h 키 토글)
5. `config.py`: speed_ema_alpha, speed_decline_threshold 추가 / `detector.py` 연동
   — pytest 57/57 PASS

### 수정 파일
- `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/detector.py`, `tests/test_phase2.py`

---

## 2026-03-25 (5차 — BUG-01, BUG-07 수정 + UC-06 전체 개발)

### 오늘 한 작업
1. BUG-01: detector.py cleanup 매 프레임 호출로 변경
2. BUG-07: detector.py mode 동적 결정 (LEARNING/RELEARNING/DETECTING)
3. traffic_analyzer.py 신규 구현 + 연동 (config.py, logger.py, detector.py)
4. visualizer.py: draw_congestion_status() 추가 (c 키 토글)
5. 나머지 버그 전체 수정 (BUG-02~04, 06, 08~09, ISSUE-01~05)
6. tests/test_bug_fixes.py (TC-15~20) + tests/test_integration.py (TC-11~12) 신규
   — pytest 63/63 PASS

### 수정 파일
- `src/detector.py`, `src/traffic_analyzer.py`, `src/visualizer.py`, `src/config.py`, `src/logger.py`, `tests/test_bug_fixes.py`, `tests/test_integration.py`

---

## 2026-03-25 (4차 — 화면설계서 검수 후 피드백 반영)

### 오늘 한 작업
1. 기획서 제목 수정 / SCR-04 헤딩 밑줄 제거 / SCR-05 예시 데이터 수정
2. SCR-09 정탐률 → 처리율 수정 / SCR-07 모바일 알림 탭 추가

### 수정 파일
- `산출물/generate_screen_design_docx.py`, `산출물/교통흐름모니터링_화면설계서.docx` (재생성)

---

---

## 2026-04-06 (68~72차 — 역주행 오탐 근본 재설계 + bbox_coverage + nm 테스트) [대원]

### 오늘 한 작업 [대원]

**역주행 오탐 가드 — edge detection 방식으로 전면 재설계**
- 기존: last_correct_frame 기준 경과 시간 체크 → 신규 등장 차량 무력화 문제
- 신규: 방향 급변 순간(edge) 감지 후 guard_frames 동안 보호
- `config.py`: `direction_change_cos_threshold = 0.0` 추가

**BBoxStabilizer 제거 + footpoint 중앙(cy) + 3프레임 warmup**

**bbox_coverage — density_score 대체**
- `Σbbox면적 / (flow_map 유효 셀 수 × 셀 면적)` — 차선 수 독립

**fallback jam_score 공식 교체**
- 신규: `0.60×slow + 0.60×stop + 0.25×bbox_coverage`

**nm 테스트 코드 작성**
- `tests/test_nm_measurement.py`, `test_nm_live.py`

### 수정 파일 [대원]
`src/config.py`, `src/state.py`, `src/judge.py`, `src/id_manager.py`,
`src/detector.py`, `src/feature_extractor.py`, `src/congestion_judge.py`,
`tests/test_nm_measurement.py` (신규), `test_nm_live.py` (신규)

---

## 2026-04-06 (73~74차 — normal mode 제거 + jam_score 밀도 보정) [대원]

### 오늘 한 작업 [대원]

**congestion_judge.py normal mode 전면 제거**
- `compute_jam_score()` 함수 삭제 (LCS 기반 정상 모드)
- `compute_jam()`: if/else 분기 → 항상 `compute_jam_score_fallback()` 호출

**jam_score smooth 구간 바닥 점수 개선**
- bbox_coverage 가중치: 0.25 → 0.35
- count_ratio 항 추가: `+0.10 × min(count_ratio, 1.0)`

### 수정 파일 [대원]
`src/congestion_judge.py`

---

## 2026-04-06 (73~76차 — normal mode 제거 + jam_score 밀도 보정 + 방향 분류 nearest-neighbor) [대원]

### 오늘 한 작업 [대원]

**B방향(상행) jam_score 저평가 원인 분석 및 수정**
- 원인: flow_map 상단 rows 0~6 (y < 252px) 전체 미학습 → `get_interpolated` = None → 기본값 'a' → 상행 차량 전부 A방향으로 오분류
- `flow_map.py`: `get_nearest_direction(x, y)` 추가 — nearest-neighbor fallback
- `detector.py._classify_direction`: `flow_v=None` 시 `get_nearest_direction` 호출

**jam_score 공식 sqrt(bbox_coverage) 적용**
- 최종: `0.80×slow + 0.70×stop + 0.35×sqrt(bbox_coverage)`

### 수정 파일 [대원]
`src/congestion_judge.py`, `src/flow_map.py`, `src/detector.py`


---

## 2026-04-07 (81~90차 — 대규모 리팩토링: 데드코드 제거 + 역주행 오탐 개선) [대원]

**데드코드 완전 제거**: `baseline_stats.py`, `passage_tracker.py`, `bbox_stabilizer.py` 삭제
**역주행 오탐 개선**: `min_wrongway_track_age=30`, footpoint EMA smoothing(alpha=0.4)
**jam_score 재보정**: `0.80×slow + 0.70×stop + 0.45×sqrt(bbox)`, `slow_jam_threshold=0.60`
수정: `src/config.py`, `congestion_judge.py`, `detector.py`, `feature_extractor.py`, `traffic_analyzer.py`, `flow_map.py`, `state.py`, `id_manager.py`, `judge.py`, `run_test.py`

---

## 2026-04-07 (77~80차 — 상행선 서행 탐지 공정성 개선) [대원]

**bbox_coverage 원근 편향 제거**: cell occupancy 방식으로 교체, 방향별 유효 셀 수 분리
**velocity 기반 방향 분류 조기 적용** (3프레임~)
**jam_score 가중치 재보정**: `0.29×slow + 0.25×stop + 0.20×sqrt(bbox) + 0.08×count`
**slow_upper_nm**: 0.50→2.5, `nm_cy_correction_k`: 0.6→0.0
수정: `src/config.py`, `congestion_judge.py`, `detector.py`, `feature_extractor.py`, `traffic_analyzer.py`

## 2026-04-08 (74~79차 — 웹 버그픽스·동기화·자동시작) [수빈]

### 74차 — cctv_state 키 불일치 버그픽스
- `reverse_detector.py`: `display_name` 추가 (`_reverse` 접미사 제거)
- `its.py`: `{name}_reverse` 키 우선 조회 후 `{name}` fallback

### 75차 — jam_score 과도 상승 + 경부선 fallback 좌표
- `reverse_detector.py`: src/congestion_judge.py 직접 import, `reset()` 교체
- `its.py`: 경부선 GYEONGBU_FALLBACK 좌표 전면 교체

### 76차 — bbox_coverage·count_ratio 분모 버그
- `reverse_detector.py`: bbox_coverage 분모 전체그리드→방향별 유효셀, count_ratio 분모 n_known→n_total

### 77차 — jam_score false positive (서버재시작 후)
- `reverse_detector.py`: `set_baseline()`→`reset()` (EMA 0.5시작→0시작)
- `its.py`: 경부선 fallback 좌표 재수정

### 78차 — 대원 81~90차 pull 반영
- `reverse_modules/config.py`: velocity_window·wrong_count_threshold·smooth_jam_threshold·slow_upper_nm 동기화
- `reverse_detector.py`: `_make_x_t()` 소표본 보정 동기화

### 79차 — 모델 고정·conf 버그·학습중 jam 차단·전체CCTV 자동시작
- `reverse_detector.py`: GPU/CPU 분기 제거, conf falsy 버그 수정, 학습중 jam 차단
- `its.py`: `_fetch_gyeongbu_cctvs(all_cameras=True)` 추가
- `app.py`: `_auto_start_all_detectors()` 백그라운드 스레드 (부팅 후 전체 CCTV 자동 시작)

### 수정 파일 (74~79차)
`C:\finalPj_웹/backend_flask/` —
`app.py`, `models.py`, `modules/traffic/its.py`,
`modules/traffic/detectors/reverse_detector.py`,
`modules/traffic/detectors/reverse_modules/config.py`

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
