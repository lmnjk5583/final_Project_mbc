# 개발 기준서 (dev_guide.md)

> 최종 수정: 2026-03-27
> **개발 관련 작업 시 이 문서를 가장 먼저 읽는다. Phase 1 핵심 내용만 포함.**
> 새 세션 시작 시: 이 문서 → work_log.md 순서로 읽고 현재 상태 파악 후 진행.
> Phase 2 이후 내용(GRU·기술스택·PTZ Phase 2·알림등급): `dev_guide_phase2.md` 참조.

---

## 0. 새 세션 인수인계 체크리스트

새 세션에서 이 프로젝트를 이어받을 때 반드시 확인할 항목.

### 현재 파일 상태 (2026-03-26 기준)

| 파일 | 현재 상태 | 다음 할 일 |
|------|-----------|-----------|
| `src/traffic_analyzer.py` | ❌ OLD 코드 — pixels_per_meter 기반, km/h 변환, TC-01~13 기준 | Phase 1 설계로 내부 전면 교체 |
| `src/config.py` | ⚠️ OLD 파라미터만 있음 (§A 신규 파라미터 미추가) | §A 파라미터 추가 |
| `src/state.py` | ⚠️ `entry_positions` 필드 없음 | 필드 1개 추가 |
| `src/id_manager.py` | ⚠️ `cleanup()`에 passage 기록 없음 | 약 15줄 추가 |
| `src/flow_map.py` | ⚠️ `speed_ref` 배열·`learn_baseline()` 없음 | 필드·메서드 추가 |
| `src/detector.py` | ⚠️ footpoint·baseline freeze·feature 추출 없음 | 최소 수정 |
| `src/feature_extractor.py` | ❌ 없음 | 신규 작성 |
| `src/baseline_stats.py` | ❌ 없음 | 신규 작성 |
| `src/passage_tracker.py` | ❌ 없음 | 신규 작성 |
| `src/congestion_judge.py` | ❌ 없음 | 신규 작성 |
| `tests/test_traffic_analyzer.py` | ⚠️ OLD 기준 TC-01~13 (pixels_per_meter) — 현재 PASS 상태 | 신규 설계로 재작성 |
| `tests/test_passage_tracker.py` | ❌ 없음 | TDD로 먼저 작성 |
| `tests/test_congestion_judge.py` | ❌ 없음 | TDD로 먼저 작성 |

### 개발 진행 순서

```
Step 1. src/baseline_stats.py 신규 작성 (데이터 클래스, 의존성 없음)
Step 2. tests/test_passage_tracker.py 작성 (TDD)
Step 3. src/passage_tracker.py 구현 → pytest PASS
Step 4. tests/test_congestion_judge.py 작성 (TDD)
Step 5. src/feature_extractor.py 신규 작성
Step 6. src/congestion_judge.py 구현 → pytest PASS
Step 7. src/config.py 파라미터 추가 (§A)
Step 8. src/state.py entry_positions 필드 추가
Step 9. src/flow_map.py speed_ref·learn_baseline() 추가
Step 10. src/id_manager.py cleanup()에 passage 기록 추가
Step 11. src/traffic_analyzer.py 내부 전면 교체 (인터페이스 유지)
Step 12. src/detector.py footpoint·freeze·feature 추출 추가
Step 13. 실영상 테스트 (run_test.py)
```

---

## 1. 우리가 할 것 / 하지 않을 것

| 구분 | 내용 |
|------|------|
| ✅ 할 것 | `traffic_analyzer.py` — 정체 탐지 (jam_score·정체레벨·예측) |
| ✅ 할 것 | 정체 이벤트 DB 저장 |
| ✅ 할 것 | 정체 해결 조치 권고 (research.md §11 기반) |
| ✅ 할 것 | YOLO11n 추가학습 (고속도로 CCTV 데이터) |
| ❌ 안 할 것 | 웹 UI / Flask 라우트 (팀원 담당) |
| ❌ 안 할 것 | 탄소배출 기능 |
| ❌ 안 할 것 | 역주행 탐지 로직 수정 (§15 원칙 적용) |
| ❌ 안 할 것 | 절대 km/h 측정 (카메라 캘리브레이션 없이 불가 — ITS API 확인 완료) |

---

## A. config.py 신규 파라미터 목록 (DetectorConfig에 추가)

기존 DetectorConfig 클래스 끝에 아래 필드를 추가한다.

```python
# ==================== Phase 1 정체 탐지 파라미터 ====================
min_active_for_baseline: int   = 2      # baseline 갱신에 필요한 최소 활성 차량 수
min_passage_dist:        float = 100.0  # 유효 passage 최소 진입-퇴장 픽셀 거리
min_passages_required:   int   = 5      # 학습 종료에 필요한 최소 완성 passage 수
stop_mag_threshold:      float = 3.0    # 이 값 이하 mag이면 정지로 판단 (픽셀)
exit_rate_window:        int   = 30     # exit_rate 계산 슬라이딩 윈도우 (프레임)
grace_period_sec:        float = 60.0   # 카메라 전환 후 판정 유예 시간 (초)

# ==================== jam_score 임계값 ====================
smooth_jam_threshold:    float = 0.25   # jam_score < 이 값 → SMOOTH
slow_jam_threshold:      float = 0.55   # jam_score < 이 값 → SLOW, 이상 → CONGESTED

# ==================== 학습 연장 ====================
max_learning_extension:  float = 1.5   # learning_frames × 이 값 = 최대 학습 프레임 수
```

---

## 2. 개발 파일 위치

```
최종 프로젝트/
  src/
    baseline_stats.py     ← 신규: PassageRecord·BaselineStats 데이터 클래스
    passage_tracker.py    ← 신규: 차량 진입/퇴장 기록·dwell 집계·LCS 산출
    feature_extractor.py  ← 신규: 7차원 feature 벡터 계산
    congestion_judge.py   ← 신규: jam_score 계산·레벨 판정·히스테리시스
    traffic_analyzer.py   ← 기존 인터페이스 유지, 내부 전면 교체
    gru_module.py         ← 신규 (Phase 2만)
    config.py             ← §A 파라미터 추가
    state.py              ← entry_positions 필드 추가
    id_manager.py         ← cleanup()에 passage 기록 추가
    flow_map.py           ← speed_ref 배열, learn_baseline() 추가
    detector.py           ← footpoint·freeze·feature 추출 최소 수정
    visualizer.py         ← 정체 시각화 추가
    logger.py             ← jam_score 로그 컬럼 추가
  tests/
    test_passage_tracker.py   ← TDD 먼저 작성
    test_congestion_judge.py  ← TDD 먼저 작성
    test_traffic_analyzer.py  ← OLD 코드 기준 TC-01~13 → 신규 재작성
```

---

## 3. 핵심 설계 원칙

> ITS API 확인: CCTV 방향·각도 미제공 → 절대 km/h 불가. **모든 지표를 학습 기준 대비 비율(ratio)로 설계.**

### 3.1 normalized_mag (원근 보정 속도)

```python
# detector.py ~218번째 줄, mag 계산 후
bbox_h = max(y2 - y1, 1)
normalized_mag = mag / bbox_h  # 원근 왜곡 제거. 줌 변화 시 자동 보정.
```

### 3.2 footpoint

```python
# detector.py ~192번째 줄, stabilize() 직후
fx = (x1 + x2) / 2  # footpoint x
fy = y2              # footpoint y (지면 접촉점). 이후 셀 배정·passage·feature 모두 fy 사용.
```

### 3.3 baseline freeze (온라인 학습 전용)

초기 학습 중에는 무조건 학습. 온라인 학습(`enable_online_flow_update=True`) 구간에서만 freeze.

```python
# detector.py ~242번째 줄
if (cfg.enable_online_flow_update and not is_wrong and st.wrong_way_count[tid] == 0):
    if (congestion_judge.get_level() == "SMOOTH"
            and len(active_ids) >= cfg.min_active_for_baseline):
        self.flow.learn_step(...)
        self.passage_tracker.update_baseline(fx, fy, normalized_mag)
```

---

## 4. 학습 시스템 설계

### 4.1 PassageRecord 데이터 클래스

```python
# src/baseline_stats.py
from dataclasses import dataclass

@dataclass
class PassageRecord:
    track_id:         int    # ByteTrack ID
    entry_frame:      int    # first_seen_frame[tid] 활용
    entry_fx:         float  # 진입 footpoint x
    entry_fy:         float  # 진입 footpoint y (= y2)
    exit_frame:       int    # _stale_counter==1 시점 frame_num
    exit_fx:          float  # 퇴장 footpoint x (trajectories[-1][0])
    exit_fy:          float  # 퇴장 footpoint y (= 마지막 y2)
    dwell_frames:     int    # exit_frame - entry_frame
    entry_exit_dist:  float  # sqrt((ex-fx)²+(ey-fy)²)
    is_complete:      bool   # True=자연퇴장, False=stale_threshold 강제 종료

    @property
    def is_valid(self) -> bool:
        """분석에 사용할 수 있는 유효한 passage인지 판단"""
        return (self.is_complete
                and self.dwell_frames > 0
                and self.entry_exit_dist >= MIN_PASSAGE_DIST)  # config에서 읽음
```

### 4.2 BaselineStats 데이터 클래스

```python
# src/baseline_stats.py (PassageRecord 아래에 추가)
@dataclass
class BaselineStats:
    free_flow_dwell:  float  # percentile(dwells,10) n≥10, min(dwells) n<10
    typical_dwell:    float  # median(dwells)
    norm_speed_ref:   float  # median(normalized_mags) 학습 구간 전체
    count_ref:        float  # mean(frame_counts) 학습 구간 전체
    bbox_slope:       float  # polyfit(cy_vals, bbox_h_vals, 1)[0]
    bbox_intercept:   float  # polyfit(cy_vals, bbox_h_vals, 1)[1]
    lcs:              float  # Learning Congestion Score 0.0~1.0
    quality_warning:  bool   # typical_dwell > free_flow_dwell × 3
    passage_count:    int    # 기준 산출에 사용한 passage 수
    is_fallback:      bool   # True=passage 부족→fallback 모드
```

**flow_map.npy 저장 포맷 버전 2:**

```python
save_dict = {
    "version": 2,                 # 버전 1=기존, 버전 2=baseline_stats 포함
    "flow":   self.flow,          # ndarray (grid_size, grid_size, 2)
    "count":  self.count,         # ndarray (grid_size, grid_size)
    "baseline_stats": {           # BaselineStats를 dict로 직렬화
        "free_flow_dwell": float,
        "typical_dwell":   float,
        "norm_speed_ref":  float,
        "count_ref":       float,
        "bbox_slope":      float,
        "bbox_intercept":  float,
        "lcs":             float,
        "quality_warning": bool,
        "passage_count":   int,
        "is_fallback":     bool,
    }
    # Phase 2에서 "gru_weights" 키 추가 예정 — 기존 키 변경 없음
}
```

로드 시 "version" 키가 없거나 1이면 baseline_stats 없이 구버전으로 처리.

### 4.3 LCS (Learning Congestion Score)

학습 구간이 얼마나 정체 상태였는지를 0~1로 점수화.

```python
def compute_lcs(dwells, norm_speeds, entry_counts, exit_counts) -> float:
    if len(dwells) < 2:
        return 0.5  # 데이터 부족 → 중간값 반환

    min_dwell = max(min(dwells), 1)
    signal_A = min((np.median(dwells) / min_dwell - 1) / 5, 1.0)  # dwell 분포 편중

    max_speed = max(max(norm_speeds), 1e-6)
    signal_B = 1.0 - np.mean(norm_speeds) / max_speed              # 속도 저하

    total_entry = sum(entry_counts) + 1e-6
    signal_C = max(0.0, 1.0 - sum(exit_counts) / total_entry)      # 차량 누적

    return 0.40 * signal_A + 0.35 * signal_B + 0.25 * signal_C
```

| LCS 범위 | 판단 | 시스템 동작 |
|----------|------|------------|
| 0.0~0.3 | 정상 학습 | 보정 없이 baseline 사용 |
| 0.3~0.6 | 부분 오염 | 경고 출력, threshold 소폭 완화 |
| 0.6~0.8 | 높은 오염 | 경고 출력, threshold 대폭 완화 |
| 0.8~1.0 | 심각 오염 | is_fallback=True → fallback 모드 |

### 4.4 학습 연장 조건

```python
# detector.py ~164번째 줄 (학습 완료 처리 블록)
if st.is_learning and st.frame_num >= cfg.learning_frames:
    max_ext   = int(cfg.learning_frames * cfg.max_learning_extension)  # 기본 750
    completed = self.passage_tracker.get_completed_count()
    if completed < cfg.min_passages_required and st.frame_num < max_ext:
        pass  # 연장: passage 부족
    else:
        self.flow.apply_spatial_smoothing()
        baseline = self.passage_tracker.finalize_baseline()
        self.congestion_judge.set_baseline(baseline)
        if cfg.flow_map_path:
            self.flow.save(cfg.flow_map_path, baseline_stats=baseline)
        st.is_learning = False
        print(f"학습 완료! passage={completed}, LCS={baseline.lcs:.2f}")
        if baseline.quality_warning:
            print("⚠️ 학습 품질 경고: 정체 중 학습 가능성 있음.")
```

---

## 5. Feature 벡터 명세 (7차원)

Phase 1 jam_score 입력 / Phase 2 GRU 입력 공통.
**모든 값은 baseline 대비 비율 → 화각 변경 후에도 의미 동일.**

| 인덱스 | 이름 | 계산식 | 범위 | 비고 |
|--------|------|--------|------|------|
| 0 | `norm_speed_ratio` | `mean(norm_mags_this_frame) / max(norm_speed_ref, 0.01)` | 0~1 | clip(0,1) |
| 1 | `count_ratio` | `active_count / max(count_ref, 1)` | 0~∞ | clip(0,3) |
| 2 | `stop_ratio` | `stopped_count / max(active_count, 1)` | 0~1 | mag < stop_mag_threshold |
| 3 | `exit_rate_ratio` | `exit_last_30f / max(baseline_exit_rate, 0.01)` | 0~∞ | clip(0,3) |
| 4 | `dwell_ratio` | `free_flow_dwell / max(mean(current_dwells), 1)` | 0~1 | clip(0,1) |
| 5 | `density_score` | `occupied_cells / total_cells` | 0~1 | 기존 밀도맵 활용 |
| 6 | `rule_jam_score` | 아래 §6.1 계산 결과 | 0~1 | Phase 2 GRU 보조 입력 |

**current_dwells 정의**: 현재 프레임에 활성 중인 차량들의 `current_frame - first_seen_frame[tid]` 값들.
(완성된 passage가 아닌, 현재 화면에 있는 차량들의 누적 체류 시간)

---

## 6. jam_score 설계 (Phase 1)

### 6.1 정상 운영 시 (baseline 있음)

```python
def compute_jam_score(x_t: dict, lcs: float, cfg) -> float:
    speed_score   = clip(1 - x_t["norm_speed_ratio"], 0, 1)   # 가중 0.50
    dwell_score   = clip(1 - x_t["dwell_ratio"], 0, 1)        # 가중 0.30
    density_score = clip((x_t["count_ratio"] - 1) / 2, 0, 1) # 가중 0.20
    jam = 0.50 * speed_score + 0.30 * dwell_score + 0.20 * density_score

    bonus = 0.0
    if x_t["exit_rate_ratio"] > 1.3:  bonus += 0.08
    if x_t["stop_ratio"] < 0.05:      bonus += 0.05
    if x_t["norm_speed_ratio"] > 1.2: bonus += 0.07
    return max(0.0, jam - bonus)
```

### 6.2 fallback 모드 (baseline 없거나 LCS≥0.8)

```python
def compute_jam_score_fallback(x_t: dict) -> float:
    """baseline 없이 밀도·정지비율·유출만으로 판단 (정확도 낮음)"""
    density_contribution = clip(x_t["density_score"] * 2, 0, 1)
    stop_contribution    = x_t["stop_ratio"]
    # exit_rate_ratio: 낮을수록 차량이 빠져나가지 못함
    outflow_contribution = clip(1 - x_t["exit_rate_ratio"] / 2, 0, 1) if x_t["exit_rate_ratio"] < 2 else 0

    return 0.40 * density_contribution + 0.35 * stop_contribution + 0.25 * outflow_contribution
```

### 6.3 학습 구간 중 (is_learning=True)

```python
# 학습 중에는 jam_score 계산 안 함 → 레벨 = "학습 중"
if st.is_learning or st.relearning:
    return  # 정체 판정 스킵
```

화면 표시: 좌하단에 "학습 중... (XX/500)" 표시.

### 6.4 레벨 판정 (히스테리시스 포함)

```python
def classify(jam_score, lcs, cfg) -> str:
    smooth_thr = cfg.smooth_jam_threshold * (1 - lcs * 0.40)  # 최저 0.15
    slow_thr   = cfg.slow_jam_threshold   * (1 - lcs * 0.30)  # 최저 0.38

    if jam_score < smooth_thr:  return "SMOOTH"
    if jam_score < slow_thr:    return "SLOW"
    return "CONGESTED"
```

히스테리시스: 기존 `_apply_hysteresis()` 로직 그대로 재사용 (15초 유지).

---

## 7. 모듈별 인터페이스 명세

### 7.1 PassageTracker

```python
class PassageTracker:
    def __init__(self, cfg, state):
        # cfg: DetectorConfig (min_passage_dist, stop_mag_threshold 등)
        # state: DetectorState (first_seen_frame, entry_positions 참조)

    def on_entry(self, track_id: int, fx: float, fy: float, frame_num: int):
        """새 ID 첫 등장 시 호출 — state.entry_positions에 기록"""

    def on_exit(self, track_id: int, fx: float, fy: float,
                frame_num: int, is_complete: bool = True):
        """차량 사라질 때 호출 — PassageRecord 생성 후 리스트에 추가"""

    def record_frame_stats(self, active_count: int, exit_count: int,
                           norm_mags: list, cy_vals: list, bbox_h_vals: list):
        """매 프레임 호출 — frame_counts·exit_counts·norm_mags 누적"""

    def get_completed_count(self) -> int:
        """유효 완성 passage 수 반환"""

    def finalize_baseline(self) -> BaselineStats:
        """학습 완료 시 호출 — BaselineStats 산출 후 반환"""

    def compute_lcs(self) -> float:
        """현재까지 수집 데이터로 LCS 계산"""

    def update_baseline(self, fx: float, fy: float, norm_mag: float):
        """온라인 학습 구간에서 baseline 점진적 갱신 (SMOOTH 시만 호출)"""

    def get_current_dwells(self, active_ids: set, frame_num: int) -> list:
        """현재 활성 차량들의 체류 프레임 수 리스트 반환"""
```

**PassageTracker와 IDManager 연결:**

```python
# id_manager.py IDManager.__init__에 passage_tracker 파라미터 추가
class IDManager:
    def __init__(self, cfg, flow_map, state, passage_tracker=None):
        ...
        self.passage_tracker = passage_tracker  # None이면 기존 동작

    def cleanup(self, active_ids):
        ...
        for tid in list(st.trajectories.keys()):
            if tid not in active_ids:
                st._stale_counter[tid] += 1

                # ← 기존: 역주행 차량만 last_pos 기록
                # ← 추가: 모든 차량 exit 기록
                if st._stale_counter[tid] == 1 and self.passage_tracker:
                    traj = st.trajectories.get(tid, [])
                    if traj:
                        last_fx = traj[-1][0]
                        # last y2 직접 보관이 안 되므로 cy 사용 (근사)
                        last_fy = traj[-1][1]
                        self.passage_tracker.on_exit(
                            tid, last_fx, last_fy, st.frame_num,
                            is_complete=(st._stale_counter[tid] < cfg.stale_threshold)
                        )
```

**주의**: `last_fy`는 원래 cy(중심)이지만, trajectories가 footpoint(fy=y2)로 변경되면 자동 반영됨.

### 7.2 FeatureExtractor

```python
class FeatureExtractor:
    def __init__(self, cfg, state, passage_tracker, baseline_stats=None):
        ...
        self.baseline = baseline_stats  # 학습 완료 후 set_baseline()으로 갱신

    def set_baseline(self, baseline_stats: BaselineStats):
        self.baseline = baseline_stats

    def compute(self, tracks: list, speeds: dict,
                flow_map, frame_num: int) -> dict:
        """매 프레임 7차원 feature 벡터 계산. baseline 없으면 None 반환."""
        if self.baseline is None:
            return None
        # tracks별 norm_mag, dwell, stopped_count, 셀 점유율 계산 후 §5 표 계산식 적용
        # → {norm_speed_ratio, count_ratio, stop_ratio, exit_rate_ratio, dwell_ratio, density_score, rule_jam_score=0.0}
```

### 7.3 CongestionJudge

```python
class CongestionJudge:
    def __init__(self, cfg, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.baseline: BaselineStats = None
        self._current_level = "SMOOTH"
        self._pending_level = "SMOOTH"
        self._level_hold_frames = 0
        self._hysteresis_frames = int(cfg.congestion_hysteresis_sec * fps)
        self._congestion_start_frame = None
        self._last_jam_score = 0.0

    def set_baseline(self, baseline: BaselineStats):
        self.baseline = baseline

    def update(self, x_t: dict, frame_num: int) -> tuple:
        """feature 벡터 받아 레벨·jam_score 반환
        Returns: (level: str, jam_score: float)"""
        if self.baseline is None or self.baseline.is_fallback:
            jam = compute_jam_score_fallback(x_t)
        else:
            jam = compute_jam_score(x_t, self.baseline.lcs, self.cfg)

        x_t["rule_jam_score"] = jam  # feature 벡터에 역주입 (GRU 입력용)
        self._last_jam_score = jam

        raw = self._classify(jam)
        level = self._apply_hysteresis(raw)
        # ... (기존 히스테리시스 로직 그대로)

        return level, jam

    def get_level(self) -> str:     return self._current_level
    def get_jam_score(self) -> float: return self._last_jam_score
    def get_duration_sec(self, frame_num: int, fps: float) -> float: ...
```

### 7.4 TrafficAnalyzer (래퍼 — 인터페이스 불변)

```python
class TrafficAnalyzer:
    """기존 인터페이스 유지. 내부에서 신규 모듈 조합."""

    def __init__(self, cfg, frame_w, frame_h, fps,
                 flow_map=None, passage_tracker=None):
        # ← detector.py에서 초기화 시 flow_map, passage_tracker 전달
        self.cfg = cfg
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.feature_extractor = FeatureExtractor(cfg, ...)
        self.congestion_judge  = CongestionJudge(cfg, fps)
        self._density_map = np.zeros((cfg.grid_size, cfg.grid_size))
        self._vehicle_count = 0

    def update(self, tracks: list, speeds: dict, frame_num: int) -> None:
        """기존과 동일한 시그니처 유지"""
        self._vehicle_count = len(tracks)
        self._update_density_map(tracks)

        x_t = self.feature_extractor.compute(tracks, speeds, None, frame_num)
        if x_t is None:
            return  # 학습 중

        level, jam = self.congestion_judge.update(x_t, frame_num)

    # 아래 public 메서드는 시그니처·반환타입 모두 기존과 동일하게 유지
    def get_congestion_level(self) -> str:    return self.congestion_judge.get_level()
    def get_density_map(self) -> np.ndarray:  return self._density_map.copy()
    def get_avg_speed(self) -> float:         ... # speed_ratio × 100 (상대값)
    def get_occupancy(self) -> float:         ...
    def get_volume(self) -> float:            ...
    def get_duration_sec(self) -> float:      ...
    def get_affected_vehicles(self) -> int:   ...
    def get_jam_score(self) -> float:         return self.congestion_judge.get_jam_score()
```

**detector.py 초기화 (~96번째 줄):**

```python
self.passage_tracker   = PassageTracker(cfg, self.state)
self.feature_extractor = FeatureExtractor(cfg, self.state, self.passage_tracker)
self.congestion_judge  = CongestionJudge(cfg, fps=fps)
self.traffic_analyzer  = TrafficAnalyzer(cfg, frame_w=fw, frame_h=fh, fps=fps,
    flow_map=self.flow, passage_tracker=self.passage_tracker, congestion_judge=self.congestion_judge)
self.idm = IDManager(cfg, self.flow, self.state, self.passage_tracker)
```

---

## 8. 기존 파일 수정 위치 상세

### state.py

```python
# DetectorState.__init__() 마지막에 추가
self.entry_positions = {}   # {track_id: (fx, fy)} footpoint 진입 위치
```

### flow_map.py

```python
# FlowMap.__init__() 끝에 추가
self.speed_ref = np.zeros((grid_size, grid_size), np.float32)  # 셀별 정상 norm_speed

# FlowMap.reset() 끝에 추가
self.speed_ref[:] = 0

# 신규 메서드 추가
def learn_baseline(self, fx: float, fy: float, norm_speed: float):
    """SMOOTH 온라인 구간에서 셀별 정상 속도 EMA 갱신"""
    r = int(np.clip(fy / self.cell_h, 0, self.grid_size - 1))
    c = int(np.clip(fx / self.cell_w, 0, self.grid_size - 1))
    if self.speed_ref[r, c] == 0:
        self.speed_ref[r, c] = norm_speed
    else:
        self.speed_ref[r, c] = (1 - self.alpha) * self.speed_ref[r, c] + self.alpha * norm_speed

# flow_map.save() — baseline_stats 파라미터 추가
def save(self, path: Path, baseline_stats=None):
    data = {"version": 2, "flow": self.flow, "count": self.count,
            "speed_ref": self.speed_ref}
    if baseline_stats is not None:
        import dataclasses
        data["baseline_stats"] = dataclasses.asdict(baseline_stats)
    np.save(path, data)

# flow_map.load() — version 체크 추가
def load(self, path: Path) -> tuple:
    """(성공여부, BaselineStats or None) 반환"""
    ...
    version = data.get("version", 1)
    if version >= 2 and "speed_ref" in data:
        self.speed_ref = data["speed_ref"]
    baseline = None
    if version >= 2 and "baseline_stats" in data:
        baseline = BaselineStats(**data["baseline_stats"])
    return True, baseline
```

### detector.py 수정 위치 요약

| 위치 | 기존 코드 | 변경 내용 |
|------|-----------|-----------|
| 192~194번째 줄 (stabilize 직후) | `x1,y1,x2,y2,cx,cy = stabilize(...)` | `fx=(x1+x2)/2; fy=y2` 추가 |
| 130~131번째 줄 (first_seen 기록) | `st.first_seen_frame[tid]=frame_num` | `passage_tracker.on_entry(tid,fx,fy,frame_num)` 추가 |
| 165~170번째 줄 (학습 완료 블록) | 단순 종료 | §4.4 연장 조건 + finalize_baseline() 추가 |
| 242~248번째 줄 (온라인 학습 블록) | 조건 없이 learn_step | baseline freeze 조건 추가 |
| 루프 끝 (traffic_analyzer.update 직전) | `self.traffic_analyzer.update(...)` | passage_tracker.record_frame_stats() 먼저 호출 |

---

## 9. TDD 테스트 케이스 명세

### test_passage_tracker.py

| TC | 검증 내용 | 기대 결과 |
|----|-----------|-----------|
| PT-01 | on_entry → on_exit 순서대로 호출 | PassageRecord 1개 생성, dwell_frames 정확 |
| PT-02 | entry_exit_dist < min_passage_dist인 passage | is_valid=False |
| PT-03 | 완성 passage 5개 이상 → finalize_baseline() | free_flow_dwell = min(dwells) (n<10) |
| PT-04 | 완성 passage 15개 → finalize_baseline() | free_flow_dwell = percentile(dwells, 10) |
| PT-05 | 모두 빠른 차량 (short dwell) | LCS < 0.3 |
| PT-06 | 모두 느린 차량 (long dwell) | LCS > 0.6 |
| PT-07 | on_exit is_complete=False | is_fallback=True (passage 부족 시) |
| PT-08 | get_current_dwells() | 활성 차량 수만큼 dwell 값 반환 |

### test_congestion_judge.py

| TC | 검증 내용 | 기대 결과 |
|----|-----------|-----------|
| CJ-01 | baseline 없이 update() | fallback jam_score 반환, None 아님 |
| CJ-02 | x_t 모두 정상(ratio=1.0) | jam_score ≈ 0.0 |
| CJ-03 | x_t 모두 최악(ratio=0.0) | jam_score ≈ 1.0 |
| CJ-04 | jam_score < 0.25 | level = "SMOOTH" |
| CJ-05 | 0.25 ≤ jam_score < 0.55 | level = "SLOW" |
| CJ-06 | jam_score ≥ 0.55 | level = "CONGESTED" |
| CJ-07 | CONGESTED → 즉시 SMOOTH | 15초 동안 CONGESTED 유지 (히스테리시스) |
| CJ-08 | LCS=0.8 → threshold 완화 | smooth_threshold < 0.25 |
| CJ-09 | exit_rate_ratio > 1.3 | improvement_bonus 적용 → jam 감소 |
| CJ-10 | update() 100회 연속 호출 | 예외 없음 |

### test_traffic_analyzer.py (재작성)

기존 TC-01~13 → `test_traffic_analyzer_legacy.py`로 이름 변경 후 비활성화.
새 TC-01~10 작성:

| TC | 검증 내용 | 기대 결과 |
|----|-----------|-----------|
| TA-01 | baseline 없이 update() → get_congestion_level() | "SMOOTH" (학습 중) |
| TA-02 | set_baseline 후 빈 tracks → get_congestion_level() | "SMOOTH" |
| TA-03 | get_density_map() 반환 타입 | ndarray shape (15,15) |
| TA-04 | get_jam_score() 범위 | 0.0~1.0 |
| TA-05 | update() 50회 반복 | 예외 없음 |
| TA-06 | 정상 속도 tracks → SMOOTH | jam_score < 0.25 |
| TA-07 | 정지 차량 다수 → CONGESTED | jam_score ≥ 0.55 (충분한 dwell 후) |
| TA-08 | get_duration_sec() SMOOTH | 0.0 |
| TA-09 | get_duration_sec() CONGESTED 30프레임 후 | > 0.0 |
| TA-10 | get_affected_vehicles() 5대 중 3대 정지 | 3 |

---

## 10. 개발 단계 로드맵

| Phase | 핵심 변경 | 완료 기준 |
|-------|-----------|-----------|
| **Phase 1** | 규칙 기반 jam_score | pytest PASS + 실영상 레벨 출력 |
| **Phase 2** | GRU 병렬 추가 (40%) | Phase 1 테스트 유지 + GRU loss 수렴 |
| **Phase 3** | GRU primary (65%) | 가중치 조정만 — 구조 변경 없음 |

**Phase 1 우선. Phase 2·3은 Phase 1 완료 후 착수.**

---

## 11. 카메라 PTZ 대응

### Phase 1 (GRU 없음)

카메라 전환 감지 → `reset_for_relearn()` + `passage_tracker.reset()`.
전환 후 `grace_period_sec`(60초) 동안 화면에 "재보정 중..." 표시, 판정 유보.

```python
# detector.py camera_switch 감지 직후 (161~162번째 줄 부근)
if self.switch.check(frame, st.frame_num, st.cooldown_until):
    st.reset_for_relearn()
    self.flow.reset()
    self.passage_tracker.reset()        # ← 추가
    self.congestion_judge.clear()       # ← 추가 (히스테리시스 초기화)
    st.grace_until = st.frame_num + int(cfg.grace_period_sec * fps)  # ← 추가
```

### PTZ 유형별 처리

| 유형 | 감지 | 처리 |
|------|------|------|
| 순간 전환 | camera_switch.py | reset_for_relearn() + grace |
| 점진적 패닝 | camera_switch.py (점진 감지 주의) | 동일 |
| 줌만 변경 | 미감지 | normalized_mag이 자동 보정 (bbox_h도 비례 변함) |
| 야간→주간 | camera_switch.py 오감지 가능 | cooldown_frames(150)로 억제 |

---

## 12. traffic_analyzer.py 출력 인터페이스 (팀원 연동) traffic_analyzer.py 출력 인터페이스 (팀원 연동)

| 메서드 | 반환타입 | 내용 | 팀원 화면 |
|--------|---------|------|----------|
| `get_congestion_level()` | str | SMOOTH/SLOW/CONGESTED | SCR-01 배지 |
| `get_avg_speed()` | float | speed_ratio × 100 (상대값, %) | SCR-01 KPI |
| `get_density_map()` | ndarray(15,15) | 셀별 차량 수 | SCR-03 히트맵 |
| `get_occupancy()` | float | 점유율 % | SCR-01 KPI |
| `get_volume()` | float | 대/시 (추정) | SCR-01 KPI |
| `get_duration_sec()` | float | 정체 지속 시간 초 | SCR-04 |
| `get_affected_vehicles()` | int | 정체 영향 차량 수 | SCR-04 |
| `get_jam_score()` | float 0~1 | jam_score | 디버그용 |

> ⚠️ `get_avg_speed()`는 절대 km/h가 아닌 "정상 대비 %".
> 팀원 웹: "정상 대비 XX%" 또는 `speed_ratio × free_flow_speed(100)` 어림값으로 표시.

---

## 13. 역주행 로직 수정 원칙

> **역주행 탐지 코드는 원칙적으로 수정하지 않는다.**

| 파일 | 원칙 | Phase 1 허용 변경 |
|------|------|-------------------|
| `state.py` | 수정 자제 | `entry_positions` 필드 1개 추가 |
| `id_manager.py` | 수정 자제 | `__init__`에 `passage_tracker` 파라미터 추가, `cleanup()`에 15줄 추가 |
| `flow_map.py` | 수정 자제 | `speed_ref` ndarray, `learn_baseline()`, `save()`/`load()` 수정 |
| `detector.py` | 최소 수정 | footpoint 계산, 학습 연장 조건, freeze 조건, passage_tracker 연동 |
| `judge.py` | 수정 금지 | - |
| `tracker.py` | 수정 불필요 | - |
| `camera_switch.py` | 수정 불필요 | - |
| `bbox_stabilizer.py` | 수정 불필요 | - |

---

## 16. TDD 방법론

- 코드 작성 전 테스트 먼저 설계
- `tests/` 폴더에서 pytest 실행
- PASS 확인 후 다음 파일 진행
- **기존 TC-01~13 비활성화**: `test_traffic_analyzer.py` → `test_traffic_analyzer_legacy.py` 이름 변경, 신규 TA-01~10 작성

---

## 14. 마일스톤

| 마일스톤 | 날짜 | 상태 |
|----------|------|------|
| M1 분석완료 | 3/25 | ✅ 완료 |
| M2 버그수정 | 3/29 | 🔲 진행 전 |
| **M3 신기능** | **4/13** | **🔲 Phase 1 개발 시작** |
| M4 백엔드+프론트 | 4/24 | 🔲 |
| M5 배포 | 4/30 | 🔲 |
| M6 발표 | 5/7 | 🔲 |
