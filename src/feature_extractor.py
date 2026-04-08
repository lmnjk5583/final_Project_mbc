# 파일 경로: C:\final_pj\src\feature_extractor.py
# 역할: 매 프레임 tracks+speeds를 받아 feature 벡터를 산출한다.
# 의존성: numpy(서드파티), collections(표준)

import collections                                     # deque — tid별 nm 슬라이딩 윈도우
import numpy as np                                     # clip, mean 등 수치 연산


# ======================================================================
# FeatureExtractor — feature 벡터 계산기
# ======================================================================

class FeatureExtractor:
    """매 프레임 tracks·speeds를 받아 정체 판정용 feature 벡터를 산출한다.

    feature 벡터:
    ┌─────┬───────────────────┬──────────────────────────────────────────────────────┐
    │ idx │ 이름              │ 계산식                                               │
    ├─────┼───────────────────┼──────────────────────────────────────────────────────┤
    │  0  │ norm_speed_ratio  │ median(upper_50%_nm_median) / self-calibrating ref  │
    │  1  │ stop_ratio        │ (nm_median<0.06 차량) / speed_known × reliability²  │
    │ 1.5 │ slow_ratio        │ (0.06≤nm_median<0.70, 정지 제외) / speed_known × r² │
    │  2  │ bbox_coverage     │ occupied_cells(궤적확인 차량만) / valid_cell_count   │
    │  3  │ rule_jam_score    │ 0.0 (congestion_judge가 채워넣음)                   │
    └─────┴───────────────────┴──────────────────────────────────────────────────────┘

    Parameters
    ----------
    cfg : DetectorConfig
        grid_size, norm_stop_threshold, slow_upper_nm 등 사용.
    state : DetectorState
        frame_w, frame_h 참조.
    """

    def __init__(self, cfg, state):
        """FeatureExtractor 초기화.

        Args:
            cfg: DetectorConfig.
            state: DetectorState — frame_w, frame_h.
        """
        self.cfg = cfg                                 # 설정 객체 저장
        self.state = state                             # 런타임 상태 저장
        self._ready = False                            # set_baseline() 호출 후 True
        self._valid_cell_count_override: int | None = None  # 방향별 유효 셀 수 (None=전체 사용)

        # ── 방향별 자기보정 nm baseline (비대칭 EMA) ─────────────────
        self._nm_baseline: float = 0.0                 # 방향 고유 정상속도 기준 (자동 보정)
        self._nm_baseline_count: int = 0               # 누적 업데이트 횟수 (warmup 판단용)

        # ── tid별 nm 슬라이딩 윈도우 ─────────────────────────────────
        # 순간 nm은 bbox jitter·차량 출입으로 튀기 때문에
        # 최근 _NM_WIN 프레임 nm의 중앙값으로 slow/stop 판정해 안정화
        self._NM_WIN: int = 5                          # 윈도우 크기 (프레임)
        self._nm_history: dict = {}                    # {tid: deque([nm, ...], maxlen=5)}

        # ── velocity_deficit 세션 warmup 카운터 ──────────────────────
        self._speed_ref_warmed_frames: int = 0         # vdr 유효 프레임 누적 수
        self._SPEED_REF_WARMUP: int = 150              # 이 값 이상이면 vdr 신뢰

        # ── slow_ratio / stop_ratio EMA (프레임 간 비율 튐 흡수) ─────
        # 차량 출입으로 speed_known_count가 바뀌면 slow_ratio가 프레임마다 크게 달라짐
        # EMA로 스무딩해서 jam 계산에 안정된 값 전달

    # ── 준비 신호 (학습 완료 후 호출) ────────────────────────────────
    def set_ready(self):
        """학습 완료 신호. 이후 compute()가 feature 벡터를 반환한다."""
        self._ready = True                             # feature 계산 활성화

    # ── 방향별 유효 셀 수 설정 ────────────────────────────────────────
    def set_valid_cell_count(self, n: int):
        """방향별 유효 셀 수를 설정한다 (bbox_coverage 원근 보정용).

        Args:
            n: 이 방향에 속하는 유효 flow_map 셀 수.
        """
        self._valid_cell_count_override = max(n, 1)   # 0 방지 후 저장

    # ── feature 벡터 계산 ────────────────────────────────────────────
    def compute(self, tracks: list, speeds: dict,
                flow_map, frame_num: int) -> dict | None:
        """매 프레임 호출 — feature 벡터를 계산한다.

        Args:
            tracks: [{id, x1, y1, x2, y2, cx, cy, ...}, ...].
            speeds: {track_id: mag(픽셀 이동량)}.
            flow_map: FlowMap 객체 (bbox_coverage 셀 수 fallback용).
            frame_num: 현재 프레임 번호.

        Returns:
            dict(feature 벡터) — 준비 안 됐으면 None.
        """
        if not self._ready:                            # 학습 완료 전이면
            return None                                # feature 계산 불가 → None

        # ── 차량별 normalized_mag 계산 ───────────────────────────────
        norm_mags = []                                 # 차량별 원근 보정 속도 리스트
        stopped_count = 0                              # 정지 차량 카운터 (nm < 0.06)
        slow_count = 0                                 # 서행 차량 카운터 (0.06 ≤ nm < slow_upper_nm)
        norm_stop_thr = getattr(                       # norm_stop_threshold 없으면 구버전 호환
            self.cfg, "norm_stop_threshold", 0.05
        )
        slow_upper_nm = getattr(                       # 서행 상한 nm — 역주행 게이트(0.15)와 별개
            self.cfg, "slow_upper_nm", 0.50            # 기본 0.50: nm≥0.50 → 정상 주행
        )
        nm_cy_k = getattr(self.cfg, "nm_cy_correction_k", 0.0)  # cy 보정 계수 (0=비활성)
        min_bbox_h = getattr(                          # min_bbox_h 없으면 구버전 호환 (30px)
            self.cfg, "min_bbox_h", 30.0
        )
        # ── 셀 크기 사전 계산 (루프 내 cell_r/c 계산 + bbox_coverage 공용) ──
        cell_w = self.state.frame_w / self.cfg.grid_size   # 셀 너비 (픽셀)
        cell_h = self.state.frame_h / self.cfg.grid_size   # 셀 높이 (픽셀)

        speed_known_count = 0                          # 궤적 확인된 차량 수 (신규 제외)
        speed_known_tids = set()                       # speeds 확인된 tid 집합 (bbox_coverage 필터용)
        speed_known_tids_deficit = {}                  # {tid: velocity_deficit} — speed_ref 학습된 셀만
        for t in tracks:                               # 각 차량 순회
            raw_bbox_h = t["y2"] - t["y1"]            # 바운딩박스 높이 (픽셀)
            bbox_h = max(raw_bbox_h, min_bbox_h)       # 최솟값 클램프 — 원거리 소형 박스 과대평가 방지
            tid = t["id"]                              # 트랙 ID
            if tid not in speeds:                      # speeds에 없으면 신규 → 완전 제외
                continue
            mag = speeds[tid]                          # 속도 조회
            speed_known_count += 1                     # 궤적 확인 차량 수 증가
            speed_known_tids.add(tid)                  # bbox_coverage 필터 목록에 추가

            if mag <= 0:                               # speeds=0: 실제 정지 확정
                # nm 슬라이딩 윈도우: 정지는 nm=0으로 기록
                if tid not in self._nm_history:
                    self._nm_history[tid] = collections.deque(maxlen=self._NM_WIN)
                self._nm_history[tid].append(0.0)
                # 슬라이딩 중앙값으로 판정
                _med = float(np.median(self._nm_history[tid]))
                if _med < norm_stop_thr:               # 중앙값 기준 정지
                    stopped_count += 1
                slow_count += 1                        # 정지는 서행의 부분집합 (중앙값 무관)
                continue

            nm = mag / bbox_h                          # normalized_mag (원근 보정)
            if nm_cy_k > 0:                            # cy 보정 활성화 시
                cy_ratio = t["cy"] / max(self.state.frame_h, 1)  # 0(상단/원거리)~1(하단/근거리)
                denom = 1.0 + nm_cy_k * (2.0 * cy_ratio - 1.0)  # 대칭 보정
                nm = nm / max(denom, 0.1)              # nm 보정

            # ── nm 슬라이딩 윈도우 업데이트 ──────────────────────────
            if tid not in self._nm_history:
                self._nm_history[tid] = collections.deque(maxlen=self._NM_WIN)
            self._nm_history[tid].append(nm)           # 현재 nm 기록

            # 슬라이딩 중앙값으로 slow/stop 판정 (순간 noise 흡수)
            _med_nm = float(np.median(self._nm_history[tid]))
            norm_mags.append(_med_nm)                  # 속도 목록에 중앙값 추가
            if _med_nm < norm_stop_thr:                # 중앙값 nm < 0.06 → 정지
                stopped_count += 1
                slow_count += 1                        # 정지 ⊂ 서행
            elif _med_nm < slow_upper_nm:              # 0.06 ≤ 중앙값 nm < 0.70 → 서행
                slow_count += 1

            # ── velocity_deficit 계산 (flow_map.speed_ref 활용) ────────
            # speed_ref[cell] = SMOOTH 구간에서 학습된 이 위치의 정상 nm
            # deficit = 1 - nm / speed_ref  (0=정상속도, 1=완전정지)
            # speed_ref가 0이면(미학습) fallback으로 slow_upper_nm 사용
            _cell_r = int(np.clip(t["cy"] / cell_h, 0, self.cfg.grid_size - 1))
            _cell_c = int(np.clip(t["cx"] / cell_w, 0, self.cfg.grid_size - 1))
            _ref_nm = (float(flow_map.speed_ref[_cell_r, _cell_c])
                       if (flow_map is not None
                           and hasattr(flow_map, "speed_ref")
                           and flow_map.speed_ref[_cell_r, _cell_c] > 0.01)
                       else 0.0)                       # 0이면 deficit 계산 스킵
            if _ref_nm > 0.01:                         # speed_ref 학습된 셀만
                _deficit = float(np.clip(1.0 - _med_nm / _ref_nm, 0.0, 1.0))
                speed_known_tids_deficit[tid] = _deficit  # tid별 deficit 저장

        # ── bbox_coverage: 셀 점유율 (cell occupancy) 방식 ──────────────
        # cell_w / cell_h 는 루프 전에 이미 계산됨

        # 유효 셀 수: 방향별 override > flow_map 실측 > 전체 그리드 순서로 사용
        if self._valid_cell_count_override is not None:    # 방향별 셀 수 주입됨
            valid_cell_count = self._valid_cell_count_override
        elif flow_map is not None and hasattr(flow_map, "count"):
            valid_cell_count = int(np.sum(flow_map.count > 0))  # 학습된 유효 셀 수
        else:
            valid_cell_count = self.cfg.grid_size * self.cfg.grid_size  # fallback: 전체

        # 차량 footpoint가 위치한 고유 셀 집합 — 신규 차량(speed 미확인) 제외
        # 신규 차량 포함 시: tracks에 등장만 해도 bbox_coverage 상승 → 차 없어도 jam 튐
        occupied_cells = len(set(
            (int(np.clip(t["cy"] / cell_h, 0, self.cfg.grid_size - 1)),   # 행 인덱스
             int(np.clip(t["cx"] / cell_w, 0, self.cfg.grid_size - 1)))   # 열 인덱스
            for t in tracks if t["id"] in speed_known_tids                # 궤적 확인된 차량만
        ))
        bbox_coverage = float(np.clip(                     # 셀 점유율 (0~1)
            occupied_cells / max(valid_cell_count, 1), 0.0, 1.0
        ))
        density_score = bbox_coverage                      # 하위 호환 별칭

        # ── norm_speed_ratio 계산 ─────────────────────────────────────
        # 상위 50% 중앙값 사용: 정체 차량 소수가 nm을 낮춰도 정상 주행 차량의 속도를 반영
        sorted_nms = sorted(norm_mags)                 # nm 오름차순 정렬
        upper_half = sorted_nms[len(sorted_nms) // 2:]  # 상위 50% 슬라이싱
        rep_norm_mag = float(np.median(upper_half)) if norm_mags else 0.0  # 상위 50% 중앙값

        # ── 방향별 자기보정 nm baseline 업데이트 (비대칭 EMA) ─────────
        _ema_up   = getattr(self.cfg, "nm_baseline_ema_up",   0.05)
        _ema_down = getattr(self.cfg, "nm_baseline_ema_down", 0.005)
        _warmup   = getattr(self.cfg, "nm_baseline_warmup",   300)
        if rep_norm_mag > 0 and speed_known_count >= 2:    # 차량 2대 이상일 때만 업데이트
            if self._nm_baseline_count == 0:               # 첫 업데이트 — 초기값 설정
                self._nm_baseline = rep_norm_mag
            else:
                _alpha = _ema_up if rep_norm_mag >= self._nm_baseline else _ema_down
                self._nm_baseline = _alpha * rep_norm_mag + (1.0 - _alpha) * self._nm_baseline
            self._nm_baseline_count += 1

        # 우선순위: 자기보정 baseline > override > 고정 fallback(0.15)
        _nm_baseline_valid = (                         # warmup 완료 + 유효값
            self._nm_baseline_count >= _warmup and self._nm_baseline > 0.01
        )
        if _nm_baseline_valid:                         # 자기보정 baseline 사용
            _speed_ref = self._nm_baseline
        else:                                          # warmup 중 fallback
            _ref_override = getattr(self.cfg, "norm_speed_ref_override", 0.0)
            _speed_ref = _ref_override if _ref_override > 0 else 0.15  # 고정 fallback

        norm_speed_ratio = float(np.clip(              # 속도 비율 clip(0, 1)
            rep_norm_mag / _speed_ref, 0.0, 1.0
        ))

        # ── velocity_deficit_ratio 집계 ──────────────────────────────
        _deficit_vals = list(speed_known_tids_deficit.values())
        deficit_count = len(_deficit_vals)             # speed_ref 유효 차량 수
        velocity_deficit_ratio = (
            float(np.mean(_deficit_vals)) if deficit_count > 0 else -1.0
        )                                              # -1 = speed_ref 미학습

        # 세션 warmup: deficit_count > 0인 프레임을 누적, 임계값 이상이면 vdr 신뢰
        if deficit_count > 0:
            self._speed_ref_warmed_frames += 1
        _vdr_ready = self._speed_ref_warmed_frames >= self._SPEED_REF_WARMUP

        # ── nm_history 만료 처리: 이번 프레임에 없는 tid 제거 ────────
        for old_tid in list(self._nm_history.keys()):
            if old_tid not in speed_known_tids:            # 이번 프레임에 없는 차량
                del self._nm_history[old_tid]              # 윈도우 삭제 (메모리 누수 방지)

        # ── stop_ratio / slow_ratio: nm_history 전체 관측 집계 ──────────
        # 문제: 이번 프레임 차량 수 기준 비율은 차량 출입마다 크게 달라짐
        #   프레임 t  : 차량 8대 slow 8대 → 1.00
        #   프레임 t+1: 차량 9대 slow 2대 → 0.22 (신규 7대 nm 히스토리 없음)
        # 해결: nm_history 전체(차량별 최근 5프레임) 관측값 합산으로 비율 계산
        #   차량 10대 × 5프레임 = 50관측 → 1대 출입 시 비율 변화 1/50 수준
        _hist_slow = 0                                     # 히스토리 전체 slow 관측 수
        _hist_stop = 0                                     # 히스토리 전체 stop 관측 수
        _hist_total = 0                                    # 히스토리 전체 관측 수
        _all_nm_vals = []                                  # nm 분산 계산용 전체 관측값
        for _, _hist_q in self._nm_history.items():       # 모든 차량 히스토리 순회
            for _nm_h in _hist_q:                         # 해당 차량의 최근 nm 값들
                _hist_total += 1
                _all_nm_vals.append(_nm_h)
                if _nm_h < norm_stop_thr:                  # 정지
                    _hist_stop += 1
                    _hist_slow += 1                        # 정지 ⊂ 서행
                elif _nm_h < slow_upper_nm:                # 서행
                    _hist_slow += 1

        if _hist_total >= 2:                               # 최소 2관측 이상
            _pure_slow_hist = _hist_slow - _hist_stop      # 순수 서행 관측 수
            slow_ratio = _pure_slow_hist / _hist_total
            stop_ratio = _hist_stop / _hist_total
        else:                                              # 관측 없음
            slow_ratio = 0.0
            stop_ratio = 0.0

        # ── nm_variance: 속도 분산 (정체 징후 보조 신호) ─────────────
        # 정체: 정지 차량 + 서행 차량 혼재 → nm 분산 높음
        # 원활: 모든 차량 비슷한 속도 → nm 분산 낮음
        # 정규화: slow_upper_nm 기준으로 [0, 1] 스케일
        if len(_all_nm_vals) >= 4:                         # 최소 4관측 이상일 때만 신뢰
            _nm_std = float(np.std(_all_nm_vals))          # nm 표준편차
            nm_variance_score = float(np.clip(             # slow_upper_nm 기준 정규화
                _nm_std / slow_upper_nm, 0.0, 1.0
            ))
        else:
            nm_variance_score = 0.0                        # 관측 부족 → 0

        # ── slow_density: 속도×밀도 결합 신호 ───────────────────────────
        # slow_ratio만 쓰면 차가 없어져도 비율이 올라 jam 상승하는 역효과
        # bbox_coverage(밀도)를 곱해 "얼마나 많은 차량이 느린가"를 단일 값으로
        slow_density = slow_ratio * float(np.sqrt(bbox_coverage))  # 속도×밀도 결합

        # ── 디버그 출력 (30프레임마다) ───────────────────────────────
        if frame_num % 30 == 0:
            print(f"[FE] f={frame_num} known={speed_known_count} "
                  f"slow_r={slow_ratio:.3f} stop_r={stop_ratio:.3f} "
                  f"bbox={bbox_coverage:.3f} slow_d={slow_density:.3f} "
                  f"nm_var={nm_variance_score:.3f}")

        # ── feature 딕셔너리 조립 ────────────────────────────────────
        return {                                       # feature 벡터
            "norm_speed_ratio":      norm_speed_ratio,       # [0] 속도 비율 (자기보정 baseline 기준)
            "nm_baseline_valid":     _nm_baseline_valid,     # baseline 준비 여부
            "stop_ratio":            stop_ratio,             # [1] 정지 비율 (nm_median < 0.06)
            "slow_ratio":            slow_ratio,             # [1.5] 순수 서행 비율 (정지 제외)
            "slow_density":          slow_density,           # [1.6] 속도×밀도 결합 (차없을때 jam 상승 방지)
            "nm_variance_score":     nm_variance_score,      # [1.7] nm 분산 (정체 징후 보조)
            "density_score":         density_score,          # [2] bbox_coverage 별칭 (하위 호환)
            "bbox_coverage":         bbox_coverage,          # [2] 도로 면적 대비 셀 점유율 (궤적확인 차량만)
            "velocity_deficit_ratio": velocity_deficit_ratio, # [2.5] 평균 속도 부족률 (-1=미학습)
            "deficit_count":         deficit_count,          # speed_ref 유효 차량 수
            "vdr_ready":             _vdr_ready,             # warmup 완료 여부
            "rule_jam_score":        0.0,                    # [3] jam_score (CJ 채움)
        }
