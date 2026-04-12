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

    def __init__(self, cfg, state, fps: float = 30.0):
        """FeatureExtractor 초기화.

        Args:
            cfg: DetectorConfig.
            state: DetectorState — frame_w, frame_h.
            fps: 영상 FPS — dwell_threshold_sec를 프레임 수로 변환하는 데 사용.
        """
        self.cfg = cfg                                 # 설정 객체 저장
        self.state = state                             # 런타임 상태 저장
        self._ready = False                            # set_baseline() 호출 후 True
        self._valid_cell_count_override: int | None = None  # 방향별 유효 셀 수 (None=전체 사용)

        # dwell_threshold_sec(초) → 프레임 수로 변환 (FPS 기반)
        # 30fps: 0.5초 × 30 = 15프레임 / 10fps: 0.5초 × 10 = 5프레임
        _dwell_sec = getattr(cfg, "dwell_threshold_sec", 0.5)
        self._dwell_thr_frames: int = max(1, int(_dwell_sec * fps))  # 최소 1프레임 보장

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

        # ── tid별 체류 상태 (dwell_cell_ratio 계산용) ────────────────
        # {tid: (cell_r, cell_c, first_frame_in_cell)}
        self._dwell_state: dict = {}

        # ── Cell Dwell EMA (핵심 정체 신호) ──────────────────────────
        # cell_dwell_ema[r][c]: 해당 셀이 얼마나 오래 점유됐는지 EMA 누적값 (0~1)
        #   점유 중 → ema 서서히 1 수렴 / 빈 셀 → ema 서서히 0 수렴
        # ID 변경 무관 (셀 점유 여부만 봄)
        # 정상 차량(2~5프레임/셀): peak ema ≈ 0.10~0.23 → 낮음
        # 정체 차량(30프레임+/셀): ema → 0.78+ → 높음
        self._cell_dwell_ema: np.ndarray | None = None  # 첫 compute()에서 초기화

        # ── occupied_cells 히스토리 (cell_persistence 계산용) ────────
        self._OCC_HIST_LEN: int = 31
        self._occ_history: collections.deque = collections.deque(maxlen=self._OCC_HIST_LEN)
        self._persist_ema: float = 0.0
        self._PERSIST_EMA_ALPHA: float = 0.30

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

        # ── flow map 기반 체류(dwell) 계산 ──────────────────────────────
        # 차량이 같은 그리드 셀에 dwell_threshold_frames 이상 머물면 체류 셀로 집계
        # nm 임계값 불필요 — 셀 이동 여부로만 판단 (CCTV 각도·거리 무관)
        _dwell_thr = self._dwell_thr_frames            # __init__에서 FPS 기반으로 계산된 임계값
        _dwell_cells_set: set = set()                  # 체류 차량이 점유한 고유 셀 집합
        for t in tracks:
            _tid = t["id"]
            if _tid not in speed_known_tids:           # 궤적 미확인 신규 차량 제외
                continue
            _cr = int(np.clip(t["cy"] / cell_h, 0, self.cfg.grid_size - 1))  # 현재 셀 행
            _cc = int(np.clip(t["cx"] / cell_w, 0, self.cfg.grid_size - 1))  # 현재 셀 열
            if _tid in self._dwell_state:
                _prev_r, _prev_c, _first_f = self._dwell_state[_tid]
                if _cr == _prev_r and _cc == _prev_c:  # 같은 셀에 머물고 있음
                    if frame_num - _first_f >= _dwell_thr:  # 체류 임계값 초과
                        _dwell_cells_set.add((_cr, _cc))    # 체류 셀로 등록
                else:                                  # 셀이 바뀜 → 체류 리셋
                    self._dwell_state[_tid] = (_cr, _cc, frame_num)
            else:                                      # 첫 등장 → 체류 시작 기록
                self._dwell_state[_tid] = (_cr, _cc, frame_num)

        # 이번 프레임에 없는 tid 체류 기록 삭제 (메모리 누수 방지)
        for _old in list(self._dwell_state.keys()):
            if _old not in speed_known_tids:
                del self._dwell_state[_old]

        # ── bbox_coverage / flow_occupancy / dwell_cell_ratio ────────────
        # 기준: valid_cell_count (flow_map이 실제 학습한 셀 수)
        # 도로 크기(2차선/4차선)에 자동 적응 — density_max_vehicles 불필요
        #
        # 유효 셀 수: 방향별 override > flow_map 실측 > 전체 그리드 순서로 사용
        if self._valid_cell_count_override is not None:    # 방향별 셀 수 주입됨
            valid_cell_count = self._valid_cell_count_override
        elif flow_map is not None and hasattr(flow_map, "count"):
            valid_cell_count = int(np.sum(flow_map.count > 0))  # 학습된 유효 셀 수
        else:
            valid_cell_count = self.cfg.grid_size * self.cfg.grid_size  # fallback: 전체

        # 차량 footpoint가 위치한 고유 셀 집합 — 신규 차량(speed 미확인) 제외
        occupied_cells_set = set(
            (int(np.clip(t["cy"] / cell_h, 0, self.cfg.grid_size - 1)),
             int(np.clip(t["cx"] / cell_w, 0, self.cfg.grid_size - 1)))
            for t in tracks if t["id"] in speed_known_tids
        )
        occupied_cells = len(occupied_cells_set)

        # flow_occupancy: 차량이 점유한 셀 / 유효 셀 전체
        # 정체 시 유효 셀 대부분이 차로 채워지면 → 1에 수렴
        flow_occupancy = float(np.clip(
            occupied_cells / max(valid_cell_count, 1), 0.0, 1.0
        ))

        # dwell_cell_ratio: 체류 셀 / 유효 셀 전체
        # 차량이 안 움직이는 셀 비율 — 차선 수 무관하게 도로 면적 기준으로 자동 정규화
        dwell_cell_ratio = float(np.clip(
            len(_dwell_cells_set) / max(valid_cell_count, 1), 0.0, 1.0
        ))

        bbox_coverage = flow_occupancy                     # 하위 호환 별칭 (= flow_occupancy)
        density_score = bbox_coverage                      # 하위 호환 별칭
        dwelling_density = dwell_cell_ratio                # 하위 호환 별칭

        # ── Cell Dwell EMA 업데이트 ──────────────────────────────────
        # 각 셀이 점유된 상태가 얼마나 지속됐는지 EMA로 누적
        # 정체: 같은 셀에 차량이 오래 머묾 → ema 높게 누적
        # 원활: 차량이 빠르게 통과 → ema 낮게 유지
        if self._cell_dwell_ema is None:
            self._cell_dwell_ema = np.zeros(
                (self.cfg.grid_size, self.cfg.grid_size), dtype=float
            )
        _cde_up   = getattr(self.cfg, "cell_dwell_ema_up",   0.05)
        _cde_down = getattr(self.cfg, "cell_dwell_ema_down", 0.02)

        for _r in range(self.cfg.grid_size):
            for _c in range(self.cfg.grid_size):
                if (_r, _c) in occupied_cells_set:
                    self._cell_dwell_ema[_r, _c] += _cde_up * (1.0 - self._cell_dwell_ema[_r, _c])
                else:
                    self._cell_dwell_ema[_r, _c] *= (1.0 - _cde_down)

        # ── cell_dwell_score 재설계 ──────────────────────────────────
        # 기존: sum(ema) / valid_cell_count → 49로 나눠 항상 희석
        # 수정: 점유 셀 평균 강도 × 점유 밀도 조합
        #   강도(intensity): 실제 점유 중인 셀들의 ema 평균 (얼마나 오래 머물렀는가)
        #   밀도(density)  : 점유 셀 수 / valid_cell_count        (얼마나 많이 막혔는가)
        #   둘을 곱하면: 많이 막히고 + 오래 머물수록 높아짐

        # 현재 점유 중인 셀들의 dwell_ema 값만 추출 (ema > 0.03 = 최소 의미있는 점유)
        # ema=0.03 미만은 방금 진입한 셀이거나 오래전에 나간 잔류값 → 노이즈로 처리
        _occupied_emas = [
            self._cell_dwell_ema[_r, _c]
            for (_r, _c) in occupied_cells_set
            if self._cell_dwell_ema[_r, _c] > 0.03
        ]

        if _occupied_emas:
            _cds_intensity = float(np.mean(_occupied_emas))  # 점유 셀 평균 EMA 강도 (얼마나 오래 머물렀는가)
            # 단방향 차선 셀 수 결정
            # - override 주입된 경우: 이미 단방향 값 → 그대로 사용
            # - 전체 셀 사용 중(양방향 통합): // 2로 단방향 추정
            if self._valid_cell_count_override is not None:
                _lane_cell_count = valid_cell_count     # 이미 단방향 값
            else:
                _lane_cell_count = max(valid_cell_count // 2, 1)  # 양방향 → 단방향 추정
            _cds_density = min(1.0, len(_occupied_emas) / _lane_cell_count)  # 단방향 기준 점유 비율 (0~1)
            # 최종 score = 강도 × (0.3 + 0.7 × 밀도)
            #   밀도=0 이어도 강도가 높으면 0.3 × intensity로 최소 반영
            #   밀도=1 이면 1.0 × intensity = 최대값
            cell_dwell_score = float(np.clip(
                _cds_intensity * (0.3 + 0.7 * _cds_density), 0.0, 1.0
            ))
        else:
            _cds_intensity = 0.0   # 점유 셀 없음 → 강도 0
            _cds_density   = 0.0   # 점유 셀 없음 → 밀도 0
            cell_dwell_score = 0.0 # 정체 신호 없음
        # ── cell_persistence: 30프레임 전 점유 셀과 현재의 Jaccard 유사도 ──
        # 문제: 20×20 그리드 셀=64×36px → 정체 차량이 90px만 이동해도 셀 이탈 → persist 낮음
        # 해결: 2×2 블록 코어스 그리드(10×10, 셀=128×72px)로 다운샘플링
        #       정체에서 90px 이동해도 128px 셀 안에 머뭄 → persist 정확하게 높아짐
        #       occ/dwell은 기존 20×20 유지 (세밀도 필요)
        _coarse_occ = frozenset(
            (int(np.clip(t["cy"] / cell_h, 0, self.cfg.grid_size - 1)) // 2,  # 2×2 블록
             int(np.clip(t["cx"] / cell_w, 0, self.cfg.grid_size - 1)) // 2)
            for t in tracks if t["id"] in speed_known_tids
        )
        self._occ_history.append(_coarse_occ)                    # 코어스 셀 집합 deque에 추가 (maxlen=31)
        if len(self._occ_history) >= self._OCC_HIST_LEN:         # 31프레임 이력이 찼을 때만 계산
            _prev_occ = self._occ_history[0]                     # deque[0] = 30프레임 전 코어스 셀 집합
            _inter = len(_coarse_occ & _prev_occ)                # 교집합: 30f 전과 지금 모두 점유된 셀 수
            _union = len(_coarse_occ | _prev_occ)                # 합집합: 둘 중 하나라도 점유된 셀 수
            _raw_persist = _inter / _union if _union > 0 else 0.0  # Jaccard 유사도 (0=완전 이동, 1=완전 정체)
            # EMA로 스무딩: 순간 Jaccard가 프레임마다 튀는 것을 완화
            self._persist_ema = (self._PERSIST_EMA_ALPHA * _raw_persist
                                 + (1.0 - self._PERSIST_EMA_ALPHA) * self._persist_ema)
        # 이력 부족(초기 31프레임 미만)이면 _persist_ema=0.0 유지 — 정체 신호 억제
        cell_persistence = self._persist_ema                     # EMA 평활화된 Jaccard 값

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

        # ── slow_cell_density / stop_cell_density ────────────────────
        # 핵심 설계 변경: 비율(ratio) 대신 절대 밀도(count / road_capacity) 사용
        #
        # 문제: slow_ratio = slow_count / detected_count
        #   → 1대만 있어도 slow_ratio=1.0 → slow_density=0.22 → jam=0.29
        #
        # 해결: road_capacity = valid_cell_count × NM_WIN (도로 최대 수용량)
        #   → slow_cell_density = _hist_slow / road_capacity
        #   → 1대 slow / (20셀×5프레임) = 5/100 = 0.05 → jam ≈ 0.13 ✓
        #   → 10대 slow / (20셀×5프레임) = 50/100 = 0.50 → jam ≈ 0.80 ✓
        # road_capacity: valid_cell_count 기준은 셀 수(49)가 실제 차량 수(10~15)보다
        # 훨씬 커서 scd가 0.25 이상 못 올라감 → density_max_vehicles(방향당 최대 차량) 사용
        _density_max = getattr(self.cfg, "density_max_vehicles", 40.0) / 2  # 방향당 절반
        _road_capacity = int(_density_max) * self._NM_WIN  # 방향당 최대 관측 수
        _pure_slow_hist = _hist_slow - _hist_stop          # 순수 서행 관측 수 (정지 제외)
        slow_cell_density = _pure_slow_hist / max(_road_capacity, 1)   # 서행 밀도
        stop_cell_density = _hist_stop / max(_road_capacity, 1)        # 정지 밀도

        # ── nm_variance: 속도 분산 (정체 징후 보조 신호) ─────────────
        # 정체: 정지+서행 혼재 → 분산 높음 / 원활: 모두 비슷 → 분산 낮음
        if len(_all_nm_vals) >= 4:
            _nm_std = float(np.std(_all_nm_vals))
            nm_variance_score = float(np.clip(_nm_std / slow_upper_nm, 0.0, 1.0))
        else:
            nm_variance_score = 0.0

        # ── 하위 호환용 slow_ratio / stop_ratio (GRU feature용) ───────
        if _hist_total >= 2:
            slow_ratio = _pure_slow_hist / _hist_total
            stop_ratio = _hist_stop / _hist_total
        else:
            slow_ratio = 0.0
            stop_ratio = 0.0

        slow_density = slow_cell_density                   # congestion_judge 전달용 별칭

        # ── 디버그 출력 (30프레임마다) ───────────────────────────────
        if frame_num % 30 == 0:
            print(f"[FE] f={frame_num} known={speed_known_count} "
                  f"occ={flow_occupancy:.3f} cds={cell_dwell_score:.3f} "
                  f"dwell={dwell_cell_ratio:.3f} persist={cell_persistence:.3f} valid={valid_cell_count}")

        # ── feature 딕셔너리 조립 ────────────────────────────────────
        return {                                       # feature 벡터
            "norm_speed_ratio":      norm_speed_ratio,       # [0] 속도 비율 (자기보정 baseline 기준)
            "nm_baseline_valid":     _nm_baseline_valid,     # baseline 준비 여부
            "stop_ratio":            stop_ratio,             # [1] 정지 비율 (GRU용)
            "slow_ratio":            slow_ratio,             # [1.5] 서행 비율 (GRU용)
            "slow_density":          slow_density,           # 하위 호환 별칭
            "slow_cell_density":     slow_cell_density,      # 서행 차량수 / road_capacity (GRU용)
            "stop_cell_density":     stop_cell_density,      # 정지 차량수 / road_capacity (GRU용)
            "nm_variance_score":     nm_variance_score,      # nm 분산 (GRU 보조)
            "density_score":         density_score,          # bbox_coverage 별칭 (하위 호환)
            "bbox_coverage":         bbox_coverage,          # 도로 면적 대비 셀 점유율
            "flow_occupancy":        flow_occupancy,         # 차량 점유 셀 / 유효 셀 (valid_cell_count 기준)
            "cell_dwell_score":      cell_dwell_score,       # 셀 누적 점유 EMA 합 / valid_cell_count (핵심)
            "dwell_cell_ratio":      dwell_cell_ratio,       # 체류 셀 / 유효 셀 (valid_cell_count 기준)
            "cell_persistence":      cell_persistence,       # 30프레임 전과 현재 점유셀 Jaccard 유사도
            "dwelling_density":      dwelling_density,       # = dwell_cell_ratio (하위 호환)
            "velocity_deficit_ratio": velocity_deficit_ratio, # 평균 속도 부족률 (-1=미학습)
            "deficit_count":         deficit_count,          # speed_ref 유효 차량 수
            "vdr_ready":             _vdr_ready,             # warmup 완료 여부
            
                "known_vehicle_count":   speed_known_count,  # 궤적 확인된 차량 수 (저규모 가드용)
            "occupied_cell_count":   occupied_cells,     # 현재 점유 셀 수 (저규모 가드용)
            "valid_cell_count":      valid_cell_count,   # 유효 셀 수 (분모 기준)

            "rule_jam_score":        0.0,                    # jam_score (CJ 채움)
        }
