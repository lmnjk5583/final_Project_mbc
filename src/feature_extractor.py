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

    주요 feature:
      norm_speed_ratio  : 상위 50% nm 중앙값 / 자기보정 baseline — 속도 비율
      flow_occupancy    : 차량 점유 셀 / 유효 셀 — 순간 도로 밀도
      cell_dwell_score  : 셀 누적 점유 EMA 합 / 유효 셀 — 체류 강도 배경 신호
      dwell_cell_ratio  : 체류 셀 / 현재 점유 셀 — 실제로 멈춘 셀 비율
      cell_persistence  : 30프레임 전·현재 점유 셀 Jaccard — 점유 패턴 고착도
      rule_jam_score    : 0.0 초기값, CongestionJudge가 계산 후 역주입

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

    # ── 준비 신호 (학습 완료 후 호출) ────────────────────────────────
    def set_ready(self):
        """학습 완료 신호. 이후 compute()가 feature 벡터를 반환한다."""
        self._ready = True                             # feature 계산 활성화

    # ── 방향별 유효 셀 수 설정 ────────────────────────────────────────
    def set_valid_cell_count(self, n: int):
        """방향별 유효 셀 수를 설정한다 (flow_occupancy·cell_dwell_score 정규화 분모용).

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
            flow_map: FlowMap 객체 (flow_occupancy 셀 수 fallback용).
            frame_num: 현재 프레임 번호.

        Returns:
            dict(feature 벡터) — 준비 안 됐으면 None.
        """
        if not self._ready:                            # 학습 완료 전이면
            return None                                # feature 계산 불가 → None

        # ── 차량별 normalized_mag 계산 ───────────────────────────────
        norm_mags = []                                 # 차량별 원근 보정 속도 리스트
        nm_cy_k = getattr(self.cfg, "nm_cy_correction_k", 0.0)  # cy 보정 계수 (0=비활성)
        min_bbox_h = getattr(                          # min_bbox_h 없으면 구버전 호환 (30px)
            self.cfg, "min_bbox_h", 30.0
        )
        # ── 셀 크기 사전 계산 (루프 내 cell_r/c 계산 공용) ──────────────
        cell_w = self.state.frame_w / self.cfg.grid_size   # 셀 너비 (픽셀)
        cell_h = self.state.frame_h / self.cfg.grid_size   # 셀 높이 (픽셀)

        speed_known_count = 0                          # 궤적 확인된 차량 수 (신규 제외)
        speed_known_tids = set()                       # speeds 확인된 tid 집합 (점유 셀 필터용)
        for t in tracks:                               # 각 차량 순회
            raw_bbox_h = t["y2"] - t["y1"]            # 바운딩박스 높이 (픽셀)
            bbox_h = max(raw_bbox_h, min_bbox_h)       # 최솟값 클램프 — 원거리 소형 박스 과대평가 방지
            tid = t["id"]                              # 트랙 ID
            if tid not in speeds:                      # speeds에 없으면 신규 → 완전 제외
                continue
            mag = speeds[tid]                          # 속도 조회
            speed_known_count += 1                     # 궤적 확인 차량 수 증가
            speed_known_tids.add(tid)                  # 점유 셀 계산 필터 목록에 추가

            if mag <= 0:                               # speeds=0: 실제 정지 확정
                # nm 슬라이딩 윈도우: 정지는 nm=0으로 기록
                if tid not in self._nm_history:
                    self._nm_history[tid] = collections.deque(maxlen=self._NM_WIN)
                self._nm_history[tid].append(0.0)
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

            # 슬라이딩 중앙값으로 norm_speed_ratio 계산용 집계
            _med_nm = float(np.median(self._nm_history[tid]))
            norm_mags.append(_med_nm)                  # 속도 목록에 중앙값 추가

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
                # 재연결 후 혼잡 셀에 진입하는 신규 tid는 체류 시간 소급 부여
                # cell_dwell_ema는 ID 독립 → 재연결 이후에도 이전 혼잡도 보존
                _ema_at_cell = (float(self._cell_dwell_ema[_cr, _cc])
                                if self._cell_dwell_ema is not None else 0.0)
                if _ema_at_cell > 0.3:                 # 이전에 혼잡했던 셀 → 즉시 체류 인정
                    self._dwell_state[_tid] = (_cr, _cc, frame_num - self._dwell_thr_frames)
                else:
                    self._dwell_state[_tid] = (_cr, _cc, frame_num)

        # 이번 프레임에 없는 tid 체류 기록 삭제 (메모리 누수 방지)
        for _old in list(self._dwell_state.keys()):
            if _old not in speed_known_tids:
                del self._dwell_state[_old]

        # ── flow_occupancy / dwell_cell_ratio ────────────────────────────
        # 기준: valid_cell_count (flow_map이 실제 학습한 셀 수)
        # 도로 크기(2차선/4차선)에 자동 적응
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

        # dwell_cell_ratio: 체류 셀 / 현재 점유 셀 (플로우맵 크기 무관)
        # "점유된 셀 중 얼마나 많은 셀이 체류 상태인가" → 차량이 멈춘 비율
        # 분모를 valid_cell_count(전체 학습 셀)로 나누면 플로우맵이 넓어질수록 희석됨
        # occupied_cells 기준으로 바꾸면 플로우맵 크기와 무관하게 정체 강도 반영
        dwell_cell_ratio = float(np.clip(
            len(_dwell_cells_set) / max(occupied_cells, 1), 0.0, 1.0
        ))

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
            # ── 고밀도 보정: 차량 수 대비 체류 EMA 높은 셀 비율 ─────────────
            # 실도로에서 차량 5~8대 → 5~8셀 / 유효셀 40~60개 = density 0.10~0.15
            # → density 기반 보정은 실제 정체를 과도하게 억제
            # 개선: 점유 셀 중 고강도(ema > 0.20) 비율로 판단
            #   ema > 0.20 = 약 5프레임 이상 연속 점유 (6fps 기준 ~0.8초)
            #   정체 차량: 수초~수분 체류 → 대부분 0.20 초과
            #   정상 통과: 1~2프레임 점유 → 대부분 0.20 미만
            _high_ema_count = sum(1 for e in _occupied_emas if e > 0.20)
            _cds_density = min(1.0, _high_ema_count / max(len(_occupied_emas), 1))  # 고강도 셀 비율
            # 최종 score = 강도 × (0.4 + 0.6 × 고강도비율)
            #   고강도 셀 없어도 intensity × 0.4 최소 반영
            #   대부분 고강도이면 intensity × 1.0 = 최대값
            cell_dwell_score = float(np.clip(
                _cds_intensity * (0.4 + 0.6 * _cds_density), 0.0, 1.0
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

        # ── nm_history 만료 처리: 이번 프레임에 없는 tid 제거 ────────
        for old_tid in list(self._nm_history.keys()):
            if old_tid not in speed_known_tids:            # 이번 프레임에 없는 차량
                del self._nm_history[old_tid]              # 윈도우 삭제 (메모리 누수 방지)


        # ── 디버그 출력 (30프레임마다) ───────────────────────────────
        if frame_num % 30 == 0:
            print(f"[FE] f={frame_num} known={speed_known_count} "
                  f"occ={flow_occupancy:.3f} cds={cell_dwell_score:.3f} "
                  f"dwell={dwell_cell_ratio:.3f} persist={cell_persistence:.3f} valid={valid_cell_count}")

        # ── feature 딕셔너리 조립 ────────────────────────────────────
        return {                                       # feature 벡터
            "norm_speed_ratio":      norm_speed_ratio,       # 속도 비율 (자기보정 baseline 기준)
            "flow_occupancy":        flow_occupancy,         # 차량 점유 셀 / 유효 셀 (순간 밀도)
            "cell_dwell_score":      cell_dwell_score,       # 셀 누적 점유 EMA 합 / 유효 셀 (핵심 정체 신호)
            "dwell_cell_ratio":      dwell_cell_ratio,       # 체류 셀 / 점유 셀 (실제로 멈춘 셀 비율)
            "cell_persistence":      cell_persistence,       # 30프레임 전·현재 점유 셀 Jaccard 유사도
            "known_vehicle_count":   speed_known_count,      # 궤적 확인된 차량 수 (저규모 가드용)
            "occupied_cell_count":   occupied_cells,         # 현재 점유 셀 수 (저규모 가드용)

            "rule_jam_score":        0.0,                    # jam_score (CJ 채움)
            "count_ref":             float(getattr(self.cfg, "count_ref", 8.0)),
            # occ_gate 정규화용 — compute_jam_score_fallback이 fixed 분모(0.30) 대신
            # count_ref/valid_cell_count 기반 밀도로 포화점을 조정할 수 있도록 전달
        }
