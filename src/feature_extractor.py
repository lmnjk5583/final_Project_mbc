# 파일 경로: C:\final_pj\src\feature_extractor.py
# 역할: 매 프레임 tracks+speeds를 받아 feature 벡터를 산출한다.
# 의존성: numpy(서드파티)

import numpy as np                                     # clip, mean 등 수치 연산


# ======================================================================
# FeatureExtractor — feature 벡터 계산기
# ======================================================================

class FeatureExtractor:
    """매 프레임 tracks·speeds를 받아 정체 판정용 feature 벡터를 산출한다.

    feature 벡터:
    ┌─────┬───────────────────┬──────────────────────────────────────────────┐
    │ idx │ 이름              │ 계산식                                       │
    ├─────┼───────────────────┼──────────────────────────────────────────────┤
    │  0  │ norm_speed_ratio  │ median(upper_50%_nm) / self-calibrating ref │
    │  1  │ stop_ratio        │ (nm<0.06 차량) / speed_known_count          │
    │ 1.5 │ slow_ratio        │ (0.06≤nm<0.50 차량) / speed_known_count     │
    │  2  │ density_score     │ occupied_cells / valid_cell_count            │
    │  3  │ rule_jam_score    │ 0.0 (congestion_judge가 채워넣음)            │
    └─────┴───────────────────┴──────────────────────────────────────────────┘

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
        speed_known_count = 0                          # 궤적 확인된 차량 수 (신규 제외)
        for t in tracks:                               # 각 차량 순회
            raw_bbox_h = t["y2"] - t["y1"]            # 바운딩박스 높이 (픽셀)
            bbox_h = max(raw_bbox_h, min_bbox_h)       # 최솟값 클램프 — 원거리 소형 박스 과대평가 방지
            tid = t["id"]                              # 트랙 ID
            if tid not in speeds:                      # speeds에 없으면 신규 → 완전 제외
                continue
            mag = speeds[tid]                          # 속도 조회
            speed_known_count += 1                     # 궤적 확인 차량 수 증가
            if mag <= 0:                               # speeds=0: 실제 정지 확정
                stopped_count += 1                     # 정지 카운트 (nm 계산 없이)
                slow_count += 1                        # 정지는 서행의 부분집합 — 서행에도 포함
                continue
            nm = mag / bbox_h                          # normalized_mag (원근 보정)
            if nm_cy_k > 0:                            # cy 보정 활성화 시
                cy_ratio = t["cy"] / max(self.state.frame_h, 1)  # 0(상단/원거리)~1(하단/근거리)
                denom = 1.0 + nm_cy_k * (2.0 * cy_ratio - 1.0)  # 대칭 보정
                nm = nm / max(denom, 0.1)              # nm 보정
            norm_mags.append(nm)                       # 속도 목록에 추가
            if nm < norm_stop_thr:                     # nm < 0.06 → 저속 정지
                stopped_count += 1                     # 정지 카운트
                slow_count += 1                        # 정지는 서행의 부분집합 — 서행에도 포함
            elif nm < slow_upper_nm:                   # 0.06 ≤ nm < 0.50 → 서행 구간
                slow_count += 1                        # 서행 카운트

        # ── bbox_coverage: 셀 점유율 (cell occupancy) 방식 ──────────────
        cell_w = self.state.frame_w / self.cfg.grid_size   # 셀 너비 (픽셀)
        cell_h = self.state.frame_h / self.cfg.grid_size   # 셀 높이 (픽셀)

        # 유효 셀 수: 방향별 override > flow_map 실측 > 전체 그리드 순서로 사용
        if self._valid_cell_count_override is not None:    # 방향별 셀 수 주입됨
            valid_cell_count = self._valid_cell_count_override
        elif flow_map is not None and hasattr(flow_map, "count"):
            valid_cell_count = int(np.sum(flow_map.count > 0))  # 학습된 유효 셀 수
        else:
            valid_cell_count = self.cfg.grid_size * self.cfg.grid_size  # fallback: 전체

        # 차량 footpoint가 위치한 고유 셀 집합
        occupied_cells = len(set(
            (int(np.clip(t["cy"] / cell_h, 0, self.cfg.grid_size - 1)),   # 행 인덱스
             int(np.clip(t["cx"] / cell_w, 0, self.cfg.grid_size - 1)))   # 열 인덱스
            for t in tracks
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

        # ── stop_ratio: 궤적 확인된 차량 중 정지 비율 ─────────────────
        # 소표본 신뢰도 보정: 차량 5대 미만이면 stop/slow 기여를 제곱 감쇠로 축소
        # 제곱 감쇠 이유: 선형 보정(÷5)은 2대만 있어도 jam≈0.5 → SLOW 오탐 발생
        #   1대: (1/5)²=0.04, 2대: (2/5)²=0.16, 3대: 0.36, 4대: 0.64, 5대: 1.0
        _MIN_RELIABLE = 5                                  # 신뢰 가능 최소 차량 수 (3→5 강화)
        _raw_stop = stopped_count / max(speed_known_count, 1)  # 원시 stop_ratio
        if speed_known_count < _MIN_RELIABLE:              # 차량 수 부족 → 신뢰도 가중치 적용
            _reliability = (speed_known_count / _MIN_RELIABLE) ** 2  # 제곱 감쇠 (0.04~0.96)
            stop_ratio = _raw_stop * _reliability          # 소표본 기여 제한
        else:                                              # 차량 수 충분 → 그대로 사용
            stop_ratio = _raw_stop                         # 신뢰도 보정 불필요

        # ── slow_ratio: 궤적 확인된 차량 중 서행 비율 ─────────────────
        _raw_slow = slow_count / max(speed_known_count, 1) # 원시 slow_ratio
        if speed_known_count < _MIN_RELIABLE:              # 차량 수 부족 → 신뢰도 가중치 적용
            slow_ratio = _raw_slow * _reliability          # _reliability는 stop_ratio에서 이미 계산됨
        else:                                              # 차량 수 충분 → 그대로 사용
            slow_ratio = _raw_slow                         # 신뢰도 보정 불필요

        # ── feature 딕셔너리 조립 ────────────────────────────────────
        return {                                       # feature 벡터
            "norm_speed_ratio":   norm_speed_ratio,    # [0] 속도 비율 (자기보정 baseline 기준)
            "nm_baseline_valid":  _nm_baseline_valid,  # baseline 준비 여부
            "stop_ratio":         stop_ratio,          # [1] 정지 비율 (nm < 0.06)
            "slow_ratio":         slow_ratio,          # [1.5] 서행 비율 (0.06 ≤ nm < 0.50)
            "density_score":      density_score,       # [2] bbox_coverage 별칭 (하위 호환)
            "bbox_coverage":      bbox_coverage,       # [2] 도로 면적 대비 셀 점유율
            "rule_jam_score":     0.0,                 # [3] jam_score (CJ 채움)
        }
