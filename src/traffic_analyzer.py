# 파일 경로: C:\final_pj\src\traffic_analyzer.py
# 역할: ByteTrack 추적 결과(tracks·speeds)를 받아
#        정체 레벨(SMOOTH/SLOW/CONGESTED), 밀도맵, KPI를 산출하는 정체 탐지 모듈.
#        Phase 1: 절대 km/h 대신 baseline 대비 비율(normalized_mag) 기반.
#        cv2·torch에 의존하지 않으며 numpy만 사용한다.

import numpy as np                                    # 수치 계산 전용

from feature_extractor import FeatureExtractor        # 7차원 feature 벡터 계산기
from congestion_judge import CongestionJudge          # jam_score 계산 + 레벨 판정


# ======================================================================
# TrafficAnalyzer — 밀도·속도·정체 레벨 판정 (Phase 1 래퍼)
# ======================================================================

class TrafficAnalyzer:
    """매 프레임 tracks+speeds를 받아 정체 상태를 관리한다.

    기존 public 인터페이스(update, get_*) 시그니처·반환 타입을 유지하면서
    내부를 FeatureExtractor + CongestionJudge 조합으로 전면 교체.

    Parameters
    ----------
    cfg : DetectorConfig
        grid_size, stop_mag_threshold, congestion_hysteresis_sec 등.
    frame_w, frame_h : int
        영상 프레임 너비·높이 (픽셀). 밀도맵 셀 크기 계산에 사용.
    fps : float
        영상 FPS.
    flow_map : FlowMap or None
        FlowMap 객체 — bbox_coverage 셀 수 fallback용.
    congestion_judge : CongestionJudge or None
        외부에서 생성된 CongestionJudge. None이면 내부 생성.
    """

    # ── 정체 레벨 상수 ──────────────────────────────────────────────
    SMOOTH = "SMOOTH"                                 # 원활 (LOS A~B)
    SLOW = "SLOW"                                     # 서행 (LOS C~D)
    CONGESTED = "CONGESTED"                           # 정체 (LOS E~F)

    def __init__(self, cfg, frame_w: int, frame_h: int, fps: float,
                 flow_map=None, congestion_judge=None, gru_module=None):
        self.cfg = cfg                                # 설정 객체 저장
        self.frame_w = frame_w                        # 프레임 너비 (픽셀)
        self.frame_h = frame_h                        # 프레임 높이 (픽셀)
        self.fps = fps                                # 영상 FPS
        self.flow_map = flow_map                      # FlowMap 참조 (Phase 2 확장용)
        self._gru_module = gru_module                 # GRUModule 참조 (Phase 2, None이면 Phase 1 모드)
        self._last_feature: dict | None = None        # 마지막 compute된 feature 벡터 (detector용)

        self.grid_size = cfg.grid_size                # 그리드 행/열 수 (기본 15)
        self.cell_w = frame_w / self.grid_size        # 셀 너비 (px)
        self.cell_h = frame_h / self.grid_size        # 셀 높이 (px)

        # ── CongestionJudge: 외부 주입 또는 내부 생성 ─────────────
        if congestion_judge is not None:              # 외부에서 전달된 경우
            self.congestion_judge = congestion_judge   # 그대로 사용
        else:                                         # None이면 내부 생성
            self.congestion_judge = CongestionJudge(cfg, fps)  # 자체 생성

        self.feature_extractor: FeatureExtractor | None = None  # set_state() 후 초기화

        # ── 내부 상태 ─────────────────────────────────────────────
        self._density_map = np.zeros(                 # 15×15 밀도맵 (차량 수)
            (self.grid_size, self.grid_size), dtype=np.float64
        )
        self._vehicle_count = 0                       # 현재 프레임 차량 수
        self._last_frame_num = 0                      # 마지막 update 프레임 번호
        self._last_norm_speed_ratio = 0.0             # 마지막 속도 비율 (0~1)
        self._last_affected_count = 0                 # 마지막 저속(정지) 차량 수
        self._last_rule_jam: float = 0.0              # 마지막 rule 기반 jam_score (로그용)
        self._last_gru_score: float | None = None     # 마지막 GRU 예측값 (없으면 None)
        self._last_future_pred: list | None = None    # 마지막 predict_future() 결과 (자기회귀)
        self._last_direct_pred: list | None = None    # 마지막 predict_direct() 결과 (1·3·5분 직접 예측)
        self._future_pred_interval: int = max(        # predict_future 호출 주기 (프레임)
            cfg.gru_seq_len, 30                       # 최소 30프레임 — 자기회귀 연산 비용
        )

    # ── DetectorState 설정 (detector.py에서 호출) ──────────────────
    def set_state(self, state):
        """FeatureExtractor 초기화에 필요한 DetectorState를 설정한다.

        detector.py에서 TrafficAnalyzer 생성 후 state가 준비되면 호출.

        Args:
            state: DetectorState 객체.
        """
        self.feature_extractor = FeatureExtractor(    # FeatureExtractor 생성
            self.cfg,                                 # 설정 객체
            state,                                    # 런타임 상태
            fps=self.fps                              # FPS 전달 — dwell_threshold_sec 프레임 변환용
        )

    # ── 준비 완료 신호 (학습 완료 후 detector.py에서 호출) ──────────
    def set_baseline(self):
        """학습 완료 신호. FeatureExtractor와 CongestionJudge를 활성화한다."""
        if self.feature_extractor is not None:        # FE가 초기화된 경우
            self.feature_extractor.set_ready()         # FE 활성화
        self.congestion_judge.set_baseline()           # CJ EMA 초기화

    # ── 방향별 유효 셀 수 전달 (학습 완료 후 detector.py에서 호출) ──
    def set_valid_cell_count(self, n: int):
        """방향별 유효 셀 수를 FeatureExtractor에 전달한다.

        bbox_coverage를 셀 점유율로 계산할 때 방향별 road area 편향 제거용.
        detector.py가 _compute_direction_cell_counts() 후 호출한다.

        Args:
            n: 이 방향에 속하는 유효 flow_map 셀 수.
        """
        if self.feature_extractor is not None:        # FE 초기화된 경우
            self.feature_extractor.set_valid_cell_count(n)  # FE에 전달

    # ── 밀도맵 갱신 (내부) ─────────────────────────────────────────
    def _update_density_map(self, tracks: list):
        """tracks의 footpoint 위치를 기반으로 15×15 밀도맵을 갱신한다.

        Args:
            tracks: [{id, x1, y1, x2, y2, cx, cy, ...}, ...].
        """
        self._density_map = np.zeros(                 # 이전 맵 초기화
            (self.grid_size, self.grid_size), dtype=np.float64
        )
        for t in tracks:                              # 각 차량 순회
            fx = t.get("fx", (t["x1"] + t["x2"]) / 2)  # footpoint x (없으면 cx 대용)
            fy = t.get("fy", t["y2"])                 # footpoint y (없으면 y2)
            col = int(np.clip(fx / self.cell_w, 0, self.grid_size - 1))  # 열 인덱스 (클램프)
            row = int(np.clip(fy / self.cell_h, 0, self.grid_size - 1))  # 행 인덱스 (클램프)
            self._density_map[row, col] += 1          # 해당 셀 차량 수 +1

    # ── 공개 메서드: 프레임 갱신 ──────────────────────────────────
    def update(self, tracks: list, speeds: dict, frame_num: int) -> None:
        """매 프레임 호출. tracks·speeds를 받아 내부 상태를 갱신한다.

        Parameters
        ----------
        tracks : list[dict]
            [{"id":int, "x1","y1","x2","y2","cx","cy": float, ...}, ...]
        speeds : dict[int, float]
            {track_id: mag(픽셀 이동량)}. 0이면 정지/궤적 부족.
        frame_num : int
            현재 프레임 번호.
        """
        self._last_frame_num = frame_num              # 최신 프레임 번호 기록
        self._vehicle_count = len(tracks)             # 차량 수 기록

        # ── 1) 밀도맵 갱신 ────────────────────────────────────────
        self._update_density_map(tracks)              # footpoint 기반 밀도맵

        # ── 차량 없으면 jam_score 점진 감소 후 종료 ───────────────
        # return 대신 jam=0.0 으로 apply_level 호출 → alpha_down EMA로 서서히 감소
        if not tracks:                                # 현재 프레임 감지 차량 없음
            self._last_norm_speed_ratio = 0.0         # 속도 비율 초기화
            self._last_affected_count = 0             # 영향 차량 수 초기화
            self.congestion_judge.apply_level(0.0, frame_num)  # jam=0 → EMA 감소
            return

        # ── 2) feature 벡터 계산 → jam_score → 레벨 판정 ──────────
        if self.feature_extractor is None:            # set_state() 미호출 시
            return                                    # feature 계산 불가

        x_t = self.feature_extractor.compute(         # 7차원 feature 벡터 계산
            tracks, speeds, self.flow_map, frame_num
        )
        if x_t is None:                               # baseline 미설정 (학습 중)
            return                                    # 정체 판정 스킵

        # ── 속도 비율 및 영향 차량 수 기록 ─────────────────────────
        self._last_norm_speed_ratio = x_t["norm_speed_ratio"]  # 속도 비율 저장

        affected = 0                                  # 정지/저속 차량 카운터
        for t in tracks:                              # 각 차량 순회
            mag = speeds.get(t["id"], 0.0)            # mag 조회 (없으면 0)
            if mag < self.cfg.stop_mag_threshold:     # mag < 3.0이면 정지로 판단
                affected += 1                         # 저속 차량 수 증가
        self._last_affected_count = affected          # 결과 저장

        # ── 3) rule_jam 계산 (x_t["rule_jam_score"] 역주입 포함) ────
        rule_jam = self.congestion_judge.compute_jam(x_t)  # rule 기반 jam_score

        # ── 4) GRU 블렌딩 (Phase 2) ──────────────────────────────────
        # GRUModule이 없거나 predict()가 None이면 rule_jam 그대로 사용
        gru_score = None                              # GRU 예측값 초기값
        if self._gru_module is not None:              # Phase 2 모드이면
            self._gru_module.push(x_t)                # feature 벡터 버퍼에 추가
            gru_score = self._gru_module.predict()    # gru_score (버퍼 부족 시 None)

        if gru_score is not None:                     # GRU 예측 가능하면 (pretrain 완료)
            blend = self.cfg.gru_blend_ratio          # 블렌딩 비율 (기본 0.40)
            final_jam = (1.0 - blend) * rule_jam + blend * gru_score  # 가중 평균
        else:                                         # GRU 미훈련·버퍼 부족 → Phase 1 모드
            final_jam = rule_jam                      # rule_jam 그대로 사용

        self._last_rule_jam = rule_jam                # rule_jam 저장 (로그용)
        self._last_gru_score = gru_score              # gru_score 저장 (로그용, None 허용)
        if frame_num % 30 == 0:                       # 30프레임마다 raw_jam 디버그
            print(f"[CJ] f={frame_num} rule={rule_jam:.3f} ema={self.congestion_judge._ema_jam:.3f}")

        # ── 5) 레벨 판정 (히스테리시스 포함, final_jam 기준) ─────────
        level, jam = self.congestion_judge.apply_level(final_jam, frame_num)

        # ── 6) 미래 예측 (매 _future_pred_interval 프레임) ───────────────
        if (self._gru_module is not None
                and frame_num % self._future_pred_interval == 0):
            # 자기회귀 롤아웃 (단기 — 수초)
            pred = self._gru_module.predict_future()
            if pred is not None:
                self._last_future_pred = pred

            # Direct 예측 (장기 — 1·3·5분 후)
            direct = self._gru_module.predict_direct()
            if direct is not None:                    # 학습 완료 후에만 결과 있음
                self._last_direct_pred = direct

        # ── feature 저장 (detector.py에서 GRU online_step용 레벨 확인에 사용) ─
        self._last_feature = x_t                      # 마지막 feature 벡터 저장

    # ── 공개 메서드: 조회 ─────────────────────────────────────────
    def get_last_feature(self) -> "dict | None":
        """마지막 update()에서 계산된 feature 벡터를 반환한다.

        detector.py에서 GRU online_step 레이블 판단에 사용.
        baseline 미설정(학습 중)이면 None.

        Returns:
            feature dict or None.
        """
        return self._last_feature                     # 마지막 feature 벡터 반환

    def get_density_map(self) -> np.ndarray:
        """15×15 그리드 셀별 차량 밀도(대/셀)를 반환한다."""
        return self._density_map.copy()               # 복사본 반환 (외부 변경 방지)

    def get_avg_speed(self) -> float:
        """평균 속도를 반환한다 (norm_speed_ratio × 100, 상대값).

        절대 km/h가 아닌 baseline 대비 비율 × 100.
        100이면 학습 구간 정상 속도와 동일, 50이면 절반.
        """
        return self._last_norm_speed_ratio * 100.0    # 비율 → 0~100 스케일

    def get_congestion_level(self) -> str:
        """현재 정체 레벨 문자열("SMOOTH"/"SLOW"/"CONGESTED")을 반환한다."""
        return self.congestion_judge.get_level()       # CJ에서 히스테리시스 적용된 레벨

    def get_jam_score(self) -> float:
        """현재 jam_score(0.0~1.0)를 반환한다."""
        return self.congestion_judge.get_jam_score()   # CJ에서 마지막 계산 값

    def get_rule_jam_score(self) -> float:
        """마지막 rule 기반 jam_score(GRU 블렌딩 전)를 반환한다."""
        return self._last_rule_jam                     # rule_jam (로그·진단용)

    def get_gru_score(self) -> float | None:
        """마지막 GRU 예측값을 반환한다. GRU 미사용 또는 warmup 중이면 None."""
        return self._last_gru_score                    # gru_score (로그·진단용)

    def get_future_prediction(self) -> list | None:
        """마지막 predict_future() 결과를 반환한다.

        Returns:
            [{"step": int, "p_smooth": float, "p_slow": float,
              "p_congested": float, "gru_score": float}, ...]
            GRU 미훈련·버퍼 부족이면 None.
        """
        return self._last_future_pred                  # 자기회귀 예측 결과 (웹 API용)

    def get_direct_prediction(self) -> list | None:
        """1·3·5분 후 직접 예측 결과를 반환한다.

        Returns:
            [{"horizon_min": 1, "predicted_level": "SLOW",
              "p_smooth": 0.1, "p_slow": 0.6, "p_congested": 0.3,
              "confidence": 0.6}, ...]
            학습 미완료이면 None.
        """
        return self._last_direct_pred                  # direct 예측 결과 (웹 API·화면 표시용)

    def get_volume(self) -> float:
        """교통량(대/시)을 추정한다.

        현재 프레임의 차량 수 × (fps × 3600 / velocity_window) 단순 추정.
        카메라 캘리브레이션이 없으므로 프레임 기반 근사.
        """
        if self._vehicle_count == 0:                  # 차량 없으면
            return 0.0                                # 교통량 0
        # velocity_window 프레임 동안 화면에 있는 차량 수 기반 추정
        vw = self.cfg.velocity_window                 # 속도 계산 프레임 간격 (15)
        if vw <= 0 or self.fps <= 0:                  # 0-division 방어
            return 0.0                                # 추정 불가
        # 초당 프레임 수 / vw = 초당 갱신 횟수, × 3600 = 시간당, × 차량 수
        cycles_per_hour = (self.fps / vw) * 3600.0    # 시간당 갱신 주기 수
        return self._vehicle_count * cycles_per_hour / self.fps  # 대/시 근사

    def get_occupancy(self) -> float:
        """점유율(%)을 반환한다. 차량이 있는 셀 / 전체 셀 × 100."""
        total_cells = self.grid_size * self.grid_size # 전체 셀 수 (225)
        occupied = int(np.count_nonzero(self._density_map))  # 차량 있는 셀 수
        return (occupied / total_cells) * 100.0       # 백분율 변환

    def get_duration_sec(self) -> float:
        """현재 정체 레벨(SLOW/CONGESTED) 지속 시간(초)을 반환한다.
        SMOOTH 상태이면 0.0."""
        return self.congestion_judge.get_duration_sec(  # CJ에 위임
            self._last_frame_num, self.fps
        )

    def get_affected_vehicles(self) -> int:
        """mag < stop_mag_threshold(기본 3.0) 미만의 정지/저속 차량 수를 반환한다."""
        return self._last_affected_count              # 마지막 update 기준 값


# ======================================================================
# CongestionPredictor — 단기 정체 예측 + 예상 회복 시간
# ======================================================================

class CongestionPredictor:
    """과거 평균 속도 히스토리에 선형 회귀(polyfit)를 적용하여
    정체 추세(WORSENING/IMPROVING/STABLE)와 예상 회복 시간(분)을 산출한다.

    Parameters
    ----------
    cfg : DetectorConfig
        prediction_history_window, prediction_horizon, free_flow_speed 사용.
    fps : float
        영상 FPS.
    """

    def __init__(self, cfg, fps: float):
        self.cfg = cfg                                            # 설정 저장
        self.fps = fps                                            # FPS 저장
        self._history: list[float] = []                           # 속도 히스토리
        self._max_len = cfg.prediction_history_window             # 최대 길이

    def update(self, avg_speed: float) -> None:
        """속도 히스토리에 최신 평균 속도를 추가한다."""
        self._history.append(avg_speed)                           # 히스토리에 추가
        if len(self._history) > self._max_len:                    # 최대 길이 초과 시
            self._history.pop(0)                                  # 가장 오래된 값 제거

    def predict(self) -> dict:
        """선형 회귀 기울기로 추세를 판단하고 회복 예상 시간을 반환한다.

        Returns
        -------
        dict
            {"trend": "WORSENING"/"IMPROVING"/"STABLE",
             "recovery_min": float (회복까지 예상 분, 음수이면 예측 불가)}
        """
        if len(self._history) < 2:                                # 데이터 부족
            return {"trend": "STABLE", "recovery_min": -1.0}      # 판단 불가

        x = np.arange(len(self._history), dtype=np.float64)       # 시간 축 (0,1,2,...)
        y = np.array(self._history, dtype=np.float64)             # 속도 축

        # 1차 다항식 (y = slope*x + intercept) 피팅
        coeffs = np.polyfit(x, y, 1)                              # [slope, intercept]
        slope = coeffs[0]                                         # 기울기 (km/h per frame)

        # 기울기 부호로 추세 결정 (임계값 0.01로 미세 변동 무시)
        if slope < -0.01:                                         # 속도 하락 → 악화
            trend = "WORSENING"
        elif slope > 0.01:                                        # 속도 상승 → 개선
            trend = "IMPROVING"
        else:                                                     # 거의 변화 없음
            trend = "STABLE"

        # 회복 예상 시간: 현재 속도가 free_flow × 0.7에 도달하기까지 (분)
        target_speed = self.cfg.free_flow_speed * 0.7             # SMOOTH 하한
        current_speed = self._history[-1]                         # 현재(마지막) 속도
        recovery_min = -1.0                                       # 기본값: 예측 불가

        if slope > 0.01 and current_speed < target_speed:         # 개선 중이고 아직 미달
            frames_to_target = (target_speed - current_speed) / slope  # 필요 프레임 수
            recovery_min = (frames_to_target / self.fps) / 60.0   # 프레임 → 분 변환

        return {"trend": trend, "recovery_min": recovery_min}     # 결과 반환
