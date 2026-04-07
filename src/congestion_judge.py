# 파일 경로: C:\final_pj\src\congestion_judge.py
# 역할: jam_score 계산(fallback) + 레벨 판정(SMOOTH/SLOW/CONGESTED) + 히스테리시스
# 의존성: math (표준 라이브러리 — sqrt)

import math                                            # sqrt — bbox_coverage 비선형 변환


# ======================================================================
# 모듈 수준 함수 — jam_score 계산
# ======================================================================

def _clip(value: float, lo: float, hi: float) -> float:
    """value를 [lo, hi] 범위로 클램프한다.

    Args:
        value: 입력값.
        lo: 하한.
        hi: 상한.

    Returns:
        클램프된 값.
    """
    if value < lo:                                     # 하한 미만이면
        return lo                                      # 하한 반환
    if value > hi:                                     # 상한 초과이면
        return hi                                      # 상한 반환
    return value                                       # 범위 내이면 그대로



def compute_jam_score_fallback(x_t: dict) -> float:
    """baseline 없이 서행비율·정지비율·밀도로 jam_score를 계산한다 (fallback 모드).

    핵심 지표: slow_ratio (0.06 ≤ nm < 0.15 차량 비율).
      기존 문제: norm_speed_ref=0.15 하드코딩 → 실제 고속도로 nm은 0.5~2.0이므로
      norm_speed_ratio가 항상 1.0으로 clip → speed_contribution이 항상 0.
      또한 stop_ratio(nm<0.06)만으로는 서행(nm=0.08~0.14) 감지 불가.
      slow_ratio는 정지와 정상 주행 사이의 "서행 구간"을 직접 카운트하여
      nm 기준값(norm_speed_ref) 의존 없이 서행을 감지한다.

    설계 목표 (slow_upper_nm=1.0 기준 — 20~30 km/h 서행 포착):
      - 원활  (slow=0,    stop=0,    cell_occ=0.04): jam ≈ 0.08 → SMOOTH
      - 서행  (slow=0.80, stop=0.10, cell_occ=0.08): jam ≈ 0.39 → SLOW
      - 정체  (slow=1.0,  stop=0.70, cell_occ=0.12): jam ≈ 0.61 → CONGESTED
      - 극심  (slow=1.0,  stop=0.90, cell_occ=0.15): jam ≈ 0.67 → CONGESTED

    가중치 (slow_upper_nm=1.0에 맞춰 재조정):
      0.29 × slow_contribution — 서행 비율 (nm < 1.0). 1.0으로 올려도 CONGESTED 미초과
      0.25 × stop_contribution — 정지 비율 (nm < 0.06). 0.70↑시 서행→정체 전환 핵심
      0.20 × bbox_contribution — 셀 점유율 sqrt (cell occupancy, 원근 독립)
      0.08 × count_contribution — 차량 수 (count_ref=8 대비)

    유출(exit_rate_ratio) 미사용 이유:
      fallback 모드에서 count_ref=15 고정값 기반 exit_rate_ratio는
      실제 교통량과 무관하게 outflow_contribution을 과대 산출함 → 신뢰 불가.

    norm_speed_ratio 미사용 이유:
      fallback baseline의 norm_speed_ref=0.15는 임의 고정값.
      실제 고속도로 nm은 0.5~2.0 → ratio가 항상 1.0으로 clip되어
      speed_contribution이 0으로 무효화됨. slow_ratio가 직접 대체.

    Args:
        x_t: 8차원 feature 벡터 dict.

    Returns:
        jam_score (0.0~1.0).
    """
    # ── 서행 비율 기여 (0.06 ≤ nm < 0.50 차량 비율) ──────────────────
    slow_contribution = _clip(                         # 서행 비율 (0~1)
        x_t.get("slow_ratio", 0.0), 0.0, 1.0
    )

    # ── 정지 비율 기여 (nm < 0.06 차량 비율) ──────────────────────────
    stop_contribution = _clip(                         # 정지 비율 (0~1)
        x_t["stop_ratio"], 0.0, 1.0
    )

    # ── bbox 점유율 기여 — sqrt 비선형 변환 ──────────────────────────
    # sqrt(bbox_coverage): 낮은 coverage를 증폭, 높은 coverage는 완만하게 반영
    # 예) coverage=0.05 → sqrt=0.22(4.5×), coverage=0.35 → sqrt=0.59(1.7×)
    # count_ref 같은 임의 기준값 없이 coverage 자체에서 밀도 신호를 키움
    raw_bbox = _clip(                                  # coverage 원값 (0~1)
        x_t.get("bbox_coverage", x_t.get("density_score", 0.0)), 0.0, 1.0
    )
    bbox_contribution = math.sqrt(raw_bbox)            # sqrt 변환 (0~1 유지)

    # ── 차량 수 기여 (count_ref=8 대비 비율, 1.0 상한) ──────────────
    # 원거리 차량(B방향 상행)은 bbox가 작아 bbox_coverage만으로 밀도 반영 한계
    # count_ratio로 차량 수 자체를 직접 반영 — count_ref=8 (실탐지 최대 대수 기준)
    count_contribution = _clip(                        # 차량 수 비율 (0~1 상한)
        x_t.get("count_ratio", 0.0), 0.0, 1.0
    )

    # ── 가중 합산 (slow 29% + stop 25% + cell_occ 20% + count 8%) ──────
    # slow_upper_nm=1.0 기준: slow_ratio가 포화(1.0)되어도 CONGESTED 미초과
    # 원활 (slow=0, stop=0, cell_occ=0.04→sqrt=0.20, count=0.5):
    #   0 + 0 + 0.20×0.20 + 0.08×0.5 = 0.04+0.04 = 0.08 → SMOOTH
    # 서행 (slow=0.80, stop=0.10, cell_occ=0.08→sqrt=0.28, count=1.0):
    #   0.29×0.8 + 0.25×0.1 + 0.20×0.28 + 0.08 = 0.232+0.025+0.056+0.08 = 0.393 → SLOW
    # 정체 (slow=1.0, stop=0.70, cell_occ=0.12→sqrt=0.35, count=1.0):
    #   0.29 + 0.25×0.7 + 0.20×0.35 + 0.08 = 0.29+0.175+0.070+0.08 = 0.615 → CONGESTED
    jam = (0.29 * slow_contribution                    # 서행 가중 29% (1.0→재조정: slow_ratio 포화 방지)
           + 0.25 * stop_contribution                  # 정지 가중 25% (서행↔정체 전환 핵심)
           + 0.20 * bbox_contribution                  # 셀 점유율 20% (cell occupancy, 원근 독립)
           + 0.08 * count_contribution)                # 차량 수 비율 8% (count_ref=8)

    return _clip(jam, 0.0, 1.0)                        # [0, 1] 범위 클램프


# ======================================================================
# CongestionJudge — 레벨 판정 + 히스테리시스 관리
# ======================================================================

class CongestionJudge:
    """jam_score를 계산하고 SMOOTH/SLOW/CONGESTED 레벨을 판정한다.

    히스테리시스: congestion_hysteresis_sec × fps 프레임 동안
    기존 레벨을 유지해야 새 레벨로 전환된다.

    Parameters
    ----------
    cfg : DetectorConfig
        smooth_jam_threshold, slow_jam_threshold, congestion_hysteresis_sec 등.
    fps : float
        영상 FPS — 히스테리시스 프레임 수 계산에 사용.
    """

    def __init__(self, cfg, fps: float):
        """CongestionJudge 초기화.

        Args:
            cfg: DetectorConfig.
            fps: 영상 FPS.
        """
        self.cfg = cfg                                 # 설정 객체 저장
        self.fps = fps                                 # 영상 FPS 저장
        self._baseline_set: bool = False               # 학습 완료 여부 (set_baseline 호출 시 True)

        # ── 히스테리시스 상태 ────────────────────────────────────────
        self._current_level: str = "SMOOTH"            # 현재 확정 레벨
        self._pending_level: str = "SMOOTH"            # 전환 대기 레벨
        self._level_hold_frames: int = 0               # 대기 레벨 유지 프레임 수
        self._hysteresis_frames: int = int(            # 히스테리시스 프레임 수
            cfg.congestion_hysteresis_sec * fps         # 15초 × 30fps = 450프레임
        )

        # ── jam_score EMA 스무딩 (비대칭) ────────────────────────────
        # 악화(올라갈 때)는 alpha_up으로 빠르게, 호전(내려갈 때)는 alpha_down으로 느리게
        self._ema_jam: float = 0.0                     # EMA 누적값 (표시에 사용)
        self._alpha_up: float = getattr(               # 악화 방향 EMA 속도
            cfg, "jam_ema_alpha_up", 0.15
        )
        self._alpha_down: float = getattr(             # 호전 방향 EMA 속도
            cfg, "jam_ema_alpha_down", 0.04
        )

        # ── 정체 지속 시간 추적 ──────────────────────────────────────
        self._congestion_start_frame: int | None = None  # 정체 시작 프레임 (SMOOTH이면 None)
        self._last_jam_score: float = 0.0              # 마지막 EMA jam_score (표시용)

    # ── 상태 초기화 (카메라 전환 시 호출) ────────────────────────────
    def reset(self):
        """카메라 전환·재학습 시작 시 EMA와 히스테리시스 상태를 초기화한다.

        baseline은 유지 — 재학습 완료 후 set_baseline()으로 교체됨.
        """
        self._ema_jam = 0.0                            # EMA 초기화
        self._current_level = "SMOOTH"                 # 레벨 초기화
        self._pending_level = "SMOOTH"                 # 대기 레벨 초기화
        self._level_hold_frames = 0                    # 히스테리시스 카운터 초기화
        self._congestion_start_frame = None            # 정체 시작 프레임 초기화
        self._last_jam_score = 0.0                     # jam_score 초기화

    # ── 기준선 설정 ──────────────────────────────────────────────────
    def set_baseline(self, _baseline=None):
        """학습 완료 신호를 받아 EMA를 중립값(0.5)으로 초기화한다.

        EMA 초기값을 0.0이 아닌 0.5로 설정하는 이유:
          - 0.0 시작 시 원활 상황에서도 0.5까지 올라오는 데 수십 프레임 걸림
          - 0.5 시작 시 원활이면 즉시 차감되어 0.1~0.2로 내려가고,
            정체이면 즉시 증가하여 0.7~0.9로 올라감
          - 학습 직후 "중립 → 실제 상태" 방향으로 빠르게 수렴

        Args:
            baseline: 무시됨 (fallback 전용 — BaselineStats 불필요).
        """
        self._baseline_set = True                      # 학습 완료 표시
        self._ema_jam = 0.5                            # EMA 중립값으로 초기화 (학습 직후 빠른 수렴)

    # ── 레벨 판정 ────────────────────────────────────────────────────
    def _classify(self, jam_score: float) -> str:
        """jam_score로 원시 레벨을 판정한다 (히스테리시스 미적용).

        Args:
            jam_score: 0.0~1.0.

        Returns:
            "SMOOTH", "SLOW", or "CONGESTED".
        """
        smooth_thr = self.get_smooth_threshold()       # SMOOTH 임계값
        slow_thr = self._get_slow_threshold()          # SLOW 임계값

        if jam_score < smooth_thr:                     # SMOOTH 임계값 미만
            return "SMOOTH"                            # 원활
        if jam_score < slow_thr:                       # SLOW 임계값 미만
            return "SLOW"                              # 서행
        return "CONGESTED"                             # 정체

    # ── 임계값 접근자 ─────────────────────────────────────────────────
    def get_smooth_threshold(self) -> float:
        """SMOOTH 판정 임계값을 반환한다 (기본 0.30).

        Returns:
            smooth_jam_threshold (LCS 보정 없음 — fallback 전용).
        """
        return self.cfg.smooth_jam_threshold           # 고정 임계값 반환

    def _get_slow_threshold(self) -> float:
        """SLOW 판정 임계값을 반환한다 (기본 0.60).

        Returns:
            slow_jam_threshold (LCS 보정 없음 — fallback 전용).
        """
        return self.cfg.slow_jam_threshold             # 고정 임계값 반환

    # ── 히스테리시스 적용 ────────────────────────────────────────────
    def _apply_hysteresis(self, raw_level: str) -> str:
        """원시 레벨과 현재 레벨이 다르면 히스테리시스 프레임만큼 유지 후 전환한다.

        Args:
            raw_level: _classify()가 반환한 원시 레벨.

        Returns:
            히스테리시스 적용 후 최종 레벨.
        """
        if raw_level == self._current_level:           # 레벨 변화 없음
            self._pending_level = raw_level            # 대기 레벨 리셋
            self._level_hold_frames = 0                # 카운터 리셋
            return self._current_level                 # 현재 레벨 유지

        # ── 레벨이 달라진 경우 ───────────────────────────────────────
        if raw_level == self._pending_level:           # 이전 대기 레벨과 동일
            self._level_hold_frames += 1               # 유지 카운터 증가
        else:                                          # 대기 레벨이 또 바뀜
            self._pending_level = raw_level            # 새 대기 레벨로 교체
            self._level_hold_frames = 1                # 카운터 1부터 시작

        if self._level_hold_frames >= self._hysteresis_frames:  # 유지 시간 충족
            self._current_level = self._pending_level  # 레벨 전환 확정
            self._level_hold_frames = 0                # 카운터 리셋

        return self._current_level                     # 현재(또는 유지 중) 레벨

    # ── Phase 2 지원: jam 계산만 수행 ────────────────────────────────
    def compute_jam(self, x_t: dict) -> float:
        """x_t로부터 rule_jam_score를 계산하고 x_t에 역주입한다.

        update()를 분리한 것. Phase 2에서 GRU 블렌딩 전 rule_jam을 얻을 때 사용.

        Args:
            x_t: 7차원 feature 벡터 dict (rule_jam_score 키가 채워짐).

        Returns:
            rule_jam_score (0.0~1.0).
        """
        jam = compute_jam_score_fallback(x_t)          # fallback 모드 (항상)
        x_t["rule_jam_score"] = jam                    # feature 벡터에 역주입 (GRU 입력용)
        return jam                                     # rule_jam_score 반환

    # ── Phase 2 지원: 레벨 판정 + 히스테리시스만 수행 ──────────────────
    def apply_level(self, jam: float, frame_num: int) -> tuple:
        """jam_score에 비대칭 EMA를 적용한 뒤 레벨을 판정하고 히스테리시스를 적용한다.

        비대칭 EMA:
          - 악화(raw_jam > ema_jam): alpha_up으로 빠르게 반응 (정체 신속 감지)
          - 호전(raw_jam < ema_jam): alpha_down으로 느리게 반응 (순간 개선에 흔들리지 않음)

        Args:
            jam: 순간 jam_score (0.0~1.0). rule_jam 또는 blended_jam.
            frame_num: 현재 프레임 번호 (정체 지속 시간 추적용).

        Returns:
            (level: str, ema_jam: float) 튜플. ema_jam이 표시·판정에 사용됨.
        """
        # ── 비대칭 EMA 적용 ──────────────────────────────────────────
        if jam >= self._ema_jam:                       # 악화 방향 (올라갈 때)
            alpha = self._alpha_up                     # 빠른 반응 (0.10)
        else:                                          # 호전 방향 (내려갈 때)
            alpha = self._alpha_down                   # 느린 반응 (0.04)
        self._ema_jam = alpha * jam + (1.0 - alpha) * self._ema_jam  # EMA 갱신
        ema_jam = self._ema_jam                        # 스무딩된 jam_score

        self._last_jam_score = ema_jam                 # EMA jam_score 저장 (표시용)

        raw_level = self._classify(ema_jam)            # EMA 기반 원시 레벨 판정
        level = self._apply_hysteresis(raw_level)      # 히스테리시스 적용

        if level in ("SLOW", "CONGESTED"):             # 정체 상태이면
            if self._congestion_start_frame is None:   # 처음 진입
                self._congestion_start_frame = frame_num  # 시작 프레임 기록
        else:                                          # SMOOTH이면
            self._congestion_start_frame = None        # 초기화

        return level, ema_jam                          # (레벨, EMA jam) 반환

    # ── 메인 갱신 ────────────────────────────────────────────────────
    def update(self, x_t: dict, frame_num: int) -> tuple:
        """feature 벡터를 받아 jam_score를 계산하고 레벨을 판정한다.

        Args:
            x_t: 7차원 feature 벡터 dict.
            frame_num: 현재 프레임 번호.

        Returns:
            (level: str, jam_score: float) 튜플.
        """
        jam = self.compute_jam(x_t)                    # rule_jam 계산 + x_t 역주입
        return self.apply_level(jam, frame_num)        # 레벨 판정 + 히스테리시스 적용

    # ── 조회 메서드 ──────────────────────────────────────────────────
    def get_level(self) -> str:
        """현재 확정 레벨을 반환한다.

        Returns:
            "SMOOTH", "SLOW", or "CONGESTED".
        """
        return self._current_level                     # 히스테리시스 적용된 레벨

    def get_jam_score(self) -> float:
        """마지막 jam_score를 반환한다.

        Returns:
            0.0~1.0.
        """
        return self._last_jam_score                    # 마지막 update() 결과

    def get_duration_sec(self, frame_num: int,
                         fps: float) -> float:
        """현재 정체(SLOW/CONGESTED) 지속 시간(초)을 반환한다.

        Args:
            frame_num: 현재 프레임 번호.
            fps: 영상 FPS.

        Returns:
            지속 시간(초). SMOOTH이면 0.0.
        """
        if self._congestion_start_frame is None:       # 정체 아님
            return 0.0                                 # 0초
        elapsed = frame_num - self._congestion_start_frame  # 경과 프레임
        return max(0.0, elapsed / fps)                 # 프레임 → 초 변환
