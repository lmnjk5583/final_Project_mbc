# 파일 경로: C:\final_pj\src\congestion_judge.py
# 역할: jam_score 계산(정상/fallback) + 레벨 판정(SMOOTH/SLOW/CONGESTED) + 히스테리시스
# 의존성: baseline_stats(로컬)

from baseline_stats import BaselineStats               # 학습 기준선 통계 데이터 클래스


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


def compute_jam_score(x_t: dict, lcs: float, cfg) -> float:
    """baseline이 있을 때 jam_score를 계산한다 (정상 운영 모드).

    설계 목표:
      - 원활(속도 정상, 정지 없음): 0.0~0.15
      - 서행(속도 60~80%, 정지 20~40%): 0.30~0.55
      - 극심한 정체(속도 거의 0, 정지 70%+): 0.85~1.0

    가중치: 0.55×speed + 0.35×stop + 0.10×congestion_bonus
    congestion_bonus: 속도 저하 + 정지 동시 발생 시 추가 가산 (곱셈 항)

    Args:
        x_t: 7차원 feature 벡터 dict.
        lcs: Learning Congestion Score (0~1).
        cfg: DetectorConfig — smooth_jam_threshold, slow_jam_threshold 등.

    Returns:
        jam_score (0.0~1.0).
    """
    # ── 각 지표의 기여도 계산 ────────────────────────────────────────
    speed_score = _clip(                               # 속도 기여: 1-ratio (느릴수록 높음)
        1.0 - x_t["norm_speed_ratio"], 0.0, 1.0
    )
    stop_score = _clip(                                # 정지 기여: 정지 비율 그대로
        x_t["stop_ratio"], 0.0, 1.0                   # 정지 차량 많을수록 높음
    )
    # 복합 정체 가산항: 속도 저하와 정지가 동시에 높을 때 추가 점수
    # 예) speed_score=0.8, stop_score=0.7 → bonus=0.56×0.10=0.056 추가
    # 원활 시: speed_score≈0, stop_score≈0 → 기여 거의 0
    congestion_synergy = speed_score * stop_score      # 두 지표 곱 (동시 악화 시 증폭)

    # ── 가중 합산 ─────────────────────────────────────────────────────
    jam = (0.55 * speed_score                          # 속도 가중 55%
           + 0.35 * stop_score                         # 정지 가중 35%
           + 0.10 * congestion_synergy)                # 복합 정체 가산 10%

    # ── 원활 보너스 차감 (원활할수록 더 많이 깎임) ────────────────────
    bonus = 0.0                                        # 보너스 초기화
    if x_t["exit_rate_ratio"] > 1.3:                   # 유출이 기준의 1.3배 초과
        bonus += 0.06                                  # 유출 원활 보너스
    if x_t["stop_ratio"] < 0.05:                       # 정지 차량 5% 미만
        bonus += 0.06                                  # 거의 정지 없음 보너스
    if x_t["norm_speed_ratio"] > 0.9:                  # 속도가 기준의 90% 초과 (원활)
        bonus += 0.08                                  # 원활 주행 보너스 (강화)

    return _clip(jam - bonus, 0.0, 1.0)                # 보너스 차감 후 [0, 1] 클램프


def compute_jam_score_fallback(x_t: dict) -> float:
    """baseline 없이 서행비율·정지비율·밀도로 jam_score를 계산한다 (fallback 모드).

    핵심 지표: slow_ratio (0.06 ≤ nm < 0.15 차량 비율).
      기존 문제: norm_speed_ref=0.15 하드코딩 → 실제 고속도로 nm은 0.5~2.0이므로
      norm_speed_ratio가 항상 1.0으로 clip → speed_contribution이 항상 0.
      또한 stop_ratio(nm<0.06)만으로는 서행(nm=0.08~0.14) 감지 불가.
      slow_ratio는 정지와 정상 주행 사이의 "서행 구간"을 직접 카운트하여
      nm 기준값(norm_speed_ref) 의존 없이 서행을 감지한다.

    설계 목표:
      - 원활  (slow≈0.02, stop≈0.01, density≈0.25):  jam ≈ 0.06 → SMOOTH
      - 서행  (slow≈0.60, stop≈0.05, density≈0.40):  jam ≈ 0.40 → SLOW
      - 정체  (slow≈0.20, stop≈0.70, density≈0.70):  jam ≈ 0.45 → SLOW~CONGESTED
      - 극심  (slow≈0.10, stop≈0.90, density≈0.80):  jam ≈ 0.48 → CONGESTED

    가중치:
      0.50 × slow_contribution    — 서행 비율: 서행 핵심 감지 (nm 0.06~0.15)
      0.30 × stop_contribution    — 정지 비율: 완전 정체 핵심 감지 (nm < 0.06)
      0.20 × density_contribution — 밀도: 보조 지표 (차량 많을수록 가산)

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

    # ── bbox 점유율 기여 (flow_map 유효 도로 면적 대비 bbox 면적 합) ───
    # 차선 수·차량 대수에 독립적 — 도로가 차량으로 얼마나 꽉 찼는지
    bbox_contribution = _clip(                         # bbox 점유율 (0~1)
        x_t.get("bbox_coverage", x_t.get("density_score", 0.0)), 0.0, 1.0
    )

    # ── 가중 합산 (slow 60% + stop 60% + bbox 25%, 합=1.45) ──────────
    # 가중치 합 > 1.0: 극심 정체 시 clip 전 1.0 초과 → 1.0으로 포화
    # 원활:  0.60×0.02 + 0.60×0.01 + 0.25×0.08 = 0.012+0.006+0.020 = 0.038
    # 서행:  0.60×0.60 + 0.60×0.05 + 0.25×0.20 = 0.360+0.030+0.050 = 0.440
    # 정체:  0.60×0.20 + 0.60×0.70 + 0.25×0.35 = 0.120+0.420+0.088 = 0.628
    # 극심:  0.60×0.95 + 0.60×0.90 + 0.25×0.40 = 0.570+0.540+0.100 = 1.210 → clip 1.0
    jam = (0.60 * slow_contribution                    # 서행 가중 60%
           + 0.60 * stop_contribution                  # 정지 가중 60%
           + 0.25 * bbox_contribution)                 # bbox 점유율 25% (보조)

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
        self.baseline: BaselineStats | None = None     # 학습 기준선 (초기 None)

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
    def set_baseline(self, baseline: BaselineStats):
        """학습 완료 후 BaselineStats를 설정하고 EMA를 중립값(0.5)으로 초기화한다.

        EMA 초기값을 0.0이 아닌 0.5로 설정하는 이유:
          - 0.0 시작 시 원활 상황에서도 0.5까지 올라오는 데 수십 프레임 걸림
          - 0.5 시작 시 원활이면 즉시 차감되어 0.1~0.2로 내려가고,
            정체이면 즉시 증가하여 0.7~0.9로 올라감
          - 학습 직후 "중립 → 실제 상태" 방향으로 빠르게 수렴

        Args:
            baseline: finalize_baseline()이 반환한 기준선 객체.
        """
        self.baseline = baseline                       # 기준선 저장
        self._ema_jam = 0.5                            # EMA 중립값으로 초기화 (학습 직후 빠른 수렴)

    # ── 레벨 판정 (LCS 보정 포함) ────────────────────────────────────
    def _classify(self, jam_score: float) -> str:
        """jam_score와 LCS로 원시 레벨을 판정한다 (히스테리시스 미적용).

        LCS가 높으면 threshold를 낮춰서 정체를 더 쉽게 판정한다.
        smooth: jam < smooth_thr × (1 - lcs×0.40)   최저 0.15
        slow:   jam < slow_thr   × (1 - lcs×0.30)   최저 0.38

        Args:
            jam_score: 0.0~1.0.

        Returns:
            "SMOOTH", "SLOW", or "CONGESTED".
        """
        smooth_thr = self.get_smooth_threshold()       # LCS 보정된 SMOOTH 임계값
        slow_thr = self._get_slow_threshold()          # LCS 보정된 SLOW 임계값

        if jam_score < smooth_thr:                     # SMOOTH 임계값 미만
            return "SMOOTH"                            # 원활
        if jam_score < slow_thr:                       # SLOW 임계값 미만
            return "SLOW"                              # 서행
        return "CONGESTED"                             # 정체

    # ── LCS 보정 임계값 접근자 ────────────────────────────────────────
    def get_smooth_threshold(self) -> float:
        """SMOOTH 판정 임계값을 반환한다.

        fallback 모드에서는 LCS 보정 미적용 — fallback 공식은 score 범위가
        정상 모드보다 보수적이므로 임계값을 낮추면 SLOW 오판 발생.
        정상 모드에서만 LCS로 임계값을 낮춰 정체 감지 민감도를 높인다.

        Returns:
            fallback: smooth_jam_threshold 그대로 (기본 0.30).
            정상:     smooth_jam_threshold × (1 - lcs × 0.40).
        """
        if self.baseline is not None and self.baseline.is_fallback:  # fallback 모드
            return self.cfg.smooth_jam_threshold               # LCS 보정 없음
        lcs = self.baseline.lcs if self.baseline else 0.0      # 정상 모드: LCS 적용
        return self.cfg.smooth_jam_threshold * (1.0 - lcs * 0.40)  # LCS 보정

    def _get_slow_threshold(self) -> float:
        """SLOW 판정 임계값을 반환한다.

        fallback 모드에서는 LCS 보정 미적용 (get_smooth_threshold 참고).

        Returns:
            fallback: slow_jam_threshold 그대로 (기본 0.55).
            정상:     slow_jam_threshold × (1 - lcs × 0.30).
        """
        if self.baseline is not None and self.baseline.is_fallback:  # fallback 모드
            return self.cfg.slow_jam_threshold                 # LCS 보정 없음
        lcs = self.baseline.lcs if self.baseline else 0.0      # 정상 모드: LCS 적용
        return self.cfg.slow_jam_threshold * (1.0 - lcs * 0.30)  # LCS 보정

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
        if self.baseline is None or self.baseline.is_fallback:  # 기준선 없음/fallback
            jam = compute_jam_score_fallback(x_t)      # fallback 모드 계산
        else:                                          # 정상 기준선 있음
            jam = compute_jam_score(                   # 정상 모드 계산
                x_t, self.baseline.lcs, self.cfg
            )
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
