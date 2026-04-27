# 파일 경로: C:\final_pj\src\congestion_judge.py
# 역할: jam_score 계산(fallback) + 레벨 판정(SMOOTH/SLOW/JAM) + 히스테리시스
# 의존성: math (표준 라이브러리 — sqrt)


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
    """flow map 기반 — Cell Dwell EMA 누적값(cell_dwell_score)으로 jam_score를 계산한다.

    cell_dwell_score: 각 셀의 누적 점유 EMA 합 / valid_cell_count
      점유 중인 셀: ema += 0.05 × (1 - ema) — 서서히 1 수렴
      빈 셀:       ema *= 0.98              — 서서히 0 수렴
      30프레임 연속 점유 시 ema ≈ 0.78

    ID 변경 무관 — 셀 점유 여부만 봄
    정상 차량(빠른 통과): ema 낮게 유지 → cds 낮음
    정체 차량(같은 셀 오래 머묾): ema 서서히 누적 → cds 높음

    설계 목표 (smooth<0.30, slow<0.60, jam≥0.60):
      - 원활 (cds=0.05, occ=0.05, dwell=0.000, valid=75, known=3): jam≈0.027 → SMOOTH
      - 서행 (cds=0.45, occ=0.18, dwell=0.010, valid=75, known=12): jam≈0.399 → SLOW
      - 정체 (cds=0.65, occ=0.22, dwell=0.060, valid=75, known=18): jam≈0.604 → JAM
    ※ occ_gate 포화점: count_ref/valid_cell_count+0.04 ≈ 0.14 (valid=75 기준)
    ※ dwell=0(15f+ 체류 차량 없음)이면 core에서 정체 기여 없음 → 고속 통과 오탐 방지

    Args:
        x_t: feature 벡터 dict.

    Returns:
        jam_score (0.0~1.0).
    """
    import math

    # ── feature 추출 ──────────────────────────────────────────────────
    cds      = _clip(x_t.get("cell_dwell_score", 0.0), 0.0, 1.0)  # 셀 누적 점유 EMA (핵심 정체 신호)
    flow_occ = _clip(x_t.get("flow_occupancy",   0.0), 0.0, 1.0)  # 차량 점유 셀 / 유효 셀 (순간 밀도)
    persist  = _clip(x_t.get("cell_persistence", 0.0), 0.0, 1.0)  # 30프레임 전·현재 점유셀 Jaccard (체류 지속성)
    dwell    = _clip(x_t.get("dwell_cell_ratio", 0.0), 0.0, 1.0)  # 체류 셀 / 유효 셀 (15f+ 머문 셀 비율)

    known_cnt    = int(x_t.get("known_vehicle_count", 0))  # 궤적 확인된 차량 수
    occupied_cnt = int(x_t.get("occupied_cell_count", 0))  # 현재 점유 셀 수

    # ── 1) 저규모 가드: 차량·셀이 너무 적으면 체류 신호 억제 ─────────
    # 근거: 차량 1~2대만 있을 때 그 차량이 우연히 같은 셀에 오래 있으면
    #       cds가 높게 찍혀 JAM 오탐 — 실제로는 "한 대 서행" 수준
    # flow_occ 0.06 = 20×20 그리드 기준 약 2.4셀 점유 → 그 이하는 거의 빈 도로
    if known_cnt <= 2 or occupied_cnt <= 2 or flow_occ < 0.06:
        return _clip(0.08 * math.sqrt(flow_occ), 0.0, 0.10)  # 최대 0.10으로 제한

    # ── 2) 규모 게이트: 차량 수가 충분할수록 체류 신호를 신뢰 ──────
    # count_gate: 2대 이하=0.0, 12대=1.0 — 소수 차량 오탐 억제
    # occ_gate  : 절대 차량 수 기준 (count_ref 대비) — 플로우맵 크기 무관
    #   이전: flow_occ / threshold 기반 → valid_cell_count 증가 시 flow_occ 희석 → 낮아짐
    #   수정: known_cnt / count_ref 기반 → 플로우맵이 넓어져도 차량 수가 같으면 동일 점수
    # scale_gate: 두 게이트의 곱 (AND 조건)
    count_gate     = _clip((known_cnt - 2) / 10.0, 0.0, 1.0)  # 2대=0, 12대=1.0
    _count_ref_val = float(x_t.get("count_ref", 8.0))          # 기준 차량 수 (기본 8)
    occ_gate       = _clip((known_cnt - 2) / max(_count_ref_val, 1), 0.0, 1.0)
    scale_gate     = count_gate * occ_gate                     # 두 게이트의 곱

    # ── 3) 핵심 jam 계산 ─────────────────────────────────────────────
    # [설계 원칙]
    # cds   × 0.55: 교통 밀도 배경 신호 — 빠른 통과 차량도 셀을 점유하므로 단독 JAM 유발 불가
    # dwell × 0.50: 점유 셀 중 체류 비율 (occupied_cells 기준)
    #   이전 분모: valid_cell_count(~80) → dwell ≈ 0.03~0.10
    #   현재 분모: occupied_cells(~10)  → dwell ≈ 0~1 (의미 전환: 점유 셀의 체류 비율)
    #   분모가 줄어 값이 최대 10배 커졌으므로 가중치를 1.00→0.50으로 절반 하향
    #   극심한 정체(dwell≈0.9): 기여 0.45 / 서행(dwell≈0.3): 기여 0.15 — 구분 충분
    # persist × 0.20: Jaccard 지속성 보조
    core = (
        0.55 * cds                   # 교통 밀도 배경 신호
        + 0.20 * persist             # 점유 지속성 보조
        + 0.50 * dwell               # 체류 비율 (1.00→0.50: occupied_cells 분모 변경에 따른 보정)
    )

    # 기저 신호: 절대 차량 수 기준 (플로우맵 크기 무관)
    _base = 0.05 * math.sqrt(float(known_cnt) / max(_count_ref_val, 1))
    jam = core * scale_gate + _base                        # 규모 게이트 적용 + 기저 신호

    return _clip(jam, 0.0, 1.0)


# ======================================================================
# CongestionJudge — 레벨 판정 + 히스테리시스 관리
# ======================================================================

class CongestionJudge:
    """jam_score를 계산하고 SMOOTH/SLOW/JAM 레벨을 판정한다.

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

        # ── 초기 확정 구간 (학습 완료 직후) ──────────────────────────
        # 학습 직후 cell_dwell_ema·cell_persistence 신호가 0에서 누적되는 동안
        # 짧은 히스테리시스(initial_hysteresis_sec)로 현재 도로 상태를 빠르게 확정.
        # initial_confirm_sec 경과 후 정규 히스테리시스(congestion_hysteresis_sec)로 전환.
        _init_confirm_sec = getattr(cfg, "initial_confirm_sec", 5.0)
        _init_hys_sec     = getattr(cfg, "initial_hysteresis_sec", 2.0)
        self._init_confirm_frames: int = int(_init_confirm_sec * fps)   # 초기 확정 구간 길이 (프레임)
        self._init_hysteresis_frames: int = int(_init_hys_sec * fps)    # 초기 히스테리시스 (프레임)
        self._baseline_frame: int | None = None        # set_baseline() 호출 시 프레임 번호 저장

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
        self._baseline_frame = None                    # 초기 확정 구간 재시작 (카메라 전환 시에도 적용)

    # ── 기준선 설정 ──────────────────────────────────────────────────
    def set_baseline(self):
        """학습 완료 신호를 받아 EMA를 중립값(0.5)으로 초기화한다.

        EMA 초기값을 0.0이 아닌 0.5로 설정하는 이유:
          - 0.0 시작 시 원활 상황에서도 0.5까지 올라오는 데 수십 프레임 걸림
          - 0.5 시작 시 원활이면 즉시 차감되어 0.1~0.2로 내려가고,
            정체이면 즉시 증가하여 0.7~0.9로 올라감
          - 학습 직후 "중립 → 실제 상태" 방향으로 빠르게 수렴
        """
        self._baseline_set = True                      # 학습 완료 표시
        self._ema_jam = 0.0                            # EMA 0에서 시작 → 실제 도로 상태로 수렴
        self._last_jam_score = 0.0
        self._baseline_frame = None                    # apply_level() 첫 호출 시 프레임 번호 기록

    # ── 레벨 판정 ────────────────────────────────────────────────────
    def _classify(self, jam_score: float) -> str:
        """jam_score로 원시 레벨을 판정한다 (히스테리시스 미적용).

        Args:
            jam_score: 0.0~1.0.

        Returns:
            "SMOOTH", "SLOW", or "JAM".
        """
        smooth_thr = self.get_smooth_threshold()       # SMOOTH 임계값
        slow_thr = self._get_slow_threshold()          # SLOW 임계값

        if jam_score < smooth_thr:                     # SMOOTH 임계값 미만
            return "SMOOTH"                            # 원활
        if jam_score < slow_thr:                       # SLOW 임계값 미만
            return "SLOW"                              # 서행
        return "JAM"                             # 정체

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
        # 1. 방향 판단: 악화(상승)이면 alpha_up, 호전(하강)이면 alpha_down
        if jam >= self._last_jam_score:  # 현재 순간값이 EMA보다 높으면 악화 방향
            alpha = self._alpha_up       # 빠르게 반응 (기본 0.70 — 정체 진입 즉시 감지)
        else:
            alpha = self._alpha_down     # 천천히 반응 (기본 0.04 — 순간 개선에 흔들리지 않음)

        # 2. EMA 갱신: new_ema = α × raw + (1-α) × prev_ema
        self._last_jam_score = alpha * jam + (1.0 - alpha) * self._last_jam_score
        self._ema_jam = self._last_jam_score           # get_jam_score() 반환값 동기화

        # 3. 초기 확정 구간 판단 — 학습 완료 후 initial_confirm_sec 동안 짧은 히스테리시스 사용
        if self._baseline_frame is None:               # 첫 apply_level() 호출 → 기준 프레임 기록
            self._baseline_frame = frame_num
        _elapsed = frame_num - self._baseline_frame    # 학습 완료 후 경과 프레임
        if _elapsed < self._init_confirm_frames:       # 초기 확정 구간 내
            self._hysteresis_frames = self._init_hysteresis_frames  # 짧은 히스테리시스 적용
        else:                                          # 초기 확정 구간 종료
            self._hysteresis_frames = int(             # 정규 히스테리시스로 전환
                self.cfg.congestion_hysteresis_sec * self.fps
            )

        # 4. 레벨 판정 (SMOOTH / SLOW / JAM)
        raw_level = self._classify(self._last_jam_score)
        level = self._apply_hysteresis(raw_level)      # 히스테리시스 적용 후 최종 레벨

        # 4. 정체 지속 시간 추적
        if level in ("SLOW", "JAM"):
            if self._congestion_start_frame is None:
                self._congestion_start_frame = frame_num
        else:
            self._congestion_start_frame = None

        return level, self._last_jam_score

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
            "SMOOTH", "SLOW", or "JAM".
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
        """현재 정체(SLOW/JAM) 지속 시간(초)을 반환한다.

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
