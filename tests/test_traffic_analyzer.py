# 파일 경로: C:\final_pj\tests\test_traffic_analyzer.py
# 역할: TrafficAnalyzer + CongestionPredictor 단위 테스트 (GRU 제거 이후)
# 실행: pytest tests/test_traffic_analyzer.py -v

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from config import DetectorConfig
from traffic_analyzer import TrafficAnalyzer, CongestionPredictor


# ======================================================================
# 공통 Mock 클래스
# ======================================================================

class _MockState:
    """DetectorState 최소 구현 — FeatureExtractor가 참조하는 필드만 포함."""

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.first_seen_frame = {}
        self.entry_positions  = {}


# ======================================================================
# 공통 헬퍼 함수
# ======================================================================

@pytest.fixture
def cfg():
    """테스트 전용 DetectorConfig.

    congestion_hysteresis_sec=0.0 → 레벨 전환 즉시 (히스테리시스 OFF).
    """
    return DetectorConfig(
        grid_size=15,
        velocity_window=15,
        free_flow_speed=100.0,
        congestion_hysteresis_sec=0.0,     # 히스테리시스 OFF → 즉시 전환
        prediction_history_window=30,
        stop_mag_threshold=3.0,
        norm_stop_threshold=0.05,
        min_bbox_h=30.0,
        exit_rate_window=30,
        smooth_jam_threshold=0.30,
        slow_jam_threshold=0.60,
    )


def _make_analyzer(cfg,
                   frame_w: int = 1920, frame_h: int = 1080,
                   fps: float = 6.0) -> TrafficAnalyzer:
    """TrafficAnalyzer 인스턴스를 생성하고 set_state()까지 완료한다."""
    ta = TrafficAnalyzer(
        cfg=cfg,
        frame_w=frame_w,
        frame_h=frame_h,
        fps=fps,
    )
    state = _MockState(frame_w=frame_w, frame_h=frame_h)
    ta.set_state(state)
    return ta


def _make_tracks(n: int, cx_base: int = 100,
                 gap: int = 60, cy: int = 500,
                 bbox_h: int = 50) -> list:
    """n대의 더미 트랙 딕셔너리 리스트를 생성한다."""
    tracks = []
    half_h = bbox_h // 2
    for i in range(n):
        cx  = float(cx_base + i * gap)
        cy_f = float(cy)
        y2  = cy_f + half_h
        tracks.append({
            "id": i + 1,
            "x1": cx - 25.0, "y1": cy_f - half_h,
            "x2": cx + 25.0, "y2": y2,
            "cx": cx, "cy": cy_f,
            "fx": cx, "fy": y2,
        })
    return tracks


# ======================================================================
# TA-01 ~ TA-05: 기본 동작 검증
# ======================================================================

class TestBasicOperation:
    """TrafficAnalyzer 초기화·기본 입출력을 검증한다."""

    def test_ta01_no_baseline_level_smooth(self, cfg):
        """TA-01: baseline 미설정 상태에서 update() → level = 'SMOOTH' (초기값 유지).

        set_baseline() 미호출 시 FeatureExtractor._ready=False → compute()→None
        → 레벨 판정 스킵 → 초기값 'SMOOTH' 유지.
        """
        ta = _make_analyzer(cfg)
        tracks = _make_tracks(5)
        speeds = {t["id"]: 5.0 for t in tracks}
        ta.update(tracks, speeds, frame_num=1)
        assert ta.get_congestion_level() == "SMOOTH"

    def test_ta02_empty_tracks_level_smooth(self, cfg):
        """TA-02: set_baseline 후 빈 tracks → level = 'SMOOTH'.

        차량 없으면 jam=0.0 → EMA 점진 감소 → SMOOTH 유지.
        """
        ta = _make_analyzer(cfg)
        ta.set_baseline()
        ta.update(tracks=[], speeds={}, frame_num=1)
        assert ta.get_congestion_level() == "SMOOTH"

    def test_ta03_density_map_shape(self, cfg):
        """TA-03: get_density_map() 반환값은 (15, 15) ndarray."""
        ta = _make_analyzer(cfg)
        tracks = _make_tracks(3)
        speeds = {t["id"]: 5.0 for t in tracks}
        ta.update(tracks, speeds, frame_num=1)
        dm = ta.get_density_map()
        assert isinstance(dm, np.ndarray)
        assert dm.shape == (15, 15)

    def test_ta04_jam_score_range(self, cfg):
        """TA-04: get_jam_score() 반환값은 0.0~1.0 범위."""
        ta = _make_analyzer(cfg)
        ta.set_baseline()
        tracks = _make_tracks(5)
        speeds = {t["id"]: 5.0 for t in tracks}
        ta.update(tracks, speeds, frame_num=1)
        score = ta.get_jam_score()
        assert 0.0 <= score <= 1.0

    def test_ta05_multiple_updates_no_exception(self, cfg):
        """TA-05: update() 50회 반복 호출 시 예외가 발생하지 않아야 한다."""
        ta = _make_analyzer(cfg)
        ta.set_baseline()
        for f in range(1, 51):
            n = (f % 5) + 1
            tracks = _make_tracks(n)
            speeds = {t["id"]: 5.0 for t in tracks}
            ta.update(tracks, speeds, frame_num=f)


# ======================================================================
# TA-06 ~ TA-07: 레벨·jam_score 반환값 검증
# ======================================================================

class TestLevelOutput:
    """set_baseline 후 레벨·jam_score 반환값을 검증한다."""

    def test_ta06_level_valid_after_baseline(self, cfg):
        """TA-06: set_baseline() 후 update() → get_congestion_level() 유효 문자열."""
        ta = _make_analyzer(cfg)
        ta.set_baseline()
        tracks = _make_tracks(5)
        speeds = {t["id"]: 5.0 for t in tracks}
        ta.update(tracks, speeds, frame_num=1)
        level = ta.get_congestion_level()
        assert level in ("SMOOTH", "SLOW", "JAM"), f"유효하지 않은 level: {level}"

    def test_ta07_get_jam_score_float(self, cfg):
        """TA-07: get_jam_score() 반환값은 float이고 0.0~1.0 범위여야 한다."""
        ta = _make_analyzer(cfg)
        ta.set_baseline()
        tracks = _make_tracks(5)
        speeds = {t["id"]: 5.0 for t in tracks}
        ta.update(tracks, speeds, frame_num=1)
        score = ta.get_jam_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ======================================================================
# TA-08: 지속 시간 — SMOOTH이면 0.0
# ======================================================================

class TestDuration:
    """정체 지속 시간 추적을 검증한다."""

    def test_ta08_duration_smooth_is_zero(self, cfg):
        """TA-08: SMOOTH 상태에서 get_duration_sec() == 0.0."""
        ta = _make_analyzer(cfg)
        ta.set_baseline()
        ta.update(tracks=[], speeds={}, frame_num=1)
        assert ta.get_duration_sec() == 0.0


# ======================================================================
# CP-01 ~ CP-02: CongestionPredictor 추세 판정
# ======================================================================

class TestCongestionPredictor:
    """CongestionPredictor의 추세(trend) 판정을 검증한다."""

    @pytest.fixture
    def predictor(self, cfg):
        """기본 CongestionPredictor 인스턴스를 반환한다."""
        return CongestionPredictor(cfg=cfg, fps=30.0)

    def test_cp01_worsening_trend(self, predictor):
        """CP-01: 속도가 계속 떨어지면 trend == 'WORSENING'."""
        for i in range(30):
            predictor.update(avg_speed=80.0 - i * 2.0)    # 80→22 하락
        result = predictor.predict()
        assert result["trend"] == "WORSENING"

    def test_cp02_improving_trend(self, predictor):
        """CP-02: 속도가 계속 올라가면 trend == 'IMPROVING'."""
        for i in range(30):
            predictor.update(avg_speed=20.0 + i * 2.0)    # 20→78 상승
        result = predictor.predict()
        assert result["trend"] == "IMPROVING"
