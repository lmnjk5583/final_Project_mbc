# 파일 경로: 최종 프로젝트/tests/test_traffic_analyzer_legacy.py
# 역할: 구버전(km/h 기반) TrafficAnalyzer 테스트 — Phase 1 교체 후 비활성화
#        collect_ignore: pytest가 이 파일을 수집하지 않도록 conftest.py에서 제외
# ※ 이 파일은 Phase 1 신규 인터페이스와 호환되지 않습니다. 참조용으로만 보존.

import sys                                        # 모듈 경로 삽입용
import pathlib                                    # 경로 조작용

# ── src/ 폴더를 import 검색 경로에 추가 ──────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np                                # ndarray 비교·생성용
import pytest                                     # 테스트 프레임워크

from config import DetectorConfig                 # 기존 설정 클래스
from traffic_analyzer import (                    # 테스트 대상
    TrafficAnalyzer,
    CongestionPredictor,
)


# ======================================================================
# 공통 픽스처
# ======================================================================

@pytest.fixture
def cfg():
    """테스트 전용 DetectorConfig를 반환한다.
    free_flow_speed=100 km/h, pixels_per_meter=8, fps 30 기준."""
    return DetectorConfig(
        grid_size=15,                             # 15×15 그리드 (기존값 유지)
        velocity_window=15,                       # 속도 계산 프레임 간격
        free_flow_speed=100.0,                    # 자유 흐름 속도 100 km/h
        pixels_per_meter=8.0,                     # 1 m = 8 px
        congestion_hysteresis_sec=15.0,           # 히스테리시스 15초
        prediction_history_window=30,             # 예측 히스토리 30프레임
        prediction_horizon=5,                     # 예측 시간 5분
    )


@pytest.fixture
def analyzer(cfg):
    """기본 TrafficAnalyzer 인스턴스를 반환한다."""
    return TrafficAnalyzer(
        cfg=cfg,                                  # 설정 전달
        frame_w=1920,                             # 프레임 너비
        frame_h=1080,                             # 프레임 높이
        fps=30.0,                                 # 영상 FPS
    )


def _make_tracks(n, cx_start=100, cy_start=200, gap=50):
    """n대의 더미 트랙을 생성한다.
    각 차량은 gap 간격으로 x좌표만 이동하며 bbox 폭·높이 50px."""
    tracks = []                                   # 결과 리스트
    for i in range(n):                            # n대 반복
        cx = cx_start + i * gap                   # 중심 x
        cy = cy_start                             # 중심 y (고정)
        tracks.append({                           # 트랙 딕셔너리 생성
            "id": i + 1,                          # 1-based ID
            "x1": cx - 25,                        # bbox 좌상 x
            "y1": cy - 25,                        # bbox 좌상 y
            "x2": cx + 25,                        # bbox 우하 x
            "y2": cy + 25,                        # bbox 우하 y
            "cx": float(cx),                      # 중심 x (float)
            "cy": float(cy),                      # 중심 y (float)
        })
    return tracks                                 # 트랙 리스트 반환


def _mag_for_kmh(kmh, cfg, fps=30.0):
    """km/h 값을 detector.py가 사용하는 mag(픽셀 이동량)으로 역변환한다.
    공식: mag = kmh / 3.6 * (velocity_window / fps) * pixels_per_meter"""
    return kmh / 3.6 * (cfg.velocity_window / fps) * cfg.pixels_per_meter


# ======================================================================
# TC-01 ~ TC-05: TrafficAnalyzer 기본 동작
# ======================================================================

class TestBasicOperation:
    """TrafficAnalyzer의 기본 입출력을 검증한다."""

    def test_tc01_no_tracks_avg_speed_zero(self, analyzer):
        """TC-01: tracks가 빈 리스트이면 평균 속도 0."""
        analyzer.update(tracks=[], speeds={}, frame_num=1)   # 빈 트랙 갱신
        assert analyzer.get_avg_speed() == 0.0               # 속도 0 확인

    def test_tc02_no_tracks_level_smooth(self, analyzer):
        """TC-02: tracks가 없으면 정체 레벨은 SMOOTH."""
        analyzer.update(tracks=[], speeds={}, frame_num=1)   # 빈 트랙 갱신
        assert analyzer.get_congestion_level() == "SMOOTH"   # SMOOTH 확인

    def test_tc03_density_map_shape(self, analyzer, cfg):
        """TC-03: get_density_map() 반환값은 (15, 15) ndarray."""
        tracks = _make_tracks(3)                             # 3대 생성
        speeds = {t["id"]: 10.0 for t in tracks}             # 임의 속도
        analyzer.update(tracks, speeds, frame_num=1)         # 갱신
        dm = analyzer.get_density_map()                      # 밀도맵 조회
        assert isinstance(dm, np.ndarray)                    # ndarray 확인
        assert dm.shape == (cfg.grid_size, cfg.grid_size)    # (15,15) 확인

    def test_tc04_affected_vehicles_count(self, analyzer, cfg):
        """TC-04: 5대 중 3대가 SLOW 기준 이하 속도이면 affected=3."""
        tracks = _make_tracks(5)                             # 5대 생성
        slow_mag = _mag_for_kmh(25.0, cfg)                   # 25 km/h → mag (CONGESTED 영역)
        fast_mag = _mag_for_kmh(90.0, cfg)                   # 90 km/h → mag (SMOOTH 영역)
        # 차량 1~3: 저속(25 km/h), 차량 4~5: 고속(90 km/h)
        speeds = {
            1: slow_mag, 2: slow_mag, 3: slow_mag,           # 저속 3대
            4: fast_mag, 5: fast_mag,                         # 고속 2대
        }
        analyzer.update(tracks, speeds, frame_num=1)         # 갱신
        affected = analyzer.get_affected_vehicles()           # 영향 차량 수
        assert affected == 3                                 # 저속 3대 확인

    def test_tc05_multiple_updates_no_exception(self, analyzer, cfg):
        """TC-05: update()를 여러 번 호출해도 예외 없음."""
        for i in range(50):                                  # 50 프레임 반복
            n = (i % 5) + 1                                  # 1~5대 교대
            tracks = _make_tracks(n)                         # 트랙 생성
            speeds = {t["id"]: _mag_for_kmh(80.0, cfg) for t in tracks}  # 80 km/h
            analyzer.update(tracks, speeds, frame_num=i + 1) # 갱신 (예외 없어야 함)


# ======================================================================
# TC-06 ~ TC-08: 정체 레벨 판정
# ======================================================================

class TestCongestionLevel:
    """정체 레벨 판정이 free_flow_speed 비율 기준과 일치하는지 검증한다."""

    def test_tc06_smooth_when_high_speed(self, analyzer, cfg):
        """TC-06: avg_speed > free_flow × 0.7 → SMOOTH."""
        tracks = _make_tracks(5)                             # 5대 생성
        mag = _mag_for_kmh(80.0, cfg)                        # 80 km/h (> 100×0.7=70)
        speeds = {t["id"]: mag for t in tracks}              # 전원 80 km/h
        # 히스테리시스 해소를 위해 충분히 많은 프레임 갱신
        for f in range(1, 500):                              # 약 16.7초 (500/30)
            analyzer.update(tracks, speeds, frame_num=f)     # 갱신
        assert analyzer.get_congestion_level() == "SMOOTH"   # SMOOTH 확인

    def test_tc07_slow_when_mid_speed(self, analyzer, cfg):
        """TC-07: free_flow × 0.3 ≤ avg_speed ≤ × 0.7 → SLOW."""
        tracks = _make_tracks(5)                             # 5대 생성
        mag = _mag_for_kmh(50.0, cfg)                        # 50 km/h (30~70 사이)
        speeds = {t["id"]: mag for t in tracks}              # 전원 50 km/h
        for f in range(1, 500):                              # 히스테리시스 해소
            analyzer.update(tracks, speeds, frame_num=f)     # 갱신
        assert analyzer.get_congestion_level() == "SLOW"     # SLOW 확인

    def test_tc08_congested_when_low_speed(self, analyzer, cfg):
        """TC-08: avg_speed < free_flow × 0.3 → CONGESTED."""
        tracks = _make_tracks(5)                             # 5대 생성
        mag = _mag_for_kmh(10.0, cfg)                        # 10 km/h (< 100×0.3=30)
        speeds = {t["id"]: mag for t in tracks}              # 전원 10 km/h
        for f in range(1, 500):                              # 히스테리시스 해소
            analyzer.update(tracks, speeds, frame_num=f)     # 갱신
        assert analyzer.get_congestion_level() == "CONGESTED"  # CONGESTED 확인


# ======================================================================
# TC-09: 히스테리시스
# ======================================================================

class TestHysteresis:
    """레벨 전환 시 15초 유지 규칙을 검증한다."""

    def test_tc09_hysteresis_holds_level(self, analyzer, cfg):
        """TC-09: CONGESTED에서 갑자기 속도 회복해도 15초 동안은 CONGESTED 유지."""
        tracks = _make_tracks(5)                             # 5대 생성
        low_mag = _mag_for_kmh(10.0, cfg)                    # 10 km/h (CONGESTED)
        high_mag = _mag_for_kmh(90.0, cfg)                   # 90 km/h (SMOOTH)

        # Phase 1: CONGESTED 상태 안정화 (20초 = 600프레임)
        for f in range(1, 601):                              # 프레임 1~600
            speeds = {t["id"]: low_mag for t in tracks}      # 전원 저속
            analyzer.update(tracks, speeds, frame_num=f)     # 갱신
        assert analyzer.get_congestion_level() == "CONGESTED"  # CONGESTED 확인

        # Phase 2: 갑자기 고속으로 전환, 히스테리시스 이내(5초 = 150프레임)
        for f in range(601, 751):                            # 프레임 601~750 (5초)
            speeds = {t["id"]: high_mag for t in tracks}     # 전원 고속
            analyzer.update(tracks, speeds, frame_num=f)     # 갱신

        # 5초밖에 안 지났으므로 아직 CONGESTED여야 함
        assert analyzer.get_congestion_level() == "CONGESTED"  # 히스테리시스 유지 확인


# ======================================================================
# TC-10 ~ TC-11: KPI 메서드
# ======================================================================

class TestKPI:
    """점유율(occupancy), 지속시간(duration) 등 KPI 메서드를 검증한다."""

    def test_tc10_occupancy_range(self, analyzer, cfg):
        """TC-10: get_occupancy() 반환값은 0~100 범위."""
        tracks = _make_tracks(10)                            # 10대 생성
        speeds = {t["id"]: _mag_for_kmh(60.0, cfg) for t in tracks}  # 60 km/h
        analyzer.update(tracks, speeds, frame_num=1)         # 갱신
        occ = analyzer.get_occupancy()                       # 점유율 조회
        assert 0.0 <= occ <= 100.0                           # 범위 확인

    def test_tc11_duration_sec_smooth_zero_slow_positive(self, analyzer, cfg):
        """TC-11: SMOOTH에서 duration=0, SLOW 진입 후 시간이 지나면 양수."""
        tracks = _make_tracks(5)                             # 5대 생성
        fast_mag = _mag_for_kmh(90.0, cfg)                   # 90 km/h (SMOOTH)

        # SMOOTH 상태 유지
        for f in range(1, 100):                              # 약 3.3초
            analyzer.update(tracks, {t["id"]: fast_mag for t in tracks}, f)
        assert analyzer.get_duration_sec() == 0.0            # SMOOTH이므로 0

        # SLOW 상태 진입 + 안정화
        slow_mag = _mag_for_kmh(50.0, cfg)                   # 50 km/h (SLOW)
        for f in range(100, 700):                            # 약 20초
            analyzer.update(tracks, {t["id"]: slow_mag for t in tracks}, f)
        duration = analyzer.get_duration_sec()               # 지속 시간 조회
        assert duration > 0.0                                # 양수 확인


# ======================================================================
# TC-12 ~ TC-13: CongestionPredictor
# ======================================================================

class TestCongestionPredictor:
    """CongestionPredictor의 추세(trend)와 회복 시간 예측을 검증한다."""

    @pytest.fixture
    def predictor(self, cfg):
        """기본 CongestionPredictor 인스턴스를 반환한다."""
        return CongestionPredictor(cfg=cfg, fps=30.0)        # fps=30

    def test_tc12_worsening_trend(self, predictor):
        """TC-12: 속도가 계속 떨어지면 trend == 'WORSENING'."""
        for i in range(30):                                  # 30 프레임 감속
            predictor.update(avg_speed=80.0 - i * 2.0)       # 80→22 km/h 하락
        result = predictor.predict()                         # 예측 실행
        assert result["trend"] == "WORSENING"                # 악화 추세 확인

    def test_tc13_improving_trend(self, predictor):
        """TC-13: 속도가 계속 올라가면 trend == 'IMPROVING'."""
        for i in range(30):                                  # 30 프레임 가속
            predictor.update(avg_speed=20.0 + i * 2.0)       # 20→78 km/h 상승
        result = predictor.predict()                         # 예측 실행
        assert result["trend"] == "IMPROVING"                # 개선 추세 확인
