# 파일 경로: C:\final_pj\tests\test_traffic_analyzer.py
# 역할: TrafficAnalyzer·CongestionPredictor의 TDD 테스트 (pytest)
#        Phase 1 신규 설계 TA-01~10 + CP-01~02 (dev_guide.md §9 명세)
#        절대 km/h → baseline 대비 비율(norm_speed_ratio) 기반

import sys                                            # 모듈 경로 삽입용
import pathlib                                        # 경로 조작용

# ── src/ 폴더를 import 검색 경로에 추가 ──────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np                                    # ndarray 비교·생성용
import pytest                                         # 테스트 프레임워크

from config import DetectorConfig                     # 설정 클래스
from baseline_stats import BaselineStats              # 학습 기준선 데이터 클래스
from traffic_analyzer import (                        # 테스트 대상
    TrafficAnalyzer,
    CongestionPredictor,
)


# ======================================================================
# 공통 Mock 클래스
# ======================================================================

class _MockState:
    """DetectorState의 최소 구현 — FeatureExtractor가 참조하는 필드만 포함."""

    def __init__(self, frame_w=1920, frame_h=1080):
        self.frame_w = frame_w                        # 프레임 너비 (픽셀)
        self.frame_h = frame_h                        # 프레임 높이 (픽셀)
        self.first_seen_frame = {}                    # {track_id: 처음 등장 프레임}
        self.entry_positions = {}                     # {track_id: (fx, fy)} 진입 위치


class _MockPassageTracker:
    """PassageTracker의 최소 구현 — FeatureExtractor가 호출하는 두 메서드만 포함."""

    def __init__(self, dwell_val: float = 50.0, exit_count: int = 3):
        """
        Args:
            dwell_val: get_current_dwells()가 반환할 차량당 체류 프레임 수.
            exit_count: get_exit_count_recent()가 반환할 퇴장 차량 수.
        """
        self._dwell_val = dwell_val                   # 체류 프레임 수 (고정값)
        self._exit_count = exit_count                 # 최근 퇴장 차량 수 (고정값)

    def get_current_dwells(self, active_ids: set,
                           frame_num: int) -> list:
        """활성 차량 수만큼 동일한 dwell_val을 반환한다."""
        return [self._dwell_val] * len(active_ids)    # 고정값 × 활성 차량 수

    def get_exit_count_recent(self, window: int) -> int:
        """최근 window 프레임 내 퇴장 차량 수를 반환한다."""
        return self._exit_count                       # 고정값 반환


# ======================================================================
# 공통 헬퍼 함수
# ======================================================================

def _make_baseline(
    norm_speed_ref: float = 0.1,    # 기준 정상 속도 (normalized_mag 단위)
    count_ref: float = 5.0,         # 기준 정상 차량 수
    free_flow_dwell: float = 50.0,  # 자유 흐름 체류 프레임 수
    lcs: float = 0.0,               # Learning Congestion Score
    is_fallback: bool = False,       # fallback 모드 여부
) -> BaselineStats:
    """테스트용 BaselineStats 객체를 생성한다.

    norm_speed_ref=0.1, count_ref=5로 설정하면:
    - bbox_h=50인 차량이 mag=5.0이면 normalized_mag=0.1 → norm_speed_ratio=1.0
    - stop_mag_threshold=3.0이므로 mag=5.0은 정상 이동 판정
    """
    return BaselineStats(
        free_flow_dwell=free_flow_dwell,              # 자유흐름 체류 기준
        typical_dwell=free_flow_dwell * 1.2,          # 일반 체류 (자유흐름의 1.2배)
        norm_speed_ref=norm_speed_ref,                # 기준 정상 속도
        count_ref=count_ref,                          # 기준 차량 수
        bbox_slope=0.0,                               # bbox 크기 기울기 (평탄 가정)
        bbox_intercept=50.0,                          # bbox 높이 기본값
        lcs=lcs,                                      # 학습 오염도 점수
        quality_warning=False,                        # 학습 품질 경고 없음
        passage_count=10,                             # 기준 산출 passage 수
        is_fallback=is_fallback,                      # fallback 여부
    )


def _make_tracks(n: int,
                 cx_base: int = 100,
                 gap: int = 60,
                 cy: int = 500,
                 bbox_h: int = 50) -> list:
    """n대의 더미 트랙 딕셔너리 리스트를 생성한다.

    각 차량:
    - x: cx_base + i*gap (i=0..n-1)
    - y: cy ± bbox_h//2
    - bbox_h: 지정값 (기본 50px)
    - footpoint (fx, fy): (cx, y2)

    bbox_h=50이고 norm_speed_ref=0.1일 때
    mag=5.0 → normalized_mag=5/50=0.1=norm_speed_ref → norm_speed_ratio=1.0
    """
    tracks = []                                       # 결과 리스트
    half_h = bbox_h // 2                              # bbox 절반 높이
    for i in range(n):                                # n대 반복
        cx = float(cx_base + i * gap)                 # 중심 x
        cy_f = float(cy)                              # 중심 y
        y1 = cy_f - half_h                            # bbox 상단 y
        y2 = cy_f + half_h                            # bbox 하단 y (= footpoint y)
        tracks.append({                               # 트랙 딕셔너리
            "id":  i + 1,                             # 1-based ID
            "x1":  cx - 25.0,                         # bbox 좌측 x
            "y1":  y1,                                # bbox 상단 y
            "x2":  cx + 25.0,                         # bbox 우측 x
            "y2":  y2,                                # bbox 하단 y
            "cx":  cx,                                # 중심 x
            "cy":  cy_f,                              # 중심 y
            "fx":  cx,                                # footpoint x (cx와 동일)
            "fy":  y2,                                # footpoint y (= y2)
        })
    return tracks                                     # 트랙 리스트 반환


def _make_analyzer(cfg, passage_tracker=None,
                   frame_w=1920, frame_h=1080, fps=30.0):
    """TrafficAnalyzer 인스턴스를 생성하고 set_state()까지 완료한다.

    set_state() 호출까지 완료해야 FeatureExtractor가 초기화되어
    feature 계산이 가능한 상태가 된다.
    """
    pt = passage_tracker or _MockPassageTracker()     # tracker 없으면 기본 mock 사용
    ta = TrafficAnalyzer(                             # TrafficAnalyzer 생성
        cfg=cfg,                                      # 설정 전달
        frame_w=frame_w,                              # 프레임 너비
        frame_h=frame_h,                              # 프레임 높이
        fps=fps,                                      # FPS
        passage_tracker=pt,                           # PassageTracker 주입
    )
    state = _MockState(frame_w=frame_w, frame_h=frame_h)  # 상태 객체 생성
    ta.set_state(state)                               # FeatureExtractor 초기화
    return ta                                         # 완성된 인스턴스 반환


# ======================================================================
# 공통 픽스처
# ======================================================================

@pytest.fixture
def cfg():
    """테스트 전용 DetectorConfig를 반환한다.

    congestion_hysteresis_sec=0.0 → 레벨 전환이 즉시 발생 (테스트 간소화).
    """
    return DetectorConfig(
        grid_size=15,                                 # 15×15 그리드
        velocity_window=15,                           # 속도 계산 프레임 간격
        free_flow_speed=100.0,                        # 자유 흐름 속도 (km/h, CongestionPredictor용)
        pixels_per_meter=8.0,                         # 1m = 8px (CongestionPredictor 호환용)
        congestion_hysteresis_sec=0.0,                # 히스테리시스 OFF → 즉시 전환
        prediction_history_window=30,                 # 예측 히스토리 창 (CongestionPredictor)
        prediction_horizon=5,                         # 예측 시간 범위 (분)
        stop_mag_threshold=3.0,                       # 구버전 호환용 (미사용)
        norm_stop_threshold=0.05,                     # 원근 보정 정지 임계값 (nm < 0.05 → 정지)
        min_bbox_h=30.0,                              # bbox_h 최솟값 보정 (원거리 소형 박스 과대평가 방지)
        exit_rate_window=30,                          # 퇴장률 슬라이딩 윈도우 (프레임)
        smooth_jam_threshold=0.25,                    # SMOOTH 상한 임계값
        slow_jam_threshold=0.55,                      # SLOW 상한 임계값
    )


# ======================================================================
# TA-01 ~ TA-05: 기본 동작 검증
# ======================================================================

class TestBasicOperation:
    """TrafficAnalyzer 초기화·기본 입출력을 검증한다."""

    def test_ta01_no_baseline_level_smooth(self, cfg):
        """TA-01: baseline 없이 update() → get_congestion_level() = 'SMOOTH'.

        학습 중(baseline 미설정)에는 feature 계산이 스킵되어
        레벨이 초기값 'SMOOTH'를 유지해야 한다.
        """
        ta = _make_analyzer(cfg)                      # baseline 없이 생성
        tracks = _make_tracks(5)                      # 5대 트랙
        speeds = {t["id"]: 5.0 for t in tracks}       # 임의 mag=5.0
        ta.update(tracks, speeds, frame_num=1)         # feature 계산 스킵
        assert ta.get_congestion_level() == "SMOOTH"  # 초기값 유지 확인

    def test_ta02_empty_tracks_level_smooth(self, cfg):
        """TA-02: set_baseline 후 빈 tracks → get_congestion_level() = 'SMOOTH'.

        차량이 없으면 feature 계산을 건너뛰어 레벨이 'SMOOTH'로 유지된다.
        """
        ta = _make_analyzer(cfg)                      # analyzer 생성
        baseline = _make_baseline()                   # 기준선 생성
        ta.set_baseline(baseline)                     # 기준선 설정

        ta.update(tracks=[], speeds={}, frame_num=1)  # 빈 트랙으로 갱신
        assert ta.get_congestion_level() == "SMOOTH"  # 레벨 확인

    def test_ta03_density_map_shape(self, cfg):
        """TA-03: get_density_map() 반환값은 (15, 15) ndarray."""
        ta = _make_analyzer(cfg)                      # analyzer 생성
        tracks = _make_tracks(3)                      # 3대 트랙
        speeds = {t["id"]: 5.0 for t in tracks}       # 임의 속도
        ta.update(tracks, speeds, frame_num=1)         # 갱신
        dm = ta.get_density_map()                     # 밀도맵 조회
        assert isinstance(dm, np.ndarray)             # ndarray 타입 확인
        assert dm.shape == (15, 15)                   # (15, 15) 형상 확인

    def test_ta04_jam_score_range(self, cfg):
        """TA-04: get_jam_score() 반환값은 0.0~1.0 범위."""
        ta = _make_analyzer(cfg)                      # analyzer 생성
        baseline = _make_baseline()                   # 기준선 생성
        ta.set_baseline(baseline)                     # 기준선 설정
        tracks = _make_tracks(5)                      # 5대 트랙
        speeds = {t["id"]: 5.0 for t in tracks}       # 임의 속도
        ta.update(tracks, speeds, frame_num=1)         # 갱신
        score = ta.get_jam_score()                    # jam_score 조회
        assert 0.0 <= score <= 1.0                    # 범위 확인

    def test_ta05_multiple_updates_no_exception(self, cfg):
        """TA-05: update() 50회 반복 호출 시 예외가 발생하지 않아야 한다."""
        ta = _make_analyzer(cfg)                      # analyzer 생성
        baseline = _make_baseline()                   # 기준선 생성
        ta.set_baseline(baseline)                     # 기준선 설정
        for f in range(1, 51):                        # 50 프레임 반복
            n = (f % 5) + 1                           # 1~5대 교대
            tracks = _make_tracks(n)                  # 트랙 생성
            speeds = {t["id"]: 5.0 for t in tracks}   # 임의 속도
            ta.update(tracks, speeds, frame_num=f)     # 갱신 (예외 없어야 함)


# ======================================================================
# TA-06 ~ TA-07: jam_score 판정
# ======================================================================

class TestJamScore:
    """feature 벡터 값에 따라 jam_score가 올바른 범위에 속하는지 검증한다."""

    def test_ta06_normal_speed_jam_score_low(self, cfg):
        """TA-06: 정상 속도 차량 → jam_score < 0.25 (SMOOTH 판정 기준).

        baseline norm_speed_ref=0.1, bbox_h=50 → mag=5.0이면 norm_speed_ratio=1.0.
        count=5=count_ref → count_ratio=1.0, dwell=50=free_flow_dwell → dwell_ratio=1.0.
        speed/dwell/density 모두 0 → jam=0.0 < 0.25.
        """
        pt = _MockPassageTracker(dwell_val=50.0, exit_count=3)   # 정상 체류 시간
        ta = _make_analyzer(cfg, passage_tracker=pt)              # analyzer 생성
        baseline = _make_baseline(                                # 기준선 설정
            norm_speed_ref=0.1,                                   # 정상 속도 기준
            count_ref=5.0,                                        # 정상 차량 수 기준
            free_flow_dwell=50.0,                                 # 자유흐름 체류 기준
        )
        ta.set_baseline(baseline)                                 # 기준선 전달

        tracks = _make_tracks(5, bbox_h=50)                       # bbox_h=50, 5대
        # mag=5.0 → normalized_mag=5/50=0.1=norm_speed_ref → ratio=1.0 → speed_score=0
        speeds = {t["id"]: 5.0 for t in tracks}                   # 정상 속도 mag=5.0

        ta.update(tracks, speeds, frame_num=1)                    # 갱신
        score = ta.get_jam_score()                                # jam_score 조회
        assert score < 0.25, f"정상 속도인데 jam_score={score:.3f} ≥ 0.25"

    def test_ta07_stopped_vehicles_jam_score_high(self, cfg):
        """TA-07: 정지 차량 다수 + 충분한 dwell → jam_score ≥ 0.55 (CONGESTED 판정).

        mag=1.0, bbox_h=50 → nm=0.02 < norm_stop_threshold=0.05 → 모두 정지 판정.
        dwell_val=600 >> free_flow_dwell=50 → dwell_ratio≈0.08 → dwell_score≈0.92.
        speed_score: norm_mag=0.02, ratio=0.02/0.1=0.2, score=0.8.
        jam = 0.50×0.8 + 0.30×0.92 = 0.40 + 0.28 = 0.68 ≥ 0.55.
        """
        pt = _MockPassageTracker(dwell_val=600.0, exit_count=0)   # 높은 체류시간
        ta = _make_analyzer(cfg, passage_tracker=pt)              # analyzer 생성
        baseline = _make_baseline(                                # 기준선 설정
            norm_speed_ref=0.1,                                   # 정상 속도 기준
            count_ref=5.0,                                        # 정상 차량 수 기준
            free_flow_dwell=50.0,                                 # 자유흐름 체류 기준
        )
        ta.set_baseline(baseline)                                 # 기준선 전달

        tracks = _make_tracks(5, bbox_h=50)                       # bbox_h=50, 5대
        # nm=1.0/50=0.02 < norm_stop_threshold=0.08 → 정지 판정
        speeds = {t["id"]: 1.0 for t in tracks}                   # 정지 속도 mag=1.0

        ta.update(tracks, speeds, frame_num=1)                    # 갱신
        score = ta.get_jam_score()                                # jam_score 조회
        assert score >= 0.55, f"정지 차량인데 jam_score={score:.3f} < 0.55"


# ======================================================================
# TA-08 ~ TA-09: 지속 시간 (duration)
# ======================================================================

class TestDuration:
    """정체 지속 시간 추적을 검증한다."""

    def test_ta08_duration_smooth_is_zero(self, cfg):
        """TA-08: SMOOTH 상태에서 get_duration_sec() = 0.0.

        level='SMOOTH'이면 _congestion_start_frame=None → duration=0.0.
        """
        ta = _make_analyzer(cfg)                                  # analyzer 생성
        baseline = _make_baseline()                               # 기준선
        ta.set_baseline(baseline)                                 # 기준선 설정
        # 빈 트랙 → SMOOTH 유지
        ta.update(tracks=[], speeds={}, frame_num=1)              # 갱신
        assert ta.get_duration_sec() == 0.0                       # 지속 시간 0 확인

    def test_ta09_duration_congested_positive(self, cfg):
        """TA-09: CONGESTED 진입 후 30프레임이 지나면 get_duration_sec() > 0.0.

        congestion_hysteresis_sec=0.0 → jam≥0.55이면 첫 프레임에 CONGESTED 확정.
        _congestion_start_frame이 설정되어 30/30=1.0초 이상 경과.
        """
        pt = _MockPassageTracker(dwell_val=600.0, exit_count=0)   # 높은 체류시간
        ta = _make_analyzer(cfg, passage_tracker=pt, fps=30.0)    # analyzer 생성
        baseline = _make_baseline(
            norm_speed_ref=0.1,                                   # 정상 속도 기준
            count_ref=5.0,                                        # 정상 차량 수 기준
            free_flow_dwell=50.0,                                  # 자유흐름 체류 기준
        )
        ta.set_baseline(baseline)                                 # 기준선 설정

        tracks = _make_tracks(5, bbox_h=50)                       # bbox_h=50, 5대
        speeds = {t["id"]: 1.0 for t in tracks}                   # 정지 → CONGESTED 유도

        # 30프레임 동안 갱신 (CONGESTED 상태 유지)
        for f in range(1, 31):                                    # 프레임 1~30
            ta.update(tracks, speeds, frame_num=f)                # 갱신

        duration = ta.get_duration_sec()                          # 지속 시간 조회
        assert duration > 0.0, f"CONGESTED 30프레임 후 duration={duration}초"


# ======================================================================
# TA-10: 영향 차량 수
# ======================================================================

class TestAffectedVehicles:
    """get_affected_vehicles()가 정지 차량 수를 정확히 반환하는지 검증한다."""

    def test_ta10_affected_vehicles_count(self, cfg):
        """TA-10: 5대 중 3대가 nm < norm_stop_threshold → affected = 3.

        norm_stop_threshold=0.05, bbox_h=50 (≥ min_bbox_h=30이므로 클램프 없음):
        - ID 1~3: mag=1.0 → nm=1/50=0.02 < 0.05 → 정지
        - ID 4~5: mag=5.0 → nm=5/50=0.10 ≥ 0.05 → 이동
        baseline 없으면 early return이 발생하므로 baseline 설정 후 테스트.
        """
        ta = _make_analyzer(cfg)                                  # analyzer 생성
        baseline = _make_baseline()                               # 기준선
        ta.set_baseline(baseline)                                 # 기준선 설정

        tracks = _make_tracks(5, bbox_h=50)                       # 5대 트랙
        # 1~3: 정지(nm=0.02 < 0.08), 4~5: 이동(nm=0.10 ≥ 0.08)
        speeds = {1: 1.0, 2: 1.0, 3: 1.0, 4: 5.0, 5: 5.0}       # 혼합 속도

        ta.update(tracks, speeds, frame_num=1)                    # 갱신
        affected = ta.get_affected_vehicles()                     # 영향 차량 수 조회
        assert affected == 3, f"정지 차량 3대인데 affected={affected}"


# ======================================================================
# CP-01 ~ CP-02: CongestionPredictor
# ======================================================================

class TestCongestionPredictor:
    """CongestionPredictor의 추세(trend) 판정을 검증한다.

    CongestionPredictor는 Phase 1 공개 인터페이스 유지 목적으로
    traffic_analyzer.py에 그대로 보존된 클래스이다.
    """

    @pytest.fixture
    def predictor(self, cfg):
        """기본 CongestionPredictor 인스턴스를 반환한다."""
        return CongestionPredictor(cfg=cfg, fps=30.0)             # fps=30

    def test_cp01_worsening_trend(self, predictor):
        """CP-01: 속도가 계속 떨어지면 trend == 'WORSENING'."""
        for i in range(30):                                       # 30 프레임 감속
            predictor.update(avg_speed=80.0 - i * 2.0)           # 80→22 하락
        result = predictor.predict()                              # 예측 실행
        assert result["trend"] == "WORSENING"                    # 악화 추세 확인

    def test_cp02_improving_trend(self, predictor):
        """CP-02: 속도가 계속 올라가면 trend == 'IMPROVING'."""
        for i in range(30):                                       # 30 프레임 가속
            predictor.update(avg_speed=20.0 + i * 2.0)           # 20→78 상승
        result = predictor.predict()                              # 예측 실행
        assert result["trend"] == "IMPROVING"                    # 개선 추세 확인
