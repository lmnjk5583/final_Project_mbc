# 파일 경로: C:\final_pj\tests\test_passage_tracker.py
# 역할: PassageTracker 단위 테스트 (TDD — 구현 전 작성)
# 실행: pytest tests/test_passage_tracker.py -v  (프로젝트 루트에서)
# TC  : PT-01 ~ PT-08

import sys                                            # 모듈 검색 경로 조작용
import os                                             # 경로 처리용
import math                                           # 거리 계산 검증용
import numpy as np                                    # percentile 등 수치 검증용
import pytest                                         # pytest 프레임워크

# src/ 폴더를 파이썬 모듈 검색 경로 최우선에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from passage_tracker import PassageTracker            # 테스트 대상 (Step 3에서 구현)
from baseline_stats import BaselineStats, MIN_PASSAGE_DIST  # 데이터 클래스


# ======================================================================
# Mock 헬퍼 클래스
# ======================================================================

class _MockCfg:
    """테스트용 DetectorConfig 대역 — Phase 1 파라미터만 포함"""
    min_passage_dist      = 100.0  # 유효 passage 최소 진입-퇴장 거리 (픽셀)
    min_passages_required = 5      # 학습 종료 최소 유효 passage 수
    stop_mag_threshold    = 3.0    # 정지 판단 mag 임계값 (픽셀)
    exit_rate_window      = 30     # exit_rate 슬라이딩 윈도우 (프레임)
    grid_size             = 15     # 그리드 크기 (밀도 계산용)


class _MockState:
    """테스트용 DetectorState 대역"""
    def __init__(self):
        self.first_seen_frame = {}   # {track_id: 첫 등장 프레임 번호}
        self.entry_positions  = {}   # {track_id: (fx, fy)} — PassageTracker가 채움
        self.frame_num        = 0    # 현재 프레임 번호 (테스트에서 직접 관리)
        self.frame_w          = 1280 # 영상 너비 (픽셀)
        self.frame_h          = 720  # 영상 높이 (픽셀)


# ======================================================================
# pytest 픽스처
# ======================================================================

@pytest.fixture
def tracker_bundle():
    """cfg + state + PassageTracker 세 객체를 튜플로 반환"""
    cfg   = _MockCfg()                   # 설정 Mock
    state = _MockState()                  # 상태 Mock
    t     = PassageTracker(cfg, state)    # 테스트 대상 인스턴스
    return t, cfg, state                  # 세 객체 반환


# ======================================================================
# 공통 헬퍼 함수
# ======================================================================

def _create_passage(tracker, state, tid,
                    entry_frame, exit_frame,
                    entry_fx=0.0,   entry_fy=500.0,
                    exit_fx=200.0,  exit_fy=500.0,
                    is_complete=True):
    """단일 passage를 기록하는 헬퍼 — on_entry + on_exit 순서 호출"""
    state.first_seen_frame[tid] = entry_frame         # 첫 등장 프레임 state에 등록
    tracker.on_entry(tid, entry_fx, entry_fy, entry_frame)  # 진입 기록
    tracker.on_exit(tid, exit_fx, exit_fy, exit_frame,      # 퇴장 기록
                    is_complete=is_complete)


def _add_frame_stats(tracker, n_frames, active_count, exit_count,
                     norm_mags_per_frame, cy=500.0, bbox_h=50.0):
    """동일한 frame_stats를 n_frames 번 반복 기록하는 헬퍼"""
    for _ in range(n_frames):                          # n_frames 반복
        tracker.record_frame_stats(
            active_count=active_count,                 # 활성 차량 수
            exit_count=exit_count,                     # 퇴장 차량 수
            norm_mags=list(norm_mags_per_frame),       # 복사본 전달 (외부 변경 방지)
            cy_vals=[cy] * active_count,               # cy 값 리스트
            bbox_h_vals=[bbox_h] * active_count        # bbox_h 값 리스트
        )


# ======================================================================
# PT-01: on_entry → on_exit 순서 호출 → PassageRecord 1개 생성, dwell_frames 정확
# ======================================================================

def test_pt01_entry_exit_creates_record_with_correct_dwell(tracker_bundle):
    """PT-01: on_entry(frame=10) → on_exit(frame=40) → PassageRecord 1개, dwell=30"""
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    entry_frame    = 10                                # 진입 프레임
    exit_frame     = 40                                # 퇴장 프레임
    expected_dwell = exit_frame - entry_frame          # 기대 체류 프레임 = 30

    # dist = sqrt((200-0)^2 + (500-500)^2) = 200.0 > 100.0 → 유효
    _create_passage(tracker, state, tid=1,
                    entry_frame=entry_frame, exit_frame=exit_frame,
                    entry_fx=0.0, entry_fy=500.0,
                    exit_fx=200.0, exit_fy=500.0)

    # ── 검증 ──
    assert tracker.get_completed_count() == 1          # 유효 PassageRecord 1개 생성
    record = tracker._passages[0]                      # 내부 records 직접 접근
    assert record.dwell_frames == expected_dwell       # 체류 프레임 = 30
    assert record.is_complete  is True                 # 정상 퇴장
    assert record.is_valid     is True                 # 유효한 passage


# ======================================================================
# PT-02: entry_exit_dist < min_passage_dist → is_valid=False
# ======================================================================

def test_pt02_short_distance_invalid(tracker_bundle):
    """PT-02: 진입~퇴장 거리 10px < 100px(min_passage_dist) → is_valid=False"""
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    # dist = sqrt((10-0)^2 + 0^2) = 10.0 < 100.0 → 무효
    _create_passage(tracker, state, tid=1,
                    entry_frame=0, exit_frame=30,
                    entry_fx=0.0,  entry_fy=500.0,
                    exit_fx=10.0,  exit_fy=500.0,
                    is_complete=True)

    # ── 검증 ──
    assert tracker.get_completed_count() == 0          # 유효 passage 없음 (거리 부족)
    record = tracker._passages[0]                      # 기록 자체는 존재
    assert record.is_valid is False                    # 유효하지 않은 passage 확인
    assert math.hypot(10.0 - 0.0, 0.0) < cfg.min_passage_dist  # 거리 < 100 명시적 확인


# ======================================================================
# PT-03: 완성 passage 5개(n<10) → finalize_baseline() → free_flow_dwell = min(dwells)
# ======================================================================

def test_pt03_finalize_n_lt_10_uses_min(tracker_bundle):
    """PT-03: 유효 passage 5개(n<10) → free_flow_dwell = min([20,30,40,50,60]) = 20"""
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    dwells = [20, 30, 40, 50, 60]                      # 체류 프레임 목록 (n=5 < 10)
    for i, dwell in enumerate(dwells):                 # 각 passage 생성
        _create_passage(tracker, state, tid=i + 1,
                        entry_frame=i * 200,
                        exit_frame=i * 200 + dwell,
                        entry_fx=0.0, exit_fx=200.0)   # dist=200 > 100 → 유효

    _add_frame_stats(tracker, n_frames=5,              # norm_mags 보충 (LCS 계산용)
                     active_count=3, exit_count=1,
                     norm_mags_per_frame=[0.3, 0.3, 0.3])

    baseline = tracker.finalize_baseline()             # 기준선 산출

    # ── 검증 ──
    assert tracker.get_completed_count() == 5          # 유효 passage 5개
    assert baseline.passage_count == 5                 # BaselineStats passage_count
    assert baseline.free_flow_dwell == min(dwells)     # n<10 → min 사용 → 20


# ======================================================================
# PT-04: 완성 passage 15개(n>=10) → finalize_baseline() → free_flow_dwell = percentile(dwells,10)
# ======================================================================

def test_pt04_finalize_n_ge_10_uses_percentile(tracker_bundle):
    """PT-04: 유효 passage 15개(n>=10) → free_flow_dwell = percentile(dwells, 10)"""
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    # dwells = [10, 20, 30, ..., 150] (15개)
    dwells = list(range(10, 160, 10))                  # 10, 20, ..., 150 — 15개
    for i, dwell in enumerate(dwells):                 # 각 passage 생성
        _create_passage(tracker, state, tid=i + 1,
                        entry_frame=i * 300,
                        exit_frame=i * 300 + dwell,
                        entry_fx=0.0, exit_fx=200.0)   # dist=200 유효

    _add_frame_stats(tracker, n_frames=15,             # norm_mags 보충
                     active_count=3, exit_count=1,
                     norm_mags_per_frame=[0.3, 0.3, 0.3])

    baseline = tracker.finalize_baseline()             # 기준선 산출

    expected_ffd = float(np.percentile(dwells, 10))    # numpy percentile(10)

    # ── 검증 ──
    assert baseline.passage_count == 15                # passage 15개
    assert abs(baseline.free_flow_dwell - expected_ffd) < 0.1  # percentile 일치 (오차 ±0.1)


# ======================================================================
# PT-05: 모두 빠른 차량 (균일 짧은 dwell, 균일 빠른 속도) → LCS < 0.3
# ======================================================================

def test_pt05_fast_vehicles_low_lcs(tracker_bundle):
    """PT-05: 균일한 dwell + 균일한 고속 + exit≈active → LCS < 0.3

    signal_A ≈ 0 (모든 dwell=10으로 동일 → median=min → 분자=0)
    signal_B ≈ 0 (norm_mags 모두 0.5 → mean=max → 비율=1)
    signal_C ≈ 0 (exit_count=active_count=5 → sum(exit)/sum(entry)=1)
    LCS = 0.40*0 + 0.35*0 + 0.25*0 = 0.0 < 0.3
    """
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    # 5개 passage 생성 (모두 동일한 dwell=10)
    for i in range(5):                                 # 5개 동일 passage
        _create_passage(tracker, state, tid=i + 1,
                        entry_frame=i * 50,
                        exit_frame=i * 50 + 10,        # dwell = 10 (균일)
                        entry_fx=0.0, exit_fx=200.0)   # dist=200 유효

    # exit_count = active_count → signal_C = 0
    _add_frame_stats(tracker, n_frames=30,
                     active_count=5, exit_count=5,     # 퇴장 = 활성 → 원활한 흐름
                     norm_mags_per_frame=[0.5, 0.5, 0.5, 0.5, 0.5])  # 균일 고속

    lcs = tracker.compute_lcs()                        # LCS 계산

    assert lcs < 0.3, f"빠른 균일 차량 LCS는 0.3 미만이어야 함 (실제={lcs:.4f})"


# ======================================================================
# PT-06: 모두 느린 차량 (긴 분산 dwell, 저속, 차량 누적) → LCS > 0.6
# ======================================================================

def test_pt06_slow_vehicles_high_lcs(tracker_bundle):
    """PT-06: 분산 큰 긴 dwell + 저속 + 유출 부족 → LCS > 0.6

    signal_A = 1.0  (dwells=[10,50,100,150,200]: median=100, min=10 → (100/10-1)/5=1.8 → clip=1.0)
    signal_B ≈ 0.86 (norm_mags: 9/10이 0.02, 1/10이 0.5 → mean≈0.068, max=0.5 → 1-0.136=0.864)
    signal_C = 0.9  (active=10, exit=1 × 30프레임 → 1-30/300=0.9)
    LCS ≈ 0.40*1.0 + 0.35*0.864 + 0.25*0.9 = 0.927 > 0.6
    """
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    dwells = [10, 50, 100, 150, 200]                   # 분산 큰 dwell 목록
    for i, dwell in enumerate(dwells):                 # 5개 passage 생성
        _create_passage(tracker, state, tid=i + 1,
                        entry_frame=i * 300,
                        exit_frame=i * 300 + dwell,    # 긴 체류
                        entry_fx=0.0, exit_fx=200.0)   # dist=200 유효

    for _ in range(30):                                # 30프레임 기록
        tracker.record_frame_stats(
            active_count=10,                           # 활성 10대 (누적)
            exit_count=1,                              # 유출 1대 (소량) → signal_C 높음
            norm_mags=[0.02] * 9 + [0.5],             # 9대 저속 + 1대 고속 → signal_B 높음
            cy_vals=[500.0] * 10,                      # cy 500
            bbox_h_vals=[50.0] * 10                    # bbox_h 50
        )

    lcs = tracker.compute_lcs()                        # LCS 계산

    assert lcs > 0.6, f"느린 정체 차량 LCS는 0.6 초과여야 함 (실제={lcs:.4f})"


# ======================================================================
# PT-07: on_exit is_complete=False → is_fallback=True (유효 passage 부족)
# ======================================================================

def test_pt07_incomplete_passages_cause_fallback(tracker_bundle):
    """PT-07: 모든 on_exit가 is_complete=False → 유효 passage 0개 → is_fallback=True"""
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    # 3개 passage 생성 — 모두 추적 소실(stale)로 비완성
    for i in range(3):                                 # 비완성 passage 3개
        _create_passage(tracker, state, tid=i + 1,
                        entry_frame=i * 100,
                        exit_frame=i * 100 + 30,
                        entry_fx=0.0, exit_fx=200.0,
                        is_complete=False)             # 추적 소실 → is_valid=False

    _add_frame_stats(tracker, n_frames=5,              # frame_stats 보충
                     active_count=2, exit_count=0,
                     norm_mags_per_frame=[0.2, 0.2])

    # ── 검증 ──
    assert tracker.get_completed_count() == 0          # 유효 passage 없음

    baseline = tracker.finalize_baseline()             # 기준선 산출 시도
    assert baseline.is_fallback is True                # passage 부족 → fallback 모드


# ======================================================================
# PT-08: get_current_dwells() → 활성 차량 수만큼 dwell 값 반환
# ======================================================================

def test_pt08_get_current_dwells_matches_active_count(tracker_bundle):
    """PT-08: 활성 차량 3대 등록 → get_current_dwells() 길이=3, 값=현재 체류 시간"""
    tracker, cfg, state = tracker_bundle               # 픽스처 분리

    current_frame = 100                                # 현재 프레임

    # 3대 차량 활성 등록 (각각 다른 시점에 등장)
    active_ids = set()                                 # 활성 ID 집합
    for i in range(1, 4):                              # ID 1, 2, 3
        entry_frame = current_frame - i * 10           # 각각 90, 80, 70프레임에 등장
        state.first_seen_frame[i] = entry_frame        # state에 등장 프레임 등록
        tracker.on_entry(i,                            # 진입 기록
                         fx=float(i * 100), fy=500.0,
                         frame_num=entry_frame)
        active_ids.add(i)                              # 활성 집합에 추가

    dwells = tracker.get_current_dwells(active_ids, current_frame)  # 체류 시간 조회

    # ── 검증 ──
    assert len(dwells) == 3                            # 활성 차량 3대 → 3개 값 반환

    # 기대값: current_frame - first_seen_frame[tid] 각각
    expected = sorted([current_frame - state.first_seen_frame[i] for i in active_ids])
    assert sorted(dwells) == expected                  # 값 일치 (순서 무관)
