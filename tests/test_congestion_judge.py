# 파일 경로: C:\final_pj\tests\test_congestion_judge.py
# 역할: CongestionJudge 단위 테스트 (TDD — 구현 전 작성)
# 실행: pytest tests/test_congestion_judge.py -v  (프로젝트 루트에서)
# TC  : CJ-01 ~ CJ-10

import sys                                             # 모듈 검색 경로 조작용
import os                                              # 경로 처리용
import pytest                                          # pytest 프레임워크

# src/ 폴더를 파이썬 모듈 검색 경로 최우선에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from congestion_judge import CongestionJudge           # 테스트 대상 (Step 6에서 구현)
from baseline_stats import BaselineStats               # 기준선 통계 데이터 클래스


# ======================================================================
# Mock 헬퍼 클래스
# ======================================================================

class _MockCfg:
    """테스트용 DetectorConfig 대역 — CongestionJudge 관련 파라미터만 포함"""
    smooth_jam_threshold      = 0.25   # jam_score < 이 값 → SMOOTH
    slow_jam_threshold        = 0.55   # jam_score < 이 값 → SLOW, 이상 → CONGESTED
    congestion_hysteresis_sec = 15.0   # 레벨 전환 유지 시간 (초)
    stop_mag_threshold        = 3.0    # 정지 판단 mag 임계값


def _make_baseline(lcs: float = 0.1,
                   is_fallback: bool = False) -> BaselineStats:
    """테스트용 BaselineStats 생성 헬퍼"""
    return BaselineStats(                              # 기준선 객체 생성
        free_flow_dwell=30.0,                          # 자유 흐름 체류 시간
        typical_dwell=50.0,                            # 일반 체류 시간
        norm_speed_ref=0.3,                            # 정상 속도 기준
        count_ref=5.0,                                 # 정상 차량 수 기준
        bbox_slope=0.05,                               # bbox 회귀 기울기
        bbox_intercept=30.0,                           # bbox 회귀 절편
        lcs=lcs,                                       # 학습 품질 점수
        quality_warning=False,                         # 품질 경고 없음
        passage_count=10,                              # 유효 passage 수
        is_fallback=is_fallback,                       # fallback 여부
    )


def _normal_x_t() -> dict:
    """정상 상태 feature 벡터 (모든 ratio ≈ 1.0, stop_ratio ≈ 0)"""
    return {
        "norm_speed_ratio": 1.0,   # 속도 정상
        "count_ratio":      1.0,   # 차량 수 정상
        "stop_ratio":       0.0,   # 정지 차량 없음
        "exit_rate_ratio":  1.0,   # 유출 정상
        "dwell_ratio":      1.0,   # 체류시간 정상
        "density_score":    0.1,   # 밀도 낮음
        "rule_jam_score":   0.0,   # jam_score 초기값
    }


def _worst_x_t() -> dict:
    """최악 상태 feature 벡터 (모든 지표 최악)"""
    return {
        "norm_speed_ratio": 0.0,   # 속도 0 (완전 정지)
        "count_ratio":      3.0,   # 차량 수 3배 (포화)
        "stop_ratio":       1.0,   # 모든 차량 정지
        "exit_rate_ratio":  0.0,   # 유출 없음
        "dwell_ratio":      0.0,   # 체류 최대 (자유흐름 대비 0)
        "density_score":    1.0,   # 밀도 최대
        "rule_jam_score":   0.0,   # jam_score 초기값
    }


# ======================================================================
# pytest 픽스처
# ======================================================================

@pytest.fixture
def judge():
    """기본 CongestionJudge 인스턴스 (baseline 없음)"""
    return CongestionJudge(_MockCfg(), fps=30.0)       # fps=30 기준


@pytest.fixture
def judge_with_baseline():
    """baseline이 설정된 CongestionJudge 인스턴스 (LCS=0.1 정상)"""
    j = CongestionJudge(_MockCfg(), fps=30.0)          # 인스턴스 생성
    j.set_baseline(_make_baseline(lcs=0.1))            # 기준선 설정
    return j                                            # 반환


# ======================================================================
# CJ-01: baseline 없이 update() 호출 → fallback jam_score 반환, None 아님
# ======================================================================

def test_cj01_no_baseline_returns_fallback_score(judge):
    """CJ-01: baseline 미설정 상태에서 update() → jam_score float 반환 (None 아님)"""
    x_t = _normal_x_t()                               # 정상 feature 벡터

    level, jam_score = judge.update(x_t, frame_num=1) # baseline 없이 호출

    assert jam_score is not None                       # None이 아님
    assert isinstance(jam_score, float)                # float 타입
    assert 0.0 <= jam_score <= 1.0                     # 0~1 범위 내


# ======================================================================
# CJ-02: x_t 모두 정상(ratio=1.0) → jam_score ≈ 0.0
# ======================================================================

def test_cj02_normal_input_low_jam_score(judge_with_baseline):
    """CJ-02: 모든 feature ratio=1.0(정상) → jam_score ≈ 0.0

    speed_score   = clip(1 - 1.0, 0, 1) = 0.0
    dwell_score   = clip(1 - 1.0, 0, 1) = 0.0
    density_score = clip((1.0 - 1) / 2, 0, 1) = 0.0
    bonus(exit>1.3): 0.0, bonus(stop<0.05): +0.05, bonus(speed>1.2): 0.0
    jam = 0 - 0.05 → max(0, -0.05) = 0.0
    """
    x_t = _normal_x_t()                               # 정상 feature 벡터

    _, jam_score = judge_with_baseline.update(x_t, frame_num=1)

    assert jam_score < 0.1, f"정상 입력 jam_score는 0.1 미만이어야 함 (실제={jam_score:.4f})"


# ======================================================================
# CJ-03: x_t 모두 최악(ratio=0.0) → jam_score ≈ 1.0
# ======================================================================

def test_cj03_worst_input_high_jam_score(judge_with_baseline):
    """CJ-03: 모든 feature 최악 → jam_score ≈ 1.0

    speed_score   = clip(1 - 0.0, 0, 1) = 1.0
    dwell_score   = clip(1 - 0.0, 0, 1) = 1.0
    density_score = clip((3.0 - 1) / 2, 0, 1) = 1.0
    jam = 0.50*1 + 0.30*1 + 0.20*1 = 1.0 (bonus=0)
    """
    x_t = _worst_x_t()                                # 최악 feature 벡터

    _, jam_score = judge_with_baseline.update(x_t, frame_num=1)

    assert jam_score >= 0.9, f"최악 입력 jam_score는 0.9 이상이어야 함 (실제={jam_score:.4f})"


# ======================================================================
# CJ-04: jam_score < 0.25 → level = "SMOOTH"
# ======================================================================

def test_cj04_low_jam_score_is_smooth(judge_with_baseline):
    """CJ-04: jam_score < 0.25 → get_level() == 'SMOOTH'"""
    x_t = _normal_x_t()                               # 정상 입력 → jam_score 낮음

    level, jam_score = judge_with_baseline.update(x_t, frame_num=1)

    assert jam_score < 0.25                            # 낮은 jam_score 확인
    assert level == "SMOOTH"                           # SMOOTH 레벨 확인
    assert judge_with_baseline.get_level() == "SMOOTH" # get_level()도 일치


# ======================================================================
# CJ-05: 0.25 ≤ jam_score < 0.55 → level = "SLOW"
# ======================================================================

def test_cj05_mid_jam_score_is_slow(judge_with_baseline):
    """CJ-05: jam_score가 SLOW 범위(0.25~0.55)면 히스테리시스 통과 후 'SLOW'

    히스테리시스 450프레임(15초×30fps)을 넘겨야 레벨 전환 확정.
    speed_score=0.5, dwell_score=0.5, density_score=0 → jam=0.40
    bonus: stop_ratio=0 → +0.05 → jam=0.35 (SLOW 범위)
    """
    x_t = {
        "norm_speed_ratio": 0.5,   # 속도 절반
        "count_ratio":      1.0,   # 차량 수 정상
        "stop_ratio":       0.0,   # 정지 없음 (bonus +0.05)
        "exit_rate_ratio":  0.5,   # 유출 절반
        "dwell_ratio":      0.5,   # 체류 절반
        "density_score":    0.2,   # 밀도 낮음
        "rule_jam_score":   0.0,   # 초기값
    }

    hysteresis = int(15.0 * 30.0) + 2                  # 히스테리시스 + 여유 프레임
    for frame in range(1, hysteresis + 1):             # 히스테리시스 통과까지 반복
        level, jam_score = judge_with_baseline.update(dict(x_t), frame_num=frame)

    assert 0.25 <= jam_score < 0.55, f"SLOW 범위(0.25~0.55)여야 함 (실제={jam_score:.4f})"
    assert level == "SLOW", f"레벨은 SLOW여야 함 (실제={level})"


# ======================================================================
# CJ-06: jam_score ≥ 0.55 → level = "CONGESTED"
# ======================================================================

def test_cj06_high_jam_score_is_congested(judge_with_baseline):
    """CJ-06: jam_score ≥ 0.55 → 히스테리시스 통과 후 'CONGESTED'"""
    x_t = _worst_x_t()                                # 최악 입력 → jam_score 높음

    hysteresis = int(15.0 * 30.0) + 2                  # 히스테리시스 + 여유 프레임
    for frame in range(1, hysteresis + 1):             # 히스테리시스 통과까지 반복
        level, jam_score = judge_with_baseline.update(dict(x_t), frame_num=frame)

    assert jam_score >= 0.55                           # 높은 jam_score 확인
    assert level == "CONGESTED"                        # CONGESTED 레벨 확인


# ======================================================================
# CJ-07: CONGESTED → 즉시 SMOOTH 입력 → 15초(fps×15프레임) 동안 CONGESTED 유지
# ======================================================================

def test_cj07_hysteresis_keeps_congested(judge_with_baseline):
    """CJ-07: CONGESTED 진입 후 즉시 SMOOTH 입력해도 히스테리시스 동안 CONGESTED 유지

    설정: congestion_hysteresis_sec=15.0, fps=30.0 → 450프레임 유지
    단계: (1) 450프레임 최악 입력 → CONGESTED 확정
         (2) 즉시 정상 입력으로 전환 → 450프레임 이내 CONGESTED 유지 확인
    """
    fps = 30.0                                         # FPS
    hysteresis_frames = int(15.0 * fps)                # 450프레임

    # ── Phase 1: CONGESTED 상태 진입 (히스테리시스 통과) ──────────────
    x_congested = _worst_x_t()                        # 최악 입력
    for frame in range(1, hysteresis_frames + 2):      # 히스테리시스 + 여유
        level, _ = judge_with_baseline.update(dict(x_congested), frame_num=frame)
    assert level == "CONGESTED"                        # CONGESTED 진입 확인

    # ── Phase 2: 즉시 SMOOTH 입력 → 히스테리시스 동안 CONGESTED 유지 ──
    base_frame = hysteresis_frames + 2                 # Phase 2 시작 프레임
    x_smooth = _normal_x_t()                          # 정상 입력
    check_count = hysteresis_frames - 2                # 히스테리시스 미만 구간
    for frame in range(base_frame, base_frame + check_count):  # 히스테리시스 내
        level, _ = judge_with_baseline.update(dict(x_smooth), frame_num=frame)

    assert level == "CONGESTED", (                     # 아직 CONGESTED 유지
        f"히스테리시스 {hysteresis_frames}프레임 이내에는 CONGESTED 유지되어야 함 (실제={level})"
    )


# ======================================================================
# CJ-08: LCS=0.8 → threshold 완화 → smooth_threshold < 0.25
# ======================================================================

def test_cj08_high_lcs_lowers_smooth_threshold():
    """CJ-08: LCS=0.8 → smooth_threshold = 0.25×(1-0.8×0.40) = 0.17 < 0.25"""
    cfg = _MockCfg()                                   # 설정 Mock
    judge = CongestionJudge(cfg, fps=30.0)             # 인스턴스 생성
    judge.set_baseline(_make_baseline(lcs=0.8))        # LCS=0.8 고오염 기준선

    # LCS=0.8이면 smooth_threshold = 0.25 * (1 - 0.8*0.40) = 0.25 * 0.68 = 0.17
    expected_smooth_thr = cfg.smooth_jam_threshold * (1 - 0.8 * 0.40)

    actual_thr = judge.get_smooth_threshold()          # threshold 조회 메서드

    assert actual_thr < cfg.smooth_jam_threshold, (    # 기본값 0.25보다 낮아야 함
        f"LCS=0.8이면 smooth_threshold < 0.25여야 함 (실제={actual_thr:.4f})"
    )
    assert abs(actual_thr - expected_smooth_thr) < 0.01  # 계산값과 일치


# ======================================================================
# CJ-09: exit_rate_ratio > 1.3 → bonus 적용 → jam_score 감소
# ======================================================================

def test_cj09_high_exit_rate_reduces_jam(judge_with_baseline):
    """CJ-09: exit_rate_ratio > 1.3 → +0.08 bonus → jam_score 감소"""
    # 기본 feature (중간 jam_score 예상)
    x_base = {
        "norm_speed_ratio": 0.5,   # 속도 절반
        "count_ratio":      1.5,   # 차량 약간 많음
        "stop_ratio":       0.2,   # 일부 정지
        "exit_rate_ratio":  0.5,   # 유출 낮음 (bonus 없음)
        "dwell_ratio":      0.5,   # 체류 높음
        "density_score":    0.3,   # 밀도 보통
        "rule_jam_score":   0.0,   # 초기값
    }
    # exit_rate_ratio를 1.5로 올린 버전 (bonus +0.08)
    x_high_exit = dict(x_base)                        # 복사
    x_high_exit["exit_rate_ratio"] = 1.5              # 유출 증가 (> 1.3 → bonus)

    _, jam_no_bonus  = judge_with_baseline.update(dict(x_base),       frame_num=1)
    _, jam_with_bonus = judge_with_baseline.update(dict(x_high_exit), frame_num=2)

    assert jam_with_bonus < jam_no_bonus, (            # bonus 적용 시 jam_score 낮아야 함
        f"exit_rate_ratio>1.3이면 jam_score 감소해야 함 "
        f"(bonus={jam_no_bonus:.4f} → {jam_with_bonus:.4f})"
    )


# ======================================================================
# CJ-10: update() 100회 연속 호출 → 예외 없음
# ======================================================================

def test_cj10_repeated_update_no_exception(judge_with_baseline):
    """CJ-10: update() 100회 연속 호출해도 예외 없이 정상 동작"""
    x_t = _normal_x_t()                               # 정상 feature 벡터

    for frame in range(1, 101):                        # 100회 반복
        try:                                           # 예외 발생 여부 확인
            level, jam_score = judge_with_baseline.update(x_t, frame_num=frame)
            assert isinstance(level, str)              # level은 문자열
            assert isinstance(jam_score, float)        # jam_score는 float
            assert level in ("SMOOTH", "SLOW", "CONGESTED")  # 유효한 레벨
        except Exception as e:                         # 예외 발생 시 실패
            pytest.fail(f"frame {frame}에서 예외 발생: {e}")
