# 파일 경로: C:\final_pj\tests\test_nm_measurement.py
# 역할: nm(normalized_mag) 계산 로직 단위 테스트
#        실제 영상 없이 가상 차량 데이터로 nm 계산 결과를 확인한다.
# 실행: pytest tests/test_nm_measurement.py -v  (프로젝트 루트에서)
#        또는 python tests/test_nm_measurement.py  (단독 실행)

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ======================================================================
# nm 계산 함수 (feature_extractor.py 로직 직접 복제)
# ======================================================================

MIN_BBOX_H    = 30.0   # bbox_h 최솟값 클램프 (원거리 과대평가 방지)
NORM_STOP_THR = 0.06   # nm < 이 값 → 정지
SLOW_UPPER_NM = 0.50   # nm < 이 값 → 서행 (이상이면 정상 주행)


def compute_nm(mag: float, bbox_h: float) -> float:
    """이동량과 bbox 높이로 nm을 계산한다.

    Args:
        mag: velocity_window 구간 이동량 (픽셀).
        bbox_h: 바운딩박스 높이 (픽셀).

    Returns:
        nm = mag / max(bbox_h, MIN_BBOX_H)
    """
    return mag / max(bbox_h, MIN_BBOX_H)


def classify_nm(nm: float) -> str:
    """nm 값으로 차량 상태를 분류한다.

    Returns:
        "stop"   nm < 0.06
        "slow"   0.06 <= nm < 0.50
        "normal" nm >= 0.50
    """
    if nm < NORM_STOP_THR:
        return "stop"
    if nm < SLOW_UPPER_NM:
        return "slow"
    return "normal"


def compute_ratios(vehicles: list) -> dict:
    """가상 차량 리스트에서 stop_ratio / slow_ratio를 계산한다.

    Args:
        vehicles: [{"mag": float, "bbox_h": float}, ...] 형태의 리스트.
                  mag=0 이면 완전 정지 (nm 계산 없이 stop으로 처리).

    Returns:
        {
            "stop_ratio": float,
            "slow_ratio": float,
            "nm_list": [float, ...],  # 각 차량 nm (mag=0은 0.0으로 기록)
            "labels": [str, ...]      # 각 차량 분류 레이블
        }
    """
    stopped = 0
    slow    = 0
    total   = len(vehicles)
    nm_list = []
    labels  = []

    for v in vehicles:
        mag    = v["mag"]
        bbox_h = v["bbox_h"]

        if mag <= 0:                        # 완전 정지
            stopped += 1
            slow    += 1                    # 정지 ⊂ 서행
            nm_list.append(0.0)
            labels.append("stop")
            continue

        nm = compute_nm(mag, bbox_h)
        nm_list.append(nm)

        label = classify_nm(nm)
        labels.append(label)

        if label == "stop":
            stopped += 1
            slow    += 1
        elif label == "slow":
            slow += 1

    if total == 0:
        return {"stop_ratio": 0.0, "slow_ratio": 0.0,
                "nm_list": [], "labels": []}

    return {
        "stop_ratio": stopped / total,
        "slow_ratio": slow    / total,
        "nm_list":    nm_list,
        "labels":     labels,
    }


# ======================================================================
# 테스트 케이스
# ======================================================================

def test_nm_정지차량():
    """완전 정지(mag=0) 차량은 nm=0.0, stop으로 분류된다."""
    nm = compute_nm(0, 80)
    assert nm == 0.0
    assert classify_nm(nm) == "stop"


def test_nm_원근보정_근거리_원거리_동일():
    """근거리·원거리 차량이 실제로 같은 속도이면 nm도 동일해야 한다."""
    # 근거리: bbox_h=150, 이동량=30px
    nm_near = compute_nm(30, 150)
    # 원거리: bbox_h=30, 이동량=6px  (화면상 1/5 크기, 1/5 이동)
    nm_far  = compute_nm(6, 30)
    assert abs(nm_near - nm_far) < 0.01, \
        f"원근 보정 실패: near={nm_near:.3f}, far={nm_far:.3f}"


def test_nm_최솟값_클램프():
    """bbox_h가 MIN_BBOX_H(30)보다 작으면 30으로 클램프된다."""
    # bbox_h=10px (극소형 원거리 차량)
    nm_clamped   = compute_nm(6, 10)   # 실제: 6/30 = 0.20
    nm_unclamped = 6 / 10              # 클램프 없으면: 0.60
    assert nm_clamped == 6 / 30, f"클램프 미적용: {nm_clamped}"
    assert nm_clamped < nm_unclamped   # 클램프하면 낮아짐 (과대평가 방지)


def test_nm_분류_경계값():
    """nm 경계값(0.06, 0.50)에서 분류가 정확히 전환된다."""
    assert classify_nm(0.059) == "stop"
    assert classify_nm(0.060) == "slow"    # 경계: slow 시작
    assert classify_nm(0.499) == "slow"
    assert classify_nm(0.500) == "normal"  # 경계: normal 시작


def test_stop_ratio_정체상황():
    """정체 상황: 차량 10대 중 7대 정지 → stop_ratio=0.70.

    정상 주행은 nm >= 0.50 이어야 한다.
    mag=50, bbox_h=80 → nm=0.625 → normal
    (mag=20이면 nm=0.25 → slow 범위이므로 주의)
    """
    vehicles = (
        [{"mag": 0,  "bbox_h": 80}] * 7 +   # 7대 정지 (nm=0.0 → stop)
        [{"mag": 50, "bbox_h": 80}] * 3      # 3대 정상 (nm=0.625 → normal)
    )
    result = compute_ratios(vehicles)
    assert result["stop_ratio"] == 0.70, result["stop_ratio"]
    assert result["slow_ratio"] == 0.70  # 정지는 서행에도 포함, 정상 3대는 미포함


def test_slow_ratio_서행상황():
    """서행 상황: 차량 10대 중 6대 서행(정지 아님) → slow_ratio=0.60."""
    vehicles = (
        [{"mag": 15, "bbox_h": 80}] * 6 +   # 6대 서행 (nm=0.19)
        [{"mag": 50, "bbox_h": 80}] * 4      # 4대 정상 (nm=0.63)
    )
    result = compute_ratios(vehicles)
    assert result["stop_ratio"] == 0.0
    assert result["slow_ratio"] == 0.60, result["slow_ratio"]


def test_slow_ratio_정지포함():
    """정지 차량은 slow_ratio에도 포함된다 (정지 ⊂ 서행)."""
    vehicles = (
        [{"mag": 0,  "bbox_h": 80}] * 3 +   # 3대 정지
        [{"mag": 15, "bbox_h": 80}] * 3 +   # 3대 서행
        [{"mag": 50, "bbox_h": 80}] * 4      # 4대 정상
    )
    result = compute_ratios(vehicles)
    assert result["stop_ratio"] == 0.30
    assert result["slow_ratio"] == 0.60      # 정지(3)+서행(3) = 6/10


def test_원활상황_nm():
    """원활 상황: 모든 차량 nm >= 0.50 → stop/slow 모두 0."""
    vehicles = [{"mag": 50, "bbox_h": 80}] * 10  # nm=0.625 (정상)
    result = compute_ratios(vehicles)
    assert result["stop_ratio"] == 0.0
    assert result["slow_ratio"] == 0.0
    assert all(l == "normal" for l in result["labels"])


def test_2차선_4차선_비율_동일():
    """2차선(차량 5대)과 4차선(차량 10대)이 같은 비율로 서행이면 slow_ratio 동일."""
    vehicles_2lane = (
        [{"mag": 15, "bbox_h": 80}] * 3 +   # 3대 서행
        [{"mag": 50, "bbox_h": 80}] * 2      # 2대 정상
    )
    vehicles_4lane = (
        [{"mag": 15, "bbox_h": 80}] * 6 +   # 6대 서행
        [{"mag": 50, "bbox_h": 80}] * 4      # 4대 정상
    )
    r2 = compute_ratios(vehicles_2lane)
    r4 = compute_ratios(vehicles_4lane)
    assert r2["slow_ratio"] == r4["slow_ratio"] == 0.60


# ======================================================================
# 단독 실행 시 결과 출력
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("nm 측정 테스트 — 단독 실행 모드")
    print("=" * 60)

    # ── 시나리오별 nm·비율 출력 ──────────────────────────────────────
    scenarios = {
        "원활": [
            {"mag": 60, "bbox_h": 80},
            {"mag": 55, "bbox_h": 90},
            {"mag": 50, "bbox_h": 70},
            {"mag": 65, "bbox_h": 85},
            {"mag": 58, "bbox_h": 80},
        ],
        "서행": [
            {"mag": 15, "bbox_h": 80},
            {"mag": 20, "bbox_h": 90},
            {"mag": 10, "bbox_h": 70},
            {"mag":  0, "bbox_h": 80},   # 정지
            {"mag": 18, "bbox_h": 85},
        ],
        "정체": [
            {"mag":  0, "bbox_h": 80},
            {"mag":  0, "bbox_h": 90},
            {"mag":  3, "bbox_h": 70},   # nm=0.043 → 정지
            {"mag":  0, "bbox_h": 85},
            {"mag":  5, "bbox_h": 80},   # nm=0.063 → 서행
        ],
    }

    for name, vehicles in scenarios.items():
        result = compute_ratios(vehicles)
        print(f"\n[{name}]")
        for i, (v, nm, label) in enumerate(
            zip(vehicles, result["nm_list"], result["labels"])
        ):
            print(f"  차량{i+1}: mag={v['mag']:4.0f}px  "
                  f"bbox_h={v['bbox_h']}px  "
                  f"nm={nm:.3f}  → {label}")
        print(f"  stop_ratio = {result['stop_ratio']:.2f}  "
              f"slow_ratio = {result['slow_ratio']:.2f}")

    # ── pytest 실행 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("pytest 실행")
    print("=" * 60)
    import pytest
    pytest.main([__file__, "-v"])
