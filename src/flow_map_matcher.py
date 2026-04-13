# 파일 경로: C:\final_pj\src\flow_map_matcher.py
# 역할: 현재 카메라 프레임과 저장된 ref_frame.jpg를 비교해
#        가장 유사한 flow_map 폴더를 자동 선택한다.
# 의존성: cv2, numpy (표준 환경)

import cv2
import numpy as np
from pathlib import Path


# ── 매칭에 사용할 축소 해상도 (속도·정확도 균형) ─────────────────────────
_MATCH_SIZE = (128, 128)   # 두 프레임 모두 이 크기로 리사이즈 후 비교


def _load_gray(path: Path) -> np.ndarray | None:
    """이미지를 그레이스케일로 로드한다."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, _MATCH_SIZE, interpolation=cv2.INTER_AREA)


def _score_orb(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """ORB 키포인트 매칭 점수 (0~1, 높을수록 유사).

    조명·각도 변화에 강함 — 주요 매칭 방법.
    """
    orb = cv2.ORB_create(nfeatures=500)
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)

    if des_a is None or des_b is None:
        return 0.0
    if len(kp_a) < 10 or len(kp_b) < 10:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    if not matches:
        return 0.0

    # 거리 기준 정렬 후 상위 50%만 사용 (노이즈 제거)
    matches = sorted(matches, key=lambda m: m.distance)
    good    = matches[:len(matches) // 2]

    # 좋은 매칭 수 / 키포인트 수로 정규화
    score = len(good) / max(len(kp_a), len(kp_b))
    return float(np.clip(score, 0.0, 1.0))


def _score_hist(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """히스토그램 상관 점수 (0~1, 높을수록 유사).

    ORB 실패 시 fallback.
    """
    hist_a = cv2.calcHist([img_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([img_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)  # -1~1
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))        # 0~1 정규화


def score_frames(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """두 BGR 프레임의 유사도를 0~1로 반환한다.

    ORB(가중치 0.7) + 히스토그램(0.3) 혼합.
    """
    gray_a = cv2.cvtColor(
        cv2.resize(frame_a, _MATCH_SIZE, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY
    )
    gray_b = cv2.cvtColor(
        cv2.resize(frame_b, _MATCH_SIZE, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY
    )
    s_orb  = _score_orb(gray_a, gray_b)
    s_hist = _score_hist(gray_a, gray_b)
    return 0.7 * s_orb + 0.3 * s_hist


class FlowMapMatcher:
    """저장된 flow_map 폴더들 중 현재 화면과 가장 유사한 것을 선택한다.

    Parameters
    ----------
    flow_maps_root : Path
        flow_maps/ 루트 폴더.
    min_score : float
        이 점수 미만이면 매칭 실패로 판단 (새로 학습).
        기본 0.35 — 도로가 같으면 조명이 달라도 보통 0.4+ 나옴.
    """

    def __init__(self, flow_maps_root: Path, min_score: float = 0.35):
        self.root      = flow_maps_root
        self.min_score = min_score

    def _candidates(self) -> list[tuple[Path, Path]]:
        """(road_dir, ref_frame_path) 목록 반환 — ref_frame.jpg 있는 폴더만."""
        result = []
        if not self.root.exists():
            return result
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            flow_npy = d / "flow_map.npy"
            ref_jpg  = d / "ref_frame.jpg"
            if flow_npy.exists() and ref_jpg.exists():
                result.append((d, ref_jpg))
        return result

    def find_best(self, current_frame: np.ndarray,
                  exclude_dir: Path | None = None
                  ) -> tuple[Path | None, float]:
        """current_frame과 가장 유사한 flow_map 폴더를 찾는다.

        Parameters
        ----------
        current_frame : np.ndarray
            현재 카메라 BGR 프레임.
        exclude_dir : Path | None
            현재 CCTV 자신의 폴더 (자기 자신과 비교 제외).

        Returns
        -------
        (best_dir, score)
            best_dir: 매칭된 폴더 (None이면 min_score 미달 → 새 학습 필요)
            score   : 유사도 0~1
        """
        candidates = self._candidates()
        if not candidates:
            return None, 0.0

        best_dir   = None
        best_score = 0.0

        for road_dir, ref_path in candidates:
            if exclude_dir is not None and road_dir == exclude_dir:
                continue  # 자기 자신 제외

            ref_img = cv2.imdecode(
                np.fromfile(str(ref_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if ref_img is None:
                continue

            s = score_frames(current_frame, ref_img)
            print(f"  [매칭] {road_dir.name}: {s:.3f}")

            if s > best_score:
                best_score = s
                best_dir   = road_dir

        if best_score < self.min_score:
            print(f"  [매칭] 최고 점수 {best_score:.3f} < 기준 {self.min_score} → 새 학습 필요")
            return None, best_score

        print(f"  [매칭] ✅ 선택: {best_dir.name} (score={best_score:.3f})")
        return best_dir, best_score


def save_ref_frame(frame: np.ndarray, road_dir: Path) -> bool:
    """학습 완료 시점의 프레임을 ref_frame.jpg로 저장한다.

    detector.py의 학습 완료 직후 호출한다.

    Parameters
    ----------
    frame : np.ndarray
        저장할 BGR 프레임.
    road_dir : Path
        저장 대상 폴더 (flow_map.npy와 같은 위치).

    Returns
    -------
    bool : 저장 성공 여부.
    """
    road_dir.mkdir(parents=True, exist_ok=True)
    out_path = road_dir / "ref_frame.jpg"
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False
    out_path.write_bytes(buf.tobytes())
    print(f"[ref_frame] 저장 완료: {out_path}")
    return True
