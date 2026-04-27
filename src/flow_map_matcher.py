# 파일 경로: C:\final_pj\src\flow_map_matcher.py
# 역할: 현재 카메라 프레임과 저장된 ref_frame.jpg를 비교해
#        가장 유사한 flow_map 폴더를 자동 선택한다.
# 의존성: cv2, numpy (표준 환경)

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


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
    """전역 히스토그램 상관 점수 (0~1)."""
    hist_a = cv2.calcHist([img_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([img_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)  # -1~1
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))        # 0~1 정규화


def _score_edge_structure(img_a: np.ndarray, img_b: np.ndarray,
                           grid: int = 4) -> float:
    """엣지 밀도 공간 분포 비교 (0~1) — 조명에 독립적인 도로 구조 유사도.

    Canny 엣지 맵을 grid×grid 셀로 나눠 각 셀의 엣지 밀도 벡터를 구하고,
    두 벡터의 피어슨 상관계수를 0~1로 정규화한다.
    가드레일·차선·교각·건물 윤곽 등 도로 구조는 조명이 바뀌어도 동일 위치에 나타나므로
    밝기·반사 변화에 강하다.
    """
    edges_a = cv2.Canny(img_a, 40, 120)
    edges_b = cv2.Canny(img_b, 40, 120)
    h, w = img_a.shape[:2]
    ch, cw = h // grid, w // grid
    dens_a, dens_b = [], []
    for r in range(grid):
        for c in range(grid):
            ea = edges_a[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            eb = edges_b[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            if ea.size == 0:
                continue
            dens_a.append(float(ea.mean()))
            dens_b.append(float(eb.mean()))
    if not dens_a:
        return 0.0
    da = np.array(dens_a)
    db = np.array(dens_b)
    # 두 밀도 벡터가 거의 평탄(엣지 극소)하면 절대 차이로 유사도 판단
    if da.std() < 1e-6 or db.std() < 1e-6:
        denom = max(da.mean(), db.mean(), 1.0)
        return float(np.clip(1.0 - abs(da.mean() - db.mean()) / denom, 0.0, 1.0))
    corr = float(np.corrcoef(da, db)[0, 1])
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _score_spatial_hist(img_a: np.ndarray, img_b: np.ndarray,
                        grid: int = 4) -> float:
    """공간 분할 히스토그램 점수 (0~1).

    이미지를 grid×grid 셀로 나눠 각 셀의 히스토그램을 비교한다.
    전역 히스토그램보다 도로 구조(건물·차선·배경)를 더 잘 반영하고,
    차량 대수 변화에 덜 민감하다.
    """
    h, w = img_a.shape[:2]
    ch, cw = h // grid, w // grid
    scores = []
    for r in range(grid):
        for c in range(grid):
            cell_a = img_a[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            cell_b = img_b[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            if cell_a.size == 0 or cell_b.size == 0:
                continue
            ha = cv2.calcHist([cell_a], [0], None, [32], [0, 256])
            hb = cv2.calcHist([cell_b], [0], None, [32], [0, 256])
            cv2.normalize(ha, ha)
            cv2.normalize(hb, hb)
            s = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
            scores.append(float(np.clip((s + 1.0) / 2.0, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


def score_frames(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """두 BGR 프레임의 유사도를 0~1로 반환한다.

    엣지 구조(0.35) + 공간 히스토그램(0.35) + 전역 히스토그램(0.20) + ORB(0.10) 혼합.
    - CLAHE 정규화: 비교 전 두 이미지의 명도를 평탄화 → 밝기 변화·햇빛 반사에 강함
    - 엣지 구조: 가드레일·차선·건물 윤곽 등 조명 불변 특징 비교
    - 공간 히스토그램: 도로 구조·배경을 셀 단위로 비교 → 차량 변화에 강함
    - ORB는 보조 역할만
    """
    small_a = cv2.resize(frame_a, _MATCH_SIZE, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(frame_b, _MATCH_SIZE, interpolation=cv2.INTER_AREA)
    gray_a  = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY)
    gray_b  = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY)

    # ── CLAHE 정규화: 밝기 차이 제거 ────────────────────────────────
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_a = clahe.apply(gray_a)
    norm_b = clahe.apply(gray_b)

    s_edge    = _score_edge_structure(norm_a, norm_b, grid=4)
    s_spatial = _score_spatial_hist(norm_a, norm_b, grid=4)
    s_hist    = _score_hist(norm_a, norm_b)
    s_orb     = _score_orb(norm_a, norm_b)
    return 0.35 * s_edge + 0.35 * s_spatial + 0.20 * s_hist + 0.10 * s_orb


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


def save_flow_snapshot(frame: np.ndarray, flow_map_obj, save_dir: Path) -> bool:
    """학습 완료 시점의 프레임과 flow_map을 타임스탬프 이름으로 저장한다.

    파일명 형식:
        flow_map_YYYYMMDD_HHMMSS.npy
        ref_frame_YYYYMMDD_HHMMSS.jpg

    Parameters
    ----------
    frame : np.ndarray
        저장할 BGR 프레임.
    flow_map_obj : FlowMap
        저장할 FlowMap 객체 (save(path) 메서드 사용).
    save_dir : Path
        저장 대상 폴더.

    Returns
    -------
    bool : 저장 성공 여부.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npy_path = save_dir / f"flow_map_{ts}.npy"
    jpg_path = save_dir / f"ref_frame_{ts}.jpg"
    flow_map_obj.save(npy_path)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False
    jpg_path.write_bytes(buf.tobytes())
    print(f"[snapshot] 저장 완료: {npy_path.name} + {jpg_path.name}")
    return True


def find_best_snapshot(current_frame: np.ndarray, save_dir: Path,
                       min_score: float = 0.25) -> tuple:
    """save_dir에서 current_frame과 가장 유사한 스냅샷 쌍을 찾는다.

    타임스탬프 파일(ref_frame_*.jpg + flow_map_*.npy)과
    레거시 파일(ref_frame.jpg + flow_map.npy) 모두 검색한다.

    Parameters
    ----------
    current_frame : np.ndarray
        현재 카메라 BGR 프레임.
    save_dir : Path
        스냅샷이 저장된 폴더.
    min_score : float
        이 점수 미만이면 매칭 실패 (새 학습 필요).

    Returns
    -------
    (best_npy_path, score)
        best_npy_path: 매칭된 .npy 경로 (None이면 매칭 실패)
        score: 유사도 0~1
    """
    if not save_dir.exists():
        return None, 0.0

    candidates = []

    # 타임스탬프 쌍 수집
    for jpg_path in sorted(save_dir.glob("ref_frame_????????_??????.jpg")):
        ts_part = jpg_path.stem[len("ref_frame_"):]
        npy_path = save_dir / f"flow_map_{ts_part}.npy"
        if npy_path.exists():
            candidates.append((jpg_path, npy_path))

    # 레거시 단일 파일
    legacy_jpg = save_dir / "ref_frame.jpg"
    legacy_npy = save_dir / "flow_map.npy"
    if legacy_jpg.exists() and legacy_npy.exists():
        candidates.append((legacy_jpg, legacy_npy))

    if not candidates:
        return None, 0.0

    best_npy   = None
    best_score = 0.0

    for jpg_path, npy_path in candidates:
        ref_img = cv2.imdecode(
            np.fromfile(str(jpg_path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if ref_img is None:
            continue
        s = score_frames(current_frame, ref_img)
        print(f"  [스냅샷] {jpg_path.name}: {s:.3f}")
        if s > best_score:
            best_score = s
            best_npy   = npy_path

    if best_score < min_score:
        print(f"  [스냅샷] 최고 점수 {best_score:.3f} < 기준 {min_score} → 새 학습 필요")
        return None, best_score

    print(f"  [스냅샷] ✅ 선택: {best_npy.name} (score={best_score:.3f})")
    return best_npy, best_score


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
