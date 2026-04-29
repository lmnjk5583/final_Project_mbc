# 파일 경로: C:\final_pj\src\flow_map_matcher.py
# 역할: 현재 카메라 프레임과 저장된 ref_frame.jpg를 비교해
#        가장 유사한 flow_map 폴더를 자동 선택한다.
# 의존성: cv2, numpy (표준 환경)

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime


# ── 매칭에 사용할 축소 해상도 (속도·정확도 균형) ─────────────────────────
_MATCH_SIZE = (128, 128)   # 두 프레임 모두 이 크기로 리사이즈 후 비교



def _score_orb(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """ORB 키포인트 매칭 점수 (0~1, 높을수록 유사). score_frames에서 가중치 0.10 보조 신호.

    ORB(Oriented FAST + Rotated BRIEF) 동작 원리:
      1. FAST로 키포인트 검출 — 모서리·경계처럼 주변과 뚜렷이 구분되는 점
         (가드레일 끝, 차선 경계, 신호등 기둥, 건물 모서리 등)
      2. BRIEF로 각 키포인트 주변 패턴을 이진 문자열(디스크립터)로 압축
      3. 두 이미지의 디스크립터를 Hamming 거리(비트 XOR 차이)로 비교해 매칭 쌍 수 집계
         → 매칭이 많을수록 두 이미지가 같은 장소·각도를 찍은 것

    카메라 전환 판별에 유효한 이유:
      같은 카메라라면 도로 구조물이 항상 같은 위치에 보여서 매칭 수가 많고,
      다른 카메라로 전환되면 배경 구조 자체가 달라서 매칭이 거의 안 됨.
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

    # 거리 기준 정렬 후 상위 50%만 사용 — 노이즈성 약매칭 제거
    matches = sorted(matches, key=lambda m: m.distance)
    good    = matches[:len(matches) // 2]

    # 좋은 매칭 수 / 키포인트 수로 정규화 → 0~1
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


def _score_static_region(current_frame: np.ndarray, ref_img: np.ndarray,
                          static_mask: np.ndarray) -> float:
    """flow_map 정적 영역(차량 미검출 구역)만 비교한 유사도 (0~1).

    저장된 flow_map의 count==0 영역은 학습 기간 동안 차량이 한 번도
    지나가지 않은 구역 — 즉 가드레일·방음벽·중앙분리대·노면 표시·
    원거리 구조물 등 카메라 위치 고유 배경을 담는다.

    ▸ 상단 20%(하늘) 제거 — 같은 도로라도 동일한 하늘이 찍힐 수 있음
    ▸ 밝기 NCC + 엣지 NCC 혼합 → 시간대·날씨 변화에 강인
    ▸ 정적 픽셀 < 40개이면 중립값 0.5 반환 (판단 불가)
    """
    small_a = cv2.resize(current_frame, _MATCH_SIZE, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(ref_img,       _MATCH_SIZE, interpolation=cv2.INTER_AREA)

    # static_mask(임의 크기, bool) → _MATCH_SIZE 이진 마스크
    mask_rs = cv2.resize(
        static_mask.astype(np.uint8) * 255,
        _MATCH_SIZE, interpolation=cv2.INTER_NEAREST
    ) > 128

    # 상단 20% (하늘) 제거
    sky_cut = max(1, int(_MATCH_SIZE[1] * 0.20))
    mask_rs[:sky_cut, :] = False

    if int(mask_rs.sum()) < 40:
        return 0.5   # 비교 가능한 정적 픽셀 부족

    gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    norm_a = clahe.apply(gray_a)
    norm_b = clahe.apply(gray_b)

    def _ncc(arr_a, arr_b, mask):
        pa = arr_a[mask].astype(np.float64)
        pb = arr_b[mask].astype(np.float64)
        da = pa - pa.mean();  db = pb - pb.mean()
        denom = np.sqrt((da ** 2).sum() * (db ** 2).sum())
        if denom < 1e-6:
            return 0.5
        return float(np.clip(((da * db).sum() / denom + 1.0) / 2.0, 0.0, 1.0))

    s_bright = _ncc(norm_a, norm_b, mask_rs)
    # 엣지 패턴 비교: 조명 변화에 완전 독립
    edge_a = cv2.Canny(norm_a, 30, 90).astype(np.float64)
    edge_b = cv2.Canny(norm_b, 30, 90).astype(np.float64)
    s_edge = _ncc(edge_a, edge_b, mask_rs)

    return 0.5 * s_bright + 0.5 * s_edge


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



def save_flow_snapshot(frame: np.ndarray, flow_map_obj, save_dir: Path,
                       dir_label_a: str = "") -> bool:
    """학습 완료 시점의 프레임·flow_map·방향 메타데이터를 타임스탬프 이름으로 저장한다.

    파일명 형식:
        flow_map_YYYYMMDD_HHMMSS.npy
        ref_frame_YYYYMMDD_HHMMSS.jpg
        meta_YYYYMMDD_HHMMSS.json   ← dir_label_a + flow_dir_x/y 저장

    Parameters
    ----------
    frame : np.ndarray
        저장할 BGR 프레임.
    flow_map_obj : FlowMap
        저장할 FlowMap 객체 (save(path) 메서드 사용).
    save_dir : Path
        저장 대상 폴더.
    dir_label_a : str
        A방향 레이블 ("UP" 또는 "DOWN"). 빈 문자열이면 메타데이터에 저장 안 함.

    Returns
    -------
    bool : 저장 성공 여부.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npy_path  = save_dir / f"flow_map_{ts}.npy"
    jpg_path  = save_dir / f"ref_frame_{ts}.jpg"
    meta_path = save_dir / f"meta_{ts}.json"
    flow_map_obj.save(npy_path)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False
    jpg_path.write_bytes(buf.tobytes())

    # ── 메타데이터 구성: dir_label_a + flow 방향 벡터 + coverage 마스크 ──
    meta: dict = {}
    if dir_label_a:
        meta["dir_label_a"] = dir_label_a
    try:
        _mask = flow_map_obj.count > 3
        if _mask.any():
            # 지배적 방향 벡터 저장 (방향 비교용)
            _vx = flow_map_obj.flow[_mask, 0]
            _vy = flow_map_obj.flow[_mask, 1]
            _w  = flow_map_obj.count[_mask].astype(float)
            _dx = float(np.average(_vx, weights=_w))
            _dy = float(np.average(_vy, weights=_w))
            _mag = np.sqrt(_dx**2 + _dy**2)
            if _mag > 1e-6:
                meta["flow_dir_x"] = round(_dx / _mag, 4)
                meta["flow_dir_y"] = round(_dy / _mag, 4)
        # coverage 마스크 저장 (차량 공간 분포 비교용)
        _cov = (flow_map_obj.count > 0)
        meta["coverage"] = _cov.tolist()
    except Exception:
        pass

    if meta:
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    print(f"[snapshot] 저장 완료: {npy_path.name} + {jpg_path.name}"
          f" (dir_label_a={dir_label_a or '없음'},"
          f" flow_dir={meta.get('flow_dir_x','?')},{meta.get('flow_dir_y','?')})")
    return True


def _load_coverage_mask(npy_path: Path) -> "np.ndarray | None":
    """후보 flow_map의 coverage 마스크(bool 2D array)를 반환한다.

    1차: meta JSON의 "coverage" 필드(빠름).
    2차: npy 직접 로드 → count>0 (구형 스냅샷 호환, 결과를 meta에 캐싱).
    """
    _stem = npy_path.stem
    _ts   = _stem[len("flow_map_"):] if _stem.startswith("flow_map_") else ""
    _meta_path = npy_path.parent / f"meta_{_ts}.json" if _ts else None

    # ── 1차: meta 캐시 ──────────────────────────────────────────────────
    if _meta_path and _meta_path.exists():
        try:
            _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            if "coverage" in _meta:
                return np.array(_meta["coverage"], dtype=bool)
        except Exception:
            pass

    # ── 2차: npy 직접 계산 ─────────────────────────────────────────────
    try:
        _data  = np.load(npy_path, allow_pickle=True).item()
        _count = _data.get("count")
        if _count is None:
            return None
        _mask = (_count > 0)
        # meta에 캐싱
        if _meta_path:
            try:
                _m = {}
                if _meta_path.exists():
                    _m = json.loads(_meta_path.read_text(encoding="utf-8"))
                _m["coverage"] = _mask.tolist()
                _meta_path.write_text(json.dumps(_m), encoding="utf-8")
            except Exception:
                pass
        return _mask
    except Exception:
        return None


def _coverage_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """두 bool 격자 마스크의 Intersection over Union (0~1).

    IoU = 교집합 셀 수 / 합집합 셀 수

    mask_a: 저장된 스냅샷의 flow_map count>0 셀 (이 카메라가 학습한 도로 영역)
    mask_b: 현재 프레임의 차량 위치 격자 (vehicle_grid)

    같은 카메라라면 차량이 학습된 도로 영역 위에 나타나므로 겹치는 셀이 많아 IoU 높음.
    다른 카메라로 전환되면 학습 영역과 실제 차량 위치가 어긋나 교집합이 줄고 IoU 낮아짐.

    예)  mask_a = [도로 셀 40개],  mask_b = [차량 위치 10개]
         교집합 8개 / 합집합 42개 → IoU = 0.19  (낮음 → 다른 카메라 의심)
         교집합 9개 / 합집합 41개 → IoU = 0.22  (높음 → 같은 카메라)
    """
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a,  mask_b).sum())
    return inter / union if union > 0 else 0.0


def load_snapshot_meta(npy_path: Path) -> dict:
    """npy_path에 대응하는 메타데이터 JSON을 로드해 반환한다.

    flow_map_*.npy 저장 시 함께 기록되는 메타파일로,
    npy와 동일한 타임스탬프를 가진 meta_*.json을 읽는다.
    예) flow_map_20240101_120000.npy → meta_20240101_120000.json

    메타데이터에 포함된 정보:
      dir_label_a : A방향 표시 레이블 (예: "상행", "하행")
                    스냅샷 재매칭 후 방향 레이블을 올바르게 복원하는 데 사용.

    Returns
    -------
    dict : {"dir_label_a": "상행"} 형태, 또는 {} (메타파일 없거나 파싱 실패).
    """
    stem = npy_path.stem  # "flow_map_YYYYMMDD_HHMMSS"
    if not stem.startswith("flow_map_"):
        return {}
    ts_part   = stem[len("flow_map_"):]
    meta_path = npy_path.parent / f"meta_{ts_part}.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _estimate_scene_flow_hint(frame: np.ndarray,
                              prev_frame: "np.ndarray | None" = None
                              ) -> "tuple[float, float] | None":
    """현재 장면의 지배적 교통 흐름 방향을 추정한다.

    전략 1 (우선): prev_frame이 있고 차량 이동이 감지되면 Farneback optical flow 사용.
    전략 2 (fallback): 현재 프레임에서 차선 마킹을 Hough 변환으로 검출 → 차선 각도.

    차량이 정체 중이라도 차선 마킹은 프레임에 남아 있으므로 전략 2는 정체 상황에서도 동작.

    Returns
    -------
    (dx, dy) 정규화 방향 벡터 또는 None (추정 실패).
    """
    # ── 전략 1: Farneback optical flow ─────────────────────────────────
    if prev_frame is not None:
        _sz = (64, 64)
        g1 = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), _sz)
        g2 = cv2.resize(cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY), _sz)
        try:
            _of = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 3, 7, 3, 5, 1.2, 0
            )
            _dx = float(np.median(_of[..., 0]))
            _dy = float(np.median(_of[..., 1]))
            _mag = np.sqrt(_dx**2 + _dy**2)
            if _mag >= 0.5:                                 # 유효한 이동량이면
                return (_dx / _mag, _dy / _mag)
        except Exception:
            pass

    # ── 전략 2: Hough 차선 검출 ─────────────────────────────────────────
    # 이미지 하단 50%(도로 영역)에서 차선 마킹 각도를 추출한다.
    # 차선은 차량 이동 방향과 평행 → 차선 각도 ≈ 흐름 방향 각도.
    try:
        _h, _w = frame.shape[:2]
        _roi = frame[int(_h * 0.40): int(_h * 0.92), :]   # 도로 ROI (상단 배경 제거)
        _gray = cv2.cvtColor(_roi, cv2.COLOR_BGR2GRAY)
        _edges = cv2.Canny(_gray, 40, 120, apertureSize=3)
        _lines = cv2.HoughLinesP(
            _edges, 1, np.pi / 180,
            threshold=20, minLineLength=20, maxLineGap=15
        )
        if _lines is not None:
            _angles = []
            for _ln in _lines:
                _x1, _y1, _x2, _y2 = _ln[0]
                _angles.append(float(np.arctan2(_y2 - _y1, _x2 - _x1)))
            if len(_angles) >= 4:                          # 최소 4개 선분 필요
                _med = float(np.median(_angles))
                return (float(np.cos(_med)), float(np.sin(_med)))
    except Exception:
        pass

    return None                                            # 추정 실패


def _load_flow_dir(npy_path: Path) -> "tuple[float, float] | None":
    """후보 flow_map의 지배적 흐름 방향을 반환한다.

    1차: meta JSON에서 저장된 flow_dir_x/y 빠른 읽기.
    2차: npy 직접 로드 → 가중 평균 방향 계산 (구형 스냅샷 호환).
    방향을 처음 계산한 경우 meta JSON에 캐싱해 다음 호출 속도를 높인다.
    """
    # ── 1차: meta 캐시 ──────────────────────────────────────────────────
    _stem = npy_path.stem                                  # "flow_map_YYYYMMDD_HHMMSS"
    _ts   = _stem[len("flow_map_"):] if _stem.startswith("flow_map_") else ""
    _meta_path = npy_path.parent / f"meta_{_ts}.json" if _ts else None

    if _meta_path and _meta_path.exists():
        try:
            _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            if "flow_dir_x" in _meta and "flow_dir_y" in _meta:
                return (float(_meta["flow_dir_x"]), float(_meta["flow_dir_y"]))
        except Exception:
            pass

    # ── 2차: npy 직접 계산 ─────────────────────────────────────────────
    try:
        _data  = np.load(npy_path, allow_pickle=True).item()
        _flow  = _data.get("flow")                         # (grid, grid, 2)
        _count = _data.get("count")                        # (grid, grid)
        if _flow is None or _count is None:
            return None
        _mask = _count > 3                                 # 의미있는 셀만 (노이즈 제거)
        if not _mask.any():
            return None
        _vx = _flow[_mask, 0]
        _vy = _flow[_mask, 1]
        _w  = _count[_mask].astype(float)
        _dx = float(np.average(_vx, weights=_w))
        _dy = float(np.average(_vy, weights=_w))
        _mag = np.sqrt(_dx**2 + _dy**2)
        if _mag < 1e-6:
            return None
        _dir = (_dx / _mag, _dy / _mag)

        # meta에 캐싱 (다음 실행 시 npy 로드 생략)
        if _meta_path:
            try:
                _m = {}
                if _meta_path.exists():
                    _m = json.loads(_meta_path.read_text(encoding="utf-8"))
                _m["flow_dir_x"] = round(_dir[0], 4)
                _m["flow_dir_y"] = round(_dir[1], 4)
                _meta_path.write_text(json.dumps(_m), encoding="utf-8")
            except Exception:
                pass

        return _dir
    except Exception:
        return None


def find_best_snapshot(current_frame: np.ndarray, save_dir: Path,
                       min_score: float = 0.25,
                       prev_frame: "np.ndarray | None" = None,
                       vehicle_grid: "np.ndarray | None" = None) -> tuple:
    """save_dir에서 current_frame과 가장 유사한 스냅샷 쌍을 찾는다.

    타임스탬프 파일(ref_frame_*.jpg + flow_map_*.npy)과
    레거시 파일(ref_frame.jpg + flow_map.npy) 모두 검색한다.

    점수 산출 (vehicle_grid 제공 시):
        combined = 시각×0.45 + coverage_IoU×0.40 + 방향×0.15
    (vehicle_grid 미제공 시):
        combined = 시각×0.65 + 방향×0.35

    vehicle_grid: 현재 탐지된 차량 위치를 flow_map 격자 크기의 bool 마스크로
                  표현한 것. 같은 카메라라면 coverage가 이 마스크와 겹친다.
                  카메라가 전환되면 차량 위치 분포가 달라져 IoU가 낮아진다.

    Parameters
    ----------
    current_frame : np.ndarray
        현재 카메라 BGR 프레임.
    save_dir : Path
        스냅샷이 저장된 폴더.
    min_score : float
        시각 점수 기준 하한 (이 미만이면 후보 제외).
    prev_frame : np.ndarray | None
        직전 프레임. 제공 시 optical flow로 방향 추정 정확도 향상.
    vehicle_grid : np.ndarray | None
        현재 프레임의 차량 위치 격자 마스크 (bool, shape=(grid_size, grid_size)).

    Returns
    -------
    (best_npy_path, visual_score)
        best_npy_path: 매칭된 .npy 경로 (None이면 매칭 실패)
        visual_score: 시각 유사도 0~1
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

    # ── 현재 장면 흐름 방향 추정 (1회만 계산) ─────────────────────────────
    _scene_hint = _estimate_scene_flow_hint(current_frame, prev_frame)
    if _scene_hint is not None:
        _hint_str = f"({_scene_hint[0]:+.2f}, {_scene_hint[1]:+.2f})"
        print(f"  [스냅샷] 장면 흐름 방향 추정: {_hint_str}")
    else:
        print(f"  [스냅샷] 장면 흐름 방향 추정 실패 → 시각 점수만 사용")

    # ── 후보별 점수 계산 ───────────────────────────────────────────────
    _use_coverage = vehicle_grid is not None
    scored = []   # [(visual, combined, npy_path)]

    for jpg_path, npy_path in candidates:
        ref_img = cv2.imdecode(
            np.fromfile(str(jpg_path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if ref_img is None:
            continue
        vis = score_frames(current_frame, ref_img)

        # ── coverage 로드 (static 비교 및 IoU에 공통 사용) ────────────
        _cov_mask = _load_coverage_mask(npy_path)  # count>0 bool mask / None

        # ── 정적 배경 영역 점수 ───────────────────────────────────────
        # flow_map count==0 구역(차량 미통과) = 카메라 위치 고유 배경
        # 하늘(상단 20%)은 제외 — 같은 도로 구도 변경 시에도 하늘은 동일 가능
        static_score = 0.5                                 # 기본값: 중립
        if _cov_mask is not None:
            _static_mask = ~_cov_mask                      # count==0 → 정적
            static_score = _score_static_region(current_frame, ref_img, _static_mask)

        # ── 흐름 방향 일치도 ───────────────────────────────────────────
        dir_score = 0.5                                    # 기본값: 중립
        if _scene_hint is not None:
            _fdir = _load_flow_dir(npy_path)
            if _fdir is not None:
                dir_score = abs(
                    _scene_hint[0] * _fdir[0] + _scene_hint[1] * _fdir[1]
                )

        # ── coverage IoU (차량 공간 분포 일치도) ─────────────────────
        cov_score = 0.5                                    # 기본값: 중립
        if _use_coverage and _cov_mask is not None and _cov_mask.shape == vehicle_grid.shape:
            cov_score = _coverage_iou(_cov_mask, vehicle_grid)

        # ── combined 점수 산출 ─────────────────────────────────────────
        # static_score: 차량 없는 구역 엣지·밝기 일치도 → 위치 판별 핵심
        # _use_coverage(카메라 전환) 시 cov_score 추가, dir은 제거
        #   (같은 고속도로 다른 카메라는 방향이 동일하여 dir 변별력 없음)
        if _use_coverage:
            # static(0.55) + vis(0.25) + cov(0.20)
            combined = 0.25 * vis + 0.55 * static_score + 0.20 * cov_score
        else:
            # static(0.40) + vis(0.40) + dir(0.20)
            combined = 0.40 * vis + 0.40 * static_score + 0.20 * dir_score

        scored.append((vis, static_score, combined, npy_path))
        print(f"  [스냅샷] {jpg_path.name}: vis={vis:.3f}  static={static_score:.3f}"
              + (f"  cov={cov_score:.2f}" if _use_coverage else "")
              + f"  dir={dir_score:.2f}  comb={combined:.3f}")

    if not scored:
        return None, 0.0

    # ── 필터: vis ≥ min_score AND static_score ≥ _ST_MIN ────────────────
    # 실측 기준 (경부선 양재 두 카메라):
    #   동일 카메라 (자기자신)        → static ≈ 0.80~1.00
    #   다른 구도 동일 도로 (실측)    → static = 0.607
    #   좌우반전 (시뮬)              → static = 0.569
    #   완전 다른 장면 (시뮬)        → static = 0.524
    # 0.62 임계값: 동일 구도만 통과 — startup/카메라전환 모두 동일 기준 적용
    _ST_MIN = 0.62
    qualifying = [
        (v, c, p) for v, st, c, p in scored
        if v >= min_score and st >= _ST_MIN
    ]
    if not qualifying:
        best_visual = max(v for v, _st, _c, _p in scored)
        best_static = max(st for _v, st, _c, _p in scored)
        print(f"  [스냅샷] 필터 미통과 (vis_max={best_visual:.3f}  static_max={best_static:.3f}"
              f"  기준 vis≥{min_score}  static≥{_ST_MIN:.2f}) → 새 학습 필요")
        return None, best_visual

    qualifying.sort(key=lambda x: x[1], reverse=True)     # combined 내림차순
    best_vis, best_comb, best_npy = qualifying[0]

    print(f"  [스냅샷] ✅ 선택: {best_npy.name} "
          f"(vis={best_vis:.3f}, comb={best_comb:.3f})")
    return best_npy, best_vis


