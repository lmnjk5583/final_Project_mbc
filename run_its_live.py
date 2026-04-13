# 파일 경로: C:\final_pj\run_its_live.py
# 역할: ITS API에서 실시간 CCTV 스트림을 받아 하루종일 탐지·학습을 수행한다.
#        flow_map 학습 → GRU feature 누적 → 5분 후 정체 예측까지 자동 진행.
# 실행: python run_its_live.py (프로젝트 루트에서)

import os
import sys
import time
import requests
import urllib3
from pathlib import Path
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # ITS 자체서명 인증서 경고 억제

# ── import 경로 설정 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))          # flat import (feature_extractor 등)
sys.path.insert(0, str(PROJECT_ROOT))                  # from src import ...

load_dotenv(PROJECT_ROOT / ".env")                     # .env에서 환경변수 로드

from src import Detector, DetectorConfig

# ======================================================================
# ── 설정 ───────────────────────────────────────────────────────────────
# ======================================================================

ITS_API_KEY  = os.getenv("ITS_API_KEY")               # .env의 ITS_API_KEY 사용
if not ITS_API_KEY:
    raise EnvironmentError(".env에 ITS_API_KEY가 없습니다. c:\\final_pj\\.env 확인")

ITS_CCTV_API_URL = "http://cctvsec.ktict.co.kr/100/spmnCZ2cxfcOECHMkqlBL8Uhrbm1M7FAqyIP9qfD5DCNBmALQ6A9LrbwLElwrDXgBMVVonSbGpyHECRng19tMxJ2wpKAmrfTZP3aMnTQkkc="  # CCTV 목록 조회 API

# 탐지할 CCTV 이름 — ITS API의 cctvname 값과 정확히 일치해야 함
# 아래 CCTV_NAME을 바꾸면 해당 도로의 독립 폴더에 학습 데이터가 쌓임
CCTV_NAME = "[경부선] 양재"                                   # ← 원하는 CCTV 이름으로 변경

# ── 직접 스트림 URL (선택) ────────────────────────────────────────────
# ITS API 없이 스트림 URL을 직접 아는 경우 여기에 입력하면 API 호출을 건너뜀.
# 사용하지 않으면 None으로 두면 됨.
# 예: DIRECT_STREAM_URL = "http://xxx.xxx.xxx.xxx:8080/stream.m3u8"
DIRECT_STREAM_URL: str | None = None                         # ← 직접 URL 알면 여기 입력

# ── 강제 재학습 옵션 ─────────────────────────────────────────────────────
# True로 바꾸면 기존 flow_map.npy·gru_a.pt·gru_b.pt를 삭제하고 처음부터 재학습
# 학습 완료 후 자동으로 False로 돌려놓지 않으므로 재학습 후 다시 False로 변경할 것
FORCE_RELEARN: bool = False

# flow_map / GRU 저장 폴더 — CCTV 이름별로 분리되어 서로 덮어쓰지 않음
ROAD_DIR = PROJECT_ROOT / "flow_maps" / CCTV_NAME
ROAD_DIR.mkdir(parents=True, exist_ok=True)            # 폴더 없으면 자동 생성

MODEL_PATH = PROJECT_ROOT / "runs" / "yolo11n_v5" / "weights" / "best.pt"

# ITS API 토큰 URL 사용 시 만료 주기 (초) — 직접 HLS URL이면 실질적으로 사용 안 됨
# detector.run()이 스트림 단절(연속 50프레임 실패)을 감지하면 자동으로 루프를 종료하고
# 여기서 새 URL을 발급받아 재시작한다.
URL_REFRESH_INTERVAL = None                            # None = 스트림 단절 시에만 재시작

# ======================================================================
# ── ITS API: CCTV URL 조회 ─────────────────────────────────────────────
# ======================================================================

def fetch_cctv_url(name: str) -> str | None:
    """ITS API에서 CCTV 이름으로 스트림 URL을 조회한다.

    Args:
        name: ITS cctvname (예: "[경부선] 양재").

    Returns:
        스트림 URL 문자열. 못 찾으면 None.
    """
    params = {
        "apiKey":   ITS_API_KEY,
        "type":     "ex",                              # 고속도로
        "cctvType": "1",                               # 실시간 스트리밍
        "minX": "126.0", "maxX": "129.5",
        "minY": "34.5",  "maxY": "38.0",
        "getType":  "json",
    }
    try:
        res = requests.get(ITS_CCTV_API_URL, params=params, timeout=10, verify=False)
        ct = res.headers.get("Content-Type", "")
        print(f"[ITS] HTTP {res.status_code} — Content-Type: {ct}")
        res.raise_for_status()

        # ITS_CCTV_API_URL이 JSON API가 아닌 스트림 URL인 경우 — M3U8 플레이리스트 응답
        # 이 경우 URL 자체를 스트림으로 사용한다 (OpenCV가 HLS 직접 재생 가능)
        if "mpegurl" in ct or res.text.lstrip().startswith("#EXTM3U"):
            print(f"[ITS] HLS 스트림 URL로 직접 사용: {ITS_CCTV_API_URL[:60]}...")
            return ITS_CCTV_API_URL

        try:
            body = res.json()
        except Exception:
            print(f"[ITS] JSON 파싱 실패 — 응답 앞 300자:\n{res.text[:300]}")
            return None
        # ITS API는 응답 구조가 버전마다 다를 수 있음 — 여러 경로 시도
        items = (body.get("response", {}).get("data", None)
                 or body.get("data", None)
                 or [])
        if not isinstance(items, list):
            print(f"[ITS] 예상치 못한 응답 구조: {str(body)[:300]}")
            return None
        print(f"[ITS] 조회된 CCTV 수: {len(items)}")
    except Exception as e:
        print(f"[ITS] API 호출 실패: {e}")
        return None

    # cctvname이 정확히 일치하는 항목 검색
    match = next((x for x in items if x.get("cctvname") == name), None)
    if match is None:
        # 부분 일치로 재시도 (이름이 약간 다를 경우 대비)
        match = next((x for x in items if name in x.get("cctvname", "")), None)

    if match is None:
        print(f"[ITS] '{name}' CCTV를 찾을 수 없음")
        print(f"[ITS] 조회된 CCTV 목록 (앞 10개):")
        for item in items[:10]:
            print(f"       - {item.get('cctvname')}")
        return None

    url = match["cctvurl"]
    print(f"[ITS] URL 발급 완료: {match['cctvname']} ({url[:60]}...)")
    return url


def list_cctvs():
    """ITS API에서 조회 가능한 CCTV 이름 전체를 출력한다 (CCTV_NAME 설정 참고용)."""
    params = {
        "apiKey":   ITS_API_KEY,
        "type":     "ex",
        "cctvType": "1",
        "minX": "126.0", "maxX": "129.5",
        "minY": "34.5",  "maxY": "38.0",
        "getType":  "json",
    }
    try:
        res = requests.get(ITS_CCTV_API_URL, params=params, timeout=10, verify=False)
        items = res.json().get("response", {}).get("data", [])
        print(f"\n[ITS] 전체 CCTV 목록 ({len(items)}개):")
        for item in items:
            print(f"  - {item.get('cctvname')}")
    except Exception as e:
        print(f"[ITS] 목록 조회 실패: {e}")


# ======================================================================
# ── flow_map 자동 매칭 ─────────────────────────────────────────────────
# ======================================================================

def _try_match_flow_map(cctv_url: str, target_flow_map_path: Path) -> bool:
    """스트림 첫 프레임과 저장된 ref_frame들을 비교해 가장 유사한 flow_map을 복사한다.

    Returns:
        True  → 매칭 성공, target_flow_map_path에 flow_map.npy 복사 완료
        False → 매칭 실패 (새 학습 필요)
    """
    try:
        from flow_map_matcher import FlowMapMatcher
        import cv2
        import shutil
    except ImportError:
        return False

    # ── 스트림 첫 프레임 읽기 ─────────────────────────────────────────
    print("[매칭] 스트림 첫 프레임 읽는 중...")
    cap = cv2.VideoCapture(cctv_url)
    frame = None
    for _ in range(30):                                # 최대 30프레임 시도
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break
    cap.release()

    if frame is None:
        print("[매칭] 프레임 읽기 실패 → 건너뜀")
        return False

    # ── 저장된 flow_map 폴더들과 비교 ────────────────────────────────
    matcher = FlowMapMatcher(
        flow_maps_root = PROJECT_ROOT / "flow_maps",
        min_score      = 0.35,                         # 이 점수 미만이면 매칭 실패
    )
    best_dir, score = matcher.find_best(
        current_frame = frame,
        exclude_dir   = ROAD_DIR,                      # 자기 자신 제외
    )

    if best_dir is None:
        return False

    # ── flow_map.npy 복사 (gru_*.pt·pkl은 복사 안 함 — 도로별로 독립) ──
    src_npy = best_dir / "flow_map.npy"
    ROAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_npy), str(target_flow_map_path))
    print(f"[매칭] flow_map 복사: {best_dir.name} → {ROAD_DIR.name} (score={score:.3f})")
    return True


# ======================================================================
# ── Detector 설정 구성 ─────────────────────────────────────────────────
# ======================================================================

def make_config(cctv_url: str) -> DetectorConfig:
    """DetectorConfig를 생성한다.

    Args:
        cctv_url: ITS 스트림 URL.

    Returns:
        DetectorConfig.
    """
    flow_map_path = ROAD_DIR / "flow_map.npy"          # 도로별 flow_map 경로

    if FORCE_RELEARN:                                  # 강제 재학습: 기존 파일 삭제
        for _f in [flow_map_path,
                   ROAD_DIR / "gru_a.pt",
                   ROAD_DIR / "gru_b.pt",
                   ROAD_DIR / "feature_log_a.pkl",
                   ROAD_DIR / "feature_log_b.pkl"]:
            if _f.exists():
                _f.unlink()
                print(f"[재학습] 삭제: {_f.name}")
        print("[재학습] 기존 데이터 초기화 완료 → 처음부터 학습 시작")

    # ── flow_map 없으면 저장된 다른 폴더와 자동 매칭 시도 ──────────────
    # cctv_url로 스트림 첫 프레임을 읽어 ref_frame들과 비교 →
    # 유사한 flow_map 폴더가 있으면 복사해서 즉시 탐지 모드로 시작
    if not flow_map_path.exists() and not FORCE_RELEARN:
        _matched = _try_match_flow_map(cctv_url, flow_map_path)
        if _matched:
            print(f"[매칭] ✅ 기존 flow_map 재사용 → 학습 생략")
        else:
            print(f"[매칭] 매칭 실패 또는 건너뜀 → 새로 학습")

    detect_only = flow_map_path.exists()               # 학습된 맵 있으면 탐지 전용 모드

    if detect_only:
        print(f"[설정] flow_map 존재 → detect_only=True (탐지 + GRU 누적 학습 모드)")
    else:
        print(f"[설정] flow_map 없음 → detect_only=False (flow_map 학습 후 탐지)")

    return DetectorConfig(
        model_path=MODEL_PATH,                         # YOLO 모델
        conf=0.3,                                      # 검출 신뢰도
        grid_size=20,                                  # 20×20 flow_map 그리드
        detect_only=detect_only,                       # 자동 판단
        flow_map_path=flow_map_path,                   # 도로별 저장 경로
        learning_frames=1800,                          # flow_map 학습 프레임 수
        log_dir=ROAD_DIR / "logs",                     # CSV 로그 저장
        night_enhance=True                             # CLAHE 야간 저조도 보정 
    )


# ======================================================================
# ── 메인 루프 — URL 만료 시 자동 갱신하며 무한 실행 ─────────────────────
# ======================================================================

def main():
    print("=" * 60)
    print(f" ITS 실시간 탐지 시작")
    print(f" CCTV : {CCTV_NAME}")
    print(f" 저장 : {ROAD_DIR}")
    print(f" URL 갱신: 스트림 단절 시 자동 재발급")
    print("=" * 60)
    print(" Ctrl+C 로 종료 — 종료 시 누적 feature 로그 자동 저장\n")

    # CCTV 이름을 모르면 아래 주석 해제해서 목록 확인
    # list_cctvs(); return

    session_count = 0                                  # 세션(URL 갱신) 횟수

    while True:
        session_count += 1
        print(f"\n{'─'*50}")
        print(f" [세션 {session_count}] URL 발급 중...")
        print(f"{'─'*50}")

        # ── ITS URL 발급 ────────────────────────────────────────────
        if DIRECT_STREAM_URL:
            # 직접 URL이 있으면 API 호출 없이 그 URL을 그대로 사용
            url = DIRECT_STREAM_URL
            print(f"[ITS] 직접 URL 사용: {url[:60]}...")
        else:
            url = fetch_cctv_url(CCTV_NAME)
            if url is None:
                print("[오류] URL 발급 실패 — 30초 후 재시도")
                time.sleep(30)
                continue

        # ── Detector 생성 (최초 1회만) ─────────────────────────────
        cfg = make_config(url)
        detector = Detector(cfg)

        # ── URL 재발급 콜백 — 스트림 단절 시 새 URL 반환 ────────────
        # Detector 상태(trajectories, flow_map, GRU 등)를 유지한 채
        # cap만 새 URL로 교체 → 끊김 없이 탐지 지속
        def _get_fresh_url():
            if DIRECT_STREAM_URL:
                return DIRECT_STREAM_URL                   # 직접 URL은 만료 없음
            for _retry in range(3):
                _u = fetch_cctv_url(CCTV_NAME)
                if _u:
                    print(f"[URL] 재발급 성공")
                    return _u
                print(f"[URL] 재발급 실패 ({_retry+1}/3) — 10초 후 재시도")
                time.sleep(10)
            return None

        try:
            detector.run(url, max_seconds=None, url_refresher=_get_fresh_url)

        except KeyboardInterrupt:
            print("\n[종료] 사용자 중단 — feature 로그 저장 후 종료")
            break
        except Exception as e:
            print(f"[오류] 탐지 중 예외 발생: {e}")
            print("[재시도] 10초 후 재시작")
            time.sleep(10)
            continue

        print(f"[세션 {session_count}] 루프 종료 — 재시작")


if __name__ == "__main__":
    main()
