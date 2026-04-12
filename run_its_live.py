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

ITS_CCTV_API_URL = "http://cctvsec.ktict.co.kr/100/spmnCZ2cxfcOECHMkqlBL8Uhrbm1M7FAqyIP9qfD5DCNBmALQ6A9LrbwLElwrDXgnB2Mt6OboQvF9h28l5zAPBJ2wpKAmrfTZP3aMnTQkkc="  # CCTV 목록 조회 API

# 탐지할 CCTV 이름 — ITS API의 cctvname 값과 정확히 일치해야 함
# 아래 CCTV_NAME을 바꾸면 해당 도로의 독립 폴더에 학습 데이터가 쌓임
CCTV_NAME = "[경부선] 양재"                                   # ← 원하는 CCTV 이름으로 변경

# ── 직접 스트림 URL (선택) ────────────────────────────────────────────
# ITS API 없이 스트림 URL을 직접 아는 경우 여기에 입력하면 API 호출을 건너뜀.
# 사용하지 않으면 None으로 두면 됨.
# 예: DIRECT_STREAM_URL = "http://xxx.xxx.xxx.xxx:8080/stream.m3u8"
DIRECT_STREAM_URL: str | None = None                         # ← 직접 URL 알면 여기 입력

# flow_map / GRU 저장 폴더 — CCTV 이름별로 분리되어 서로 덮어쓰지 않음
ROAD_DIR = PROJECT_ROOT / "flow_maps" / CCTV_NAME
ROAD_DIR.mkdir(parents=True, exist_ok=True)            # 폴더 없으면 자동 생성

MODEL_PATH = PROJECT_ROOT / "runs" / "yolo11n_v5" / "weights" / "best.pt"

# ITS 토큰 URL은 4분 후 만료 → 만료 전 갱신 주기 (초)
URL_REFRESH_INTERVAL = 200                             # 3분 20초마다 URL 갱신

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
    detect_only   = flow_map_path.exists()              # 학습된 맵 있으면 탐지 전용 모드

    if detect_only:
        print(f"[설정] flow_map 존재 → detect_only=True (탐지 + GRU 누적 학습 모드)")
    else:
        print(f"[설정] flow_map 없음 → detect_only=False (flow_map 학습 후 탐지)")

    return DetectorConfig(
        model_path=MODEL_PATH,                         # YOLO 모델
        conf=0.4,                                      # 검출 신뢰도
        grid_size=20,                                  # 20×20 flow_map 그리드
        detect_only=detect_only,                       # 자동 판단
        flow_map_path=flow_map_path,                   # 도로별 저장 경로
        learning_frames=1800,                          # flow_map 학습 프레임 수
        log_dir=ROAD_DIR / "logs",                     # CSV 로그 저장
    )


# ======================================================================
# ── 메인 루프 — URL 만료 시 자동 갱신하며 무한 실행 ─────────────────────
# ======================================================================

def main():
    print("=" * 60)
    print(f" ITS 실시간 탐지 시작")
    print(f" CCTV : {CCTV_NAME}")
    print(f" 저장 : {ROAD_DIR}")
    print(f" URL 갱신 주기: {URL_REFRESH_INTERVAL}초")
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

        # ── Detector 생성 및 실행 ───────────────────────────────────
        cfg = make_config(url)
        detector = Detector(cfg)

        try:
            # run()에 URL을 직접 전달 — OpenCV가 스트림으로 열음
            # URL_REFRESH_INTERVAL 초 후 run()이 반환되면 새 URL로 재시작
            detector.run(url, max_seconds=URL_REFRESH_INTERVAL)

        except KeyboardInterrupt:
            print("\n[종료] 사용자 중단 — feature 로그 저장 후 종료")
            break
        except Exception as e:
            print(f"[오류] 탐지 중 예외 발생: {e}")
            print("[재시도] 10초 후 새 URL로 재시작")
            time.sleep(10)
            continue

        print(f"[세션 {session_count}] 완료 — 새 URL로 갱신 후 계속")


if __name__ == "__main__":
    main()
