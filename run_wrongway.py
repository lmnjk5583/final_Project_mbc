# 파일 경로: C:\final_pj\run_wrongway.py
# 역할: 역주행 탐지 실행 진입점 (프로젝트 루트에서 실행)
#        learn 모드: flow_map 없으면 자동 학습 → 저장
#        detect 모드: detect_only=True → 기존 flow_map 로드 후 탐지 전용
# 실행: python run_wrongway.py  (C:\final_pj 폴더에서)

import sys                                              # sys.path 조작용
from pathlib import Path                                # 경로 조작용

# ── import 경로 설정 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent          # 최종 프로젝트/ 절대 경로
# src/ 폴더: Phase 1 flat import(from feature_extractor import ...) 해소용
sys.path.insert(0, str(PROJECT_ROOT / "src"))           # flat import 경로 (Phase 1 파일용)
sys.path.insert(0, str(PROJECT_ROOT))                   # from src import ... 패키지 경로

from src import Detector, DetectorConfig                # src/__init__.py 경유 (run_test.py 동일 방식)

# ── 경로 설정 (이 PC 환경에 맞게 수정) ──────────────────────────
MODEL_PATH = Path(                                      # YOLO 모델 가중치 (.pt)
    r"N:\개인\박대원\0211~0313_miniproject"
    r"\highway-anomaly-detection\runs"
    r"\yolo11n_vehicle_v5\weights\best.pt"
)

VIDEO_DIR = Path(                                       # 입력 영상이 있는 폴더
    r"N:\개인\대원&수빈\최종 프로젝트\임시"
    r"\2026-03-25_13-57-49\videos"
)
VIDEO_FILE = "record_2026-03-25_13-57-49.mp4"               # 학습용 정상 주행 영상 (역주행 차량 없음)

FLOW_MAP_PATH = PROJECT_ROOT / "임시" / "flow_map.npy"  # flow_map 저장/로드 경로
RESULT_DIR    = PROJECT_ROOT / "임시" / "results"        # 결과 영상 저장 폴더
LOG_DIR       = PROJECT_ROOT / "임시" / "logs"           # CSV 로그 저장 폴더

# ── DetectorConfig 구성 ───────────────────────────────────────────
cfg = DetectorConfig(
    model_path=MODEL_PATH,                              # YOLO 모델 경로
    conf=0.7,                                           # 검출 신뢰도 임계값
    grid_size=20,                                       # 20×20 Flow Map 그리드
    target_classes=None,                                # None이면 모든 클래스 탐지
    enable_online_flow_update=True,                     # 정상 흐름 온라인 학습 활성
    detect_only=False,                                   # 기존 flow_map 로드 후 탐지 전용
    log_dir=LOG_DIR,                                    # 로그 저장 폴더
    flow_map_path=FLOW_MAP_PATH,                        # flow_map 저장/로드 경로
    result_dir=RESULT_DIR,                              # 결과 영상 저장 폴더
    data_dir=VIDEO_DIR,                                 # 입력 영상 폴더
)

# ── 실행 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print(" 역주행 탐지 실행 시작")
    print(f" 영상: {VIDEO_FILE}")
    print(f" 모드: {'탐지 전용' if cfg.detect_only else '학습 모드'}")
    print("=" * 50)

    detector = Detector(cfg)                            # Detector 초기화
    detector.run(VIDEO_FILE)                            # 영상 실행
