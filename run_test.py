# 파일 경로: 최종 프로젝트/run_test.py
# 역할: traffic_analyzer 연동 확인용 테스트 실행 스크립트
#        detect_only=False → 영상 초반 자동 학습 후 탐지 시작
#        화면 좌하단에 "Traffic: SMOOTH/SLOW/JAM  xx.xkm/h" 표시되면 연동 성공

import sys                                              # sys.path 조작용
from pathlib import Path                                # 경로 조작용

# ── import 경로 설정 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent          # 최종 프로젝트/ 절대 경로
# src/ 폴더: Phase 1 flat import(from feature_extractor import ...) 해소용
sys.path.insert(0, str(PROJECT_ROOT / "src"))           # flat import 경로 (Phase 1 파일용)
sys.path.insert(0, str(PROJECT_ROOT))                   # from src import ... 패키지 경로

from src import Detector, DetectorConfig                # src/__init__.py 경유

# ── 경로 설정 ──────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "runs" / "yolo11n_v6" / "weights" / "best.pt"  # YOLO 모델 가중치

VIDEO_DIR = Path(                                       # 영상이 들어있는 폴더
    r"N:\개인\박대원\0211~0313_miniproject\highway-anomaly-detection\data"
)
VIDEO_FILE = "역주행테스트.mp4"          # 실행할 영상 파일명

FLOW_MAP_PATH = PROJECT_ROOT / "flow_maps" / "flow_map.npy"  # 한강 학습 완료 flow_map
# RESULT_DIR    = PROJECT_ROOT / "results"            # 결과 영상 저장 폴더
RESULT_DIR    = Path(r"N:\개인\대원&수빈\최종 프로젝트\임시\results")      # 결과 영상 저장 폴더

# ── DetectorConfig 구성 ───────────────────────────────────────────
cfg = DetectorConfig(
    model_path=MODEL_PATH,                              # YOLO 모델 경로
    conf=0.4,                                           # 검출 신뢰도 임계값
    grid_size=20,                                       # 20×20 Flow Map 그리드
    target_classes=None,                                # 모든 클래스 탐지
    detect_only=True,                                  # 학습 후 탐지 모드
    flow_map_path=FLOW_MAP_PATH,                        # 학습 완료 후 저장될 경로
    result_dir=RESULT_DIR,                              # 결과 영상 저장 폴더
    data_dir=VIDEO_DIR,                                 # 입력 영상 폴더
    log_dir=RESULT_DIR / "logs",                        # CSV 로그 저장 폴더
)

# ── 실행 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print(" 테스트 실행 시작")
    print(f" 영상: {VIDEO_FILE}")
    print(f" 모드: 학습 후 탐지 (detect_only=False)")
    print(f" 확인 포인트: 화면 좌하단 'Traffic: ...' 텍스트")
    print("=" * 50)

    detector = Detector(cfg)                            # Detector 초기화
    detector.run(VIDEO_FILE)                            # 영상 실행
