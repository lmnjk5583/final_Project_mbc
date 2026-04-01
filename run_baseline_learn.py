# 파일 경로: 최종 프로젝트/run_baseline_learn.py
# 역할: 한강대교 10분 영상으로 베이스라인(flow_map.npy) 학습·저장
#        학습 완료 후 baselines/hangang_baseline.npy 생성됨
#        이후 run_baseline_detect.py에서 로드해 즉시 탐지 시작

import sys                                              # sys.path 조작용
from pathlib import Path                                # 경로 조작용

# ── import 경로 설정 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent          # 최종 프로젝트/ 절대 경로
sys.path.insert(0, str(PROJECT_ROOT / "src"))           # flat import 경로
sys.path.insert(0, str(PROJECT_ROOT))                   # from src import ... 패키지 경로

from src import Detector, DetectorConfig                # src/__init__.py 경유

# ── 경로 설정 ──────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "runs" / "yolo11n_v1" / "weights" / "best.pt"  # YOLO 모델 가중치

VIDEO_DIR  = Path(                                      # 한강대교 10분 영상 폴더
    r"N:\개인\대원&수빈\최종 프로젝트\임시\2026-03-31_10-44-07\videos"
)
VIDEO_FILE = "record_2026-03-31_10-44-07.mp4"          # 베이스라인 학습용 영상 파일명

FLOW_MAP_PATH = PROJECT_ROOT / "baselines" / "hangang_baseline.npy"  # 베이스라인 저장 경로
RESULT_DIR    = PROJECT_ROOT / "results"                     # 결과 영상 저장 폴더

# ── DetectorConfig 구성 ───────────────────────────────────────────
cfg = DetectorConfig(
    model_path=MODEL_PATH,                              # YOLO 모델 경로
    conf=0.5,                                           # 검출 신뢰도 임계값
    grid_size=20,                                       # 15×15 Flow Map 그리드
    target_classes=None,                                # 모든 클래스 탐지
    detect_only=False,                                  # 학습 모드 — flow_map 새로 생성
    enable_online_flow_update=False,                    # 베이스라인 학습 중 온라인 갱신 비활성
    flow_map_path=FLOW_MAP_PATH,                        # 학습 완료 후 저장될 경로
    result_dir=RESULT_DIR,                              # 결과 영상 저장 폴더
    data_dir=VIDEO_DIR,                                 # 입력 영상 폴더
)

# ── 실행 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" 베이스라인 학습 시작")
    print(f" 영상: {VIDEO_FILE}")
    print(f" 학습 프레임: {cfg.learning_frames:,} (≈10분 @ 30fps)")
    print(f" 저장 경로: {FLOW_MAP_PATH}")
    print(" 학습 완료 시 'LCS=X.XX' 출력 확인")
    print("=" * 55)

    detector = Detector(cfg)                            # Detector 초기화
    detector.run(VIDEO_FILE)                            # 영상 실행 → 학습 후 .npy 저장
