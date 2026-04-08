# 파일 경로: C:\final_pj\test_nm_live.py
# 역할: 실제 영상에서 차량별 nm(normalized_mag) 값을 실시간으로 시각화하는 테스트 스크립트
#        각 차량 bbox 위에 nm 값과 분류(stop/slow/normal)를 표시한다.
# 실행: 프로젝트 루트(C:\final_pj)에서 실행
#        C:\Users\User\.conda\envs\cv\python.exe test_nm_live.py

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── import 경로 설정 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import YoloTracker                                # YOLO+ByteTrack

# ── 설정 ─────────────────────────────────────────────────────────────
MODEL_PATH  = PROJECT_ROOT / "runs" / "yolo11n_v1" / "weights" / "best.pt"
# VIDEO_PATH  = Path(r"N:\개인\대원&수빈\최종 프로젝트\임시") / "정체_완화_테스트.mp4"
VIDEO_PATH  = Path(r"N:\개인\대원&수빈\최종 프로젝트\임시") / "서행_테스트.mp4"
CONF        = 0.45          # YOLO 신뢰도 임계값

VELOCITY_WINDOW    = 10   # nm 계산 프레임 간격 — 10으로 낮춰 원거리 차량 커버
MIN_BBOX_H         = 30.0 # bbox_h 최솟값 클램프
NORM_STOP_THR      = 0.06 # nm < 이 값 → stop
SLOW_UPPER_NM      = 0.70 # nm < 이 값 → slow, 이상 → normal
NM_CY_CORRECTION_K = 0.4  # cy 보정 계수

# ── 색상 ─────────────────────────────────────────────────────────────
COLOR = {
    "stop":   (0,   0,   255),  # 빨강 — 정지
    "slow":   (0,   165, 255),  # 주황 — 서행
    "normal": (0,   255, 0),    # 초록 — 정상
}


def classify_nm(nm: float) -> str:
    """nm 값으로 차량 상태를 분류한다."""
    if nm < NORM_STOP_THR:
        return "stop"
    if nm < SLOW_UPPER_NM:
        return "slow"
    return "normal"


def main():
    # ── 초기화 ───────────────────────────────────────────────────────
    tracker      = YoloTracker(MODEL_PATH, CONF)
    cap          = cv2.VideoCapture(str(VIDEO_PATH))
    trajectories = defaultdict(list)    # {tid: [(cx, cy), ...]} 궤적
    frame_num    = 0

    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"영상 해상도: {fw}×{fh}")
    print("단축키: Q=종료 / P=일시정지 / S=nm 통계 터미널 출력")

    win_name = "nm Live Test"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

        # ── YOLO 추적 ────────────────────────────────────────────────
        tracks = tracker.track(frame)

        # ── 차량별 nm 계산 및 시각화 ─────────────────────────────────
        nm_log = []         # 이 프레임의 (tid, nm, label) 목록 (S키 출력용)

        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])
            cx  = (x1 + x2) / 2
            cy  = (y1 + y2) / 2

            # 궤적 추가
            trajectories[tid].append((cx, cy))
            if len(trajectories[tid]) > VELOCITY_WINDOW + 5:
                trajectories[tid].pop(0)

            # nm 계산 — velocity_window 이상 궤적이 있을 때만
            traj = trajectories[tid]
            if len(traj) >= VELOCITY_WINDOW:
                ox, oy = traj[-VELOCITY_WINDOW]          # VELOCITY_WINDOW 전 위치
                dx     = cx - ox                         # x 이동량 (부호 있음)
                dy     = cy - oy                         # y 이동량 (부호 있음)
                mag    = np.sqrt(dx**2 + dy**2)          # 이동량 (픽셀)
                bbox_h = max(y2 - y1, MIN_BBOX_H)        # 최솟값 클램프
                nm_raw = mag / bbox_h                    # normalized_mag (보정 전)
                nm = nm_raw
                if NM_CY_CORRECTION_K > 0:               # cy 보정 (대칭)
                    cy_ratio = cy / max(fh, 1)
                    denom = 1.0 + NM_CY_CORRECTION_K * (2.0 * cy_ratio - 1.0)
                    nm = nm / max(denom, 0.1)

                # 방향 판별 — dx > 0: 오른쪽(→) 상행, dx < 0: 왼쪽(←) 하행
                if mag > 2.0:                            # 정지에 가까우면 방향 불명
                    direction = "상행" if dx > 0 else "하행"
                else:
                    direction = "정지"

                label = classify_nm(nm)
                color = COLOR[label]

                # bbox 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 모든 차량에 nm + 방향 표시
                if label != "normal":
                    text = f"{direction} nm={nm:.2f} [{label}]"
                    font_scale = 0.45
                else:
                    text = f"{direction} {nm:.2f}"       # normal은 간결하게
                    font_scale = 0.38
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(frame,                                 # 텍스트 배경
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              (0, 0, 0), -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

                nm_log.append((tid, nm, label, mag, bbox_h, nm_raw, direction))
            else:
                # 궤적 부족 — 회색 박스만 (텍스트 없음)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

        # ── 이 프레임 nm 분포 요약 (화면 좌상단) ─────────────────────
        if nm_log:
            nms     = [r[1] for r in nm_log]
            stops   = sum(1 for r in nm_log if r[2] == "stop")
            slows   = sum(1 for r in nm_log if r[2] == "slow")
            normals = sum(1 for r in nm_log if r[2] == "normal")
            total   = len(nm_log)

            summary_lines = [
                f"Frame {frame_num}  |  탐지: {total}대",
                f"nm  min={min(nms):.2f}  max={max(nms):.2f}  avg={np.mean(nms):.2f}",
                f"stop={stops}  slow={slows}  normal={normals}",
                f"stop_ratio={stops/total:.2f}  slow_ratio={(stops+slows)/total:.2f}",
            ]
        else:
            summary_lines = [f"Frame {frame_num}  |  탐지 없음"]

        for i, line in enumerate(summary_lines):
            cv2.putText(frame, line, (10, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

        # ── 단축키 안내 (우상단) ─────────────────────────────────────
        guide = "[Q] 종료  [P] 일시정지  [S] 터미널 출력"
        cv2.putText(frame, guide, (fw - 420, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(win_name, frame)

        # ── 키 입력 ───────────────────────────────────────────────────
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"{'[일시정지]' if paused else '[재개]'}")
        elif key == ord("s"):
            print(f"\n=== Frame {frame_num} nm 상세 ===")
            for tid, nm, label, mag, bbox_h, nm_raw, direction in sorted(nm_log, key=lambda x: x[1]):
                print(f"  [{direction:2s}] ID:{tid:4d}  nm={nm:.3f}(raw={nm_raw:.3f})  [{label:6s}]  "
                      f"mag={mag:5.1f}px  bbox_h={bbox_h:5.1f}px")
            if nm_log:
                nms = [r[1] for r in nm_log]
                up_nms   = [r[1] for r in nm_log if r[6] == "상행"]
                down_nms = [r[1] for r in nm_log if r[6] == "하행"]
                print(f"  → 전체 avg={np.mean(nms):.3f}  stop={stops/total:.2f}  slow={(stops+slows)/total:.2f}")
                if up_nms:
                    print(f"  → 상행 avg={np.mean(up_nms):.3f}  min={min(up_nms):.3f}  max={max(up_nms):.3f}  n={len(up_nms)}")
                if down_nms:
                    print(f"  → 하행 avg={np.mean(down_nms):.3f}  min={min(down_nms):.3f}  max={max(down_nms):.3f}  n={len(down_nms)}")

    cap.release()
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    main()
