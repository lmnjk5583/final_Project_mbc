# 파일 경로: C:\final_pj\auto_label.py
# 역할: YOLO 자동 라벨링 + 수동 수정 통합 도구
# 실행: python auto_label.py
#
# 사용법:
#   1) 실행 → 이미지 폴더 선택 (이미지가 직접 들어있는 폴더)
#   2) 자동 라벨링 여부 선택 (이미 labels/ 있으면 스킵 가능)
#   3) 라벨 편집기 실행
#
# 키 조작:
#   A / D         : 이전 / 다음 이미지
#   , / .         : -10 / +10 이동
#   ; / '         : -100 / +100 이동
#   0~9           : 선택된 박스 클래스 변경 (또는 pending 박스 확정)
#   마우스 드래그  : 새 박스 그리기 → 숫자 키로 클래스 지정
#   우클릭        : 박스 삭제
#   박스 모서리 드래그: 크기 조절
#   박스 테두리 드래그: 이동
#   화살표 키      : 선택 박스 미세 이동
#   Z / Ctrl+Z    : Undo
#   Y / Ctrl+Y    : Redo
#   Q / ESC       : 종료

import os
import sys
import copy
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path

import cv2
import numpy as np

# ── 프로젝트 루트 기준 YOLO 모델 경로 ──────────────────────────────────────
_HERE = Path(__file__).parent
DEFAULT_MODEL = _HERE / "runs" / "yolo11n_v5" / "weights" / "best.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. YOLO 자동 라벨링
# ═══════════════════════════════════════════════════════════════════════════════

def auto_label(image_dir: Path, label_dir: Path, model_path: Path, conf: float = 0.3,
               target_classes=None, skip_existing: bool = True):
    """
    image_dir 내 이미지를 YOLO로 추론 → label_dir에 YOLO 포맷 txt 저장.
    skip_existing=True: 이미 txt가 있는 이미지는 건너뜀.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] ultralytics 패키지가 없습니다: pip install ultralytics")
        return

    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in image_dir.iterdir()
                     if f.suffix.lower() in IMAGE_EXTENSIONS])
    if not images:
        print("[자동 라벨링] 이미지 없음 → 스킵")
        return

    print(f"[자동 라벨링] 모델 로드: {model_path}")
    model = YOLO(str(model_path))

    done = 0
    skip = 0
    for img_path in images:
        txt_path = label_dir / (img_path.stem + ".txt")
        if skip_existing and txt_path.exists():
            skip += 1
            continue

        results = model.predict(
            str(img_path),
            conf=conf,
            classes=target_classes,
            verbose=False,
        )
        boxes = results[0].boxes
        lines = []
        if boxes is not None and len(boxes):
            for box in boxes:
                cls = int(box.cls[0])
                xc, yc, w, h = box.xywhn[0].tolist()   # 정규화 좌표
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(images) - skip} 완료...")

    print(f"[자동 라벨링] 완료: {done}개 라벨 생성 / {skip}개 기존 유지")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 수동 편집기
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_COLORS = [
    (255, 50,  50),  # 0 빨강
    (50,  255, 50),  # 1 초록
    (50,  50,  255), # 2 파랑
    (0,   255, 255), # 3 시안
    (255, 0,   255), # 4 마젠타
    (255, 255, 0),   # 5 노랑
    (0,   165, 255), # 6 주황
    (128, 0,   128), # 7 보라
    (0,   128, 0),   # 8 다크그린
    (128, 128, 128), # 9 회색
]


class LabelEditor:
    def __init__(self, image_dir: Path, label_dir: Path):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_dir.mkdir(parents=True, exist_ok=True)

        self.image_list = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not self.image_list:
            messagebox.showerror("오류", "이미지 파일이 없습니다.")
            self.initialized = False
            return

        # 화면 크기 조회
        root = tk.Tk()
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()
        root.destroy()

        self.current_idx    = 0
        self.boxes          = []        # [[cls, cx, cy, w, h], ...]  (정규화)
        self.history        = []        # undo 스택
        self.redo_history   = []
        self.selected_idx   = -1
        self.drag_mode      = None      # "draw" | "move" | "resize"
        self.pending_box    = None      # 그리는 중인 박스 [cx, cy, w, h]
        self.start_x        = 0.0
        self.start_y        = 0.0
        self.canvas_shape   = (0, 0, 0) # (H, W, C)
        self.initialized    = True

    # ── 파일 I/O ─────────────────────────────────────────────────────────────

    def _txt_path(self, idx=None):
        if idx is None:
            idx = self.current_idx
        return self.label_dir / (self.image_list[idx].stem + ".txt")

    def load_labels(self):
        path = self._txt_path()
        boxes = []
        if path.exists():
            for line in path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    boxes.append([cls, cx, cy, w, h])
        return boxes

    def save_labels(self):
        path = self._txt_path()
        lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                 for b in self.boxes]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))

    # ── 이미지 전환 ──────────────────────────────────────────────────────────

    def change_image(self, step: int):
        new_idx = max(0, min(self.current_idx + step, len(self.image_list) - 1))
        if new_idx == self.current_idx:
            return
        self.current_idx  = new_idx
        self.boxes        = self.load_labels()
        self.selected_idx = -1
        self.pending_box  = None
        self.history.clear()
        self.redo_history.clear()

    # ── Undo / Redo ──────────────────────────────────────────────────────────

    def save_history(self):
        self.history.append(copy.deepcopy(self.boxes))
        if len(self.history) > 50:
            self.history.pop(0)
        self.redo_history.clear()

    def undo(self):
        if self.history:
            self.redo_history.append(copy.deepcopy(self.boxes))
            self.boxes = self.history.pop()
            self.selected_idx = -1
            self.save_labels()

    def redo(self):
        if self.redo_history:
            self.history.append(copy.deepcopy(self.boxes))
            self.boxes = self.redo_history.pop()
            self.selected_idx = -1
            self.save_labels()

    # ── 박스 미세 이동 ───────────────────────────────────────────────────────

    def move_box(self, dx: int, dy: int):
        h, w, _ = self.canvas_shape
        if w == 0 or h == 0:
            return
        norm_dx = (dx * 2) / (w * 0.75)
        norm_dy = (dy * 2) / h
        if self.pending_box:
            self.pending_box[0] = max(0.0, min(1.0, self.pending_box[0] + norm_dx))
            self.pending_box[1] = max(0.0, min(1.0, self.pending_box[1] + norm_dy))
        elif self.selected_idx != -1:
            self.save_history()
            self.boxes[self.selected_idx][1] = max(0.0, min(1.0, self.boxes[self.selected_idx][1] + norm_dx))
            self.boxes[self.selected_idx][2] = max(0.0, min(1.0, self.boxes[self.selected_idx][2] + norm_dy))
            self.save_labels()

    # ── 마우스 이벤트 ────────────────────────────────────────────────────────

    def mouse_events(self, event, x, y, flags, param):
        # 마우스 휠 → 이미지 전환
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = np.int16(flags >> 16) if flags > 100000 else flags
            self.change_image(-1 if delta > 0 else 1)
            return

        h_f, w_f, _ = self.canvas_shape
        w_img = int(w_f * 0.75)           # 이미지 영역 너비
        if x > w_img:
            return                         # 우측 UI 패널 클릭 무시

        nx, ny = x / w_img, y / h_f       # 정규화 좌표

        # ── 좌클릭 다운 ──────────────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = nx, ny
            self.selected_idx = -1
            corner_th = 0.03
            edge_th   = 0.015

            for i in reversed(range(len(self.boxes))):
                _, bx, by, bw, bh = self.boxes[i]
                x1, y1 = bx - bw / 2, by - bh / 2
                x2, y2 = bx + bw / 2, by + bh / 2

                # 우하단 모서리 → 리사이즈
                if abs(nx - x2) < corner_th and abs(ny - y2) < corner_th:
                    self.save_history()
                    self.drag_mode    = "resize"
                    self.selected_idx = i
                    return

                # 테두리 근처 → 이동
                near = (
                    (abs(nx - x1) < edge_th and y1 <= ny <= y2) or
                    (abs(nx - x2) < edge_th and y1 <= ny <= y2) or
                    (abs(ny - y1) < edge_th and x1 <= nx <= x2) or
                    (abs(ny - y2) < edge_th and x1 <= nx <= x2)
                )
                if near:
                    self.save_history()
                    self.drag_mode    = "move"
                    self.selected_idx = i
                    return

            self.drag_mode = "draw"        # 빈 공간 → 새 박스 그리기

        # ── 우클릭 → 박스 삭제 ──────────────────────────────
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i in reversed(range(len(self.boxes))):
                _, bx, by, bw, bh = self.boxes[i]
                x1, y1 = bx - bw / 2, by - bh / 2
                x2, y2 = bx + bw / 2, by + bh / 2
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    self.save_history()
                    self.boxes.pop(i)
                    self.selected_idx = -1
                    self.save_labels()
                    break

        # ── 마우스 이동 ──────────────────────────────────────
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode == "move" and self.selected_idx != -1:
                dx, dy = nx - self.start_x, ny - self.start_y
                self.boxes[self.selected_idx][1] = max(0.0, min(1.0, self.boxes[self.selected_idx][1] + dx))
                self.boxes[self.selected_idx][2] = max(0.0, min(1.0, self.boxes[self.selected_idx][2] + dy))
                self.start_x, self.start_y = nx, ny
            elif self.drag_mode == "resize" and self.selected_idx != -1:
                bx = self.boxes[self.selected_idx][1]
                by = self.boxes[self.selected_idx][2]
                self.boxes[self.selected_idx][3] = max(0.005, abs(nx - bx) * 2)
                self.boxes[self.selected_idx][4] = max(0.005, abs(ny - by) * 2)
            self.save_labels()

        # ── 좌클릭 업 → 박스 확정 ───────────────────────────
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_mode == "draw":
                w = abs(self.start_x - nx)
                h = abs(self.start_y - ny)
                if w > 0.01 and h > 0.01:
                    # pending: 숫자 키로 클래스 지정 대기
                    self.pending_box = [
                        (self.start_x + nx) / 2,
                        (self.start_y + ny) / 2,
                        w, h
                    ]
            self.drag_mode = None

    # ── UI 패널 그리기 ───────────────────────────────────────────────────────

    def draw_ui(self, canvas, w_img: int):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        mx    = w_img + 15                 # 우측 패널 시작 x
        H     = self.canvas_shape[0]

        # 파일 정보
        cv2.putText(canvas, f"{self.current_idx + 1} / {len(self.image_list)}",
                    (mx, 35), font, 0.75, (0, 255, 255), 2)
        cv2.putText(canvas, self.image_list[self.current_idx].name[:30],
                    (mx, 60), font, 0.38, (180, 180, 180), 1)
        cv2.line(canvas, (mx, 75), (self.canvas_shape[1] - 10, 75), (80, 80, 80), 1)

        # 박스 목록
        y = 100
        cv2.putText(canvas, f"BOXES: {len(self.boxes)}", (mx, y), font, 0.5, (0, 255, 255), 1)
        y += 25
        for i, (cls, bx, by, bw, bh) in enumerate(self.boxes):
            if y > H - 160:
                break
            sel    = (i == self.selected_idx)
            color  = (255, 255, 255) if sel else (160, 160, 160)
            prefix = ">>" if sel else "  "
            cv2.putText(canvas, f"{prefix}[{i}] cls={cls}  {bx:.2f},{by:.2f}",
                        (mx, y), font, 0.4, color, 1 if not sel else 2)
            y += 22

        # 조작 가이드
        guides = [
            "A/D: Prev/Next",
            ",.  : -10/+10",
            ";'  : -100/+100",
            "Drag: New Box",
            "0-9 : Set Class",
            "R-Click: Delete",
            "Arrows: Move",
            "Z/Y  : Undo/Redo",
            "Q/ESC: Quit",
        ]
        y_g = H - 10 - len(guides) * 20
        for txt in guides:
            cv2.putText(canvas, txt, (mx, y_g), font, 0.38, (110, 110, 110), 1)
            y_g += 20

    # ── 메인 루프 ────────────────────────────────────────────────────────────

    def run(self):
        if not self.initialized:
            return

        cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Label Editor", self.mouse_events)
        self.boxes = self.load_labels()

        while True:
            img_path = self.image_list[self.current_idx]
            raw_full = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                                    cv2.IMREAD_COLOR)
            if raw_full is None:
                print(f"읽기 실패: {img_path}")
                self.change_image(1)
                continue

            # 화면 크기에 맞게 리사이즈
            oh, ow = raw_full.shape[:2]
            max_w = int(self.screen_w * 0.70)
            max_h = int(self.screen_h * 0.85)
            scale = min(max_w / ow, max_h / oh, 1.0)
            nw, nh = int(ow * scale), int(oh * scale)
            raw = cv2.resize(raw_full, (nw, nh), interpolation=cv2.INTER_AREA)

            # 캔버스 생성 (우측 25% = UI 패널)
            h, w = raw.shape[:2]
            canvas_w = int(w / 0.75)
            canvas = np.full((h, canvas_w, 3), 40, dtype=np.uint8)
            canvas[:, :w] = raw
            self.canvas_shape = canvas.shape

            # 박스 그리기
            for i, (cls, bx, by, bw, bh) in enumerate(self.boxes):
                x1 = int((bx - bw / 2) * w)
                y1 = int((by - bh / 2) * h)
                x2 = int((bx + bw / 2) * w)
                y2 = int((by + bh / 2) * h)
                color = CLASS_COLORS[int(cls) % len(CLASS_COLORS)]
                thick = 2 if i != self.selected_idx else 3
                if i == self.selected_idx:
                    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                                  (255, 255, 255), 1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
                cv2.putText(canvas, str(int(cls)), (x1 + 3, y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # pending 박스 (흰 점선 느낌)
            if self.pending_box:
                px, py, pw, ph = self.pending_box
                cv2.rectangle(
                    canvas,
                    (int((px - pw / 2) * w), int((py - ph / 2) * h)),
                    (int((px + pw / 2) * w), int((py + ph / 2) * h)),
                    (255, 255, 255), 1
                )
                cv2.putText(canvas, "Press 0-9 for class",
                            (int((px - pw / 2) * w), int((py - ph / 2) * h) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            self.draw_ui(canvas, w)
            cv2.imshow("Label Editor", canvas)

            full_key = cv2.waitKeyEx(30)
            if full_key == -1:
                continue
            key = full_key & 0xFF

            # Ctrl+Z / Ctrl+Y
            if key == 26:
                self.undo(); continue
            if key == 25:
                self.redo(); continue

            if key in (ord('q'), 27):          # Q / ESC → 종료
                break
            elif key == ord('a'):
                self.change_image(-1)
            elif key == ord('d'):
                self.change_image(1)
            elif key == ord(','):
                self.change_image(-10)
            elif key == ord('.'):
                self.change_image(10)
            elif key == ord(';'):
                self.change_image(-100)
            elif key == ord("'"):
                self.change_image(100)
            elif key == ord('z'):
                self.undo()
            elif key == ord('y'):
                self.redo()
            elif ord('0') <= key <= ord('9'):
                cls = key - ord('0')
                if self.pending_box:                   # 새 박스 클래스 확정
                    self.save_history()
                    cx, cy, bw, bh = self.pending_box
                    self.boxes.append([cls, cx, cy, bw, bh])
                    self.pending_box = None
                    self.save_labels()
                elif self.selected_idx != -1:           # 선택 박스 클래스 변경
                    self.save_history()
                    self.boxes[self.selected_idx][0] = cls
                    self.save_labels()
            # 화살표 키
            elif full_key in (2490368, 65362):   # ↑
                self.move_box(0, -1)
            elif full_key in (2621440, 65364):   # ↓
                self.move_box(0, 1)
            elif full_key in (2424832, 65361):   # ←
                self.move_box(-1, 0)
            elif full_key in (2555904, 65363):   # →
                self.move_box(1, 0)

        cv2.destroyAllWindows()
        print("편집기 종료.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 진입점
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  YOLO 자동 라벨링 + 수동 수정 도구")
    print("=" * 55)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # ── 이미지 폴더 선택 ──────────────────────────────────────────────────────
    img_folder = filedialog.askdirectory(title="이미지가 들어있는 폴더 선택")
    root.destroy()
    if not img_folder:
        print("폴더 선택 취소.")
        return

    image_dir = Path(img_folder)
    label_dir = image_dir / "labels"

    images = [f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"\n이미지 폴더: {image_dir}")
    print(f"이미지 수  : {len(images)}장")
    print(f"라벨 폴더  : {label_dir}")

    # ── 자동 라벨링 여부 선택 ─────────────────────────────────────────────────
    root2 = tk.Tk()
    root2.withdraw()
    root2.attributes("-topmost", True)

    existing_labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
    do_auto = messagebox.askyesno(
        "자동 라벨링",
        f"YOLO로 자동 라벨링을 실행하시겠습니까?\n\n"
        f"• 이미지: {len(images)}장\n"
        f"• 기존 라벨: {len(existing_labels)}개\n\n"
        f"'예' → 자동 라벨링 후 편집기 실행\n"
        f"'아니오' → 편집기만 실행"
    )
    root2.destroy()

    if do_auto:
        # 모델 경로 확인
        model_path = DEFAULT_MODEL
        if not model_path.exists():
            root3 = tk.Tk()
            root3.withdraw()
            root3.attributes("-topmost", True)
            sel = filedialog.askopenfilename(
                title="YOLO 모델 파일 선택 (.pt)",
                filetypes=[("PyTorch 모델", "*.pt")]
            )
            root3.destroy()
            if not sel:
                print("모델 선택 취소 → 편집기만 실행합니다.")
                do_auto = False
            else:
                model_path = Path(sel)

        if do_auto:
            skip_existing = bool(existing_labels)
            print(f"\n[자동 라벨링] 시작 (skip_existing={skip_existing})")
            auto_label(
                image_dir   = image_dir,
                label_dir   = label_dir,
                model_path  = model_path,
                conf        = 0.3,
                target_classes = None,   # 모든 클래스 — 필요시 [0,1,2] 등 지정
                skip_existing = skip_existing,
            )

    # ── 수동 편집기 실행 ──────────────────────────────────────────────────────
    print("\n[편집기] 시작...")
    editor = LabelEditor(image_dir, label_dir)
    if editor.initialized:
        editor.run()


if __name__ == "__main__":
    main()
