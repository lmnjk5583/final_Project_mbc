# 파일 경로: C:\final_pj\merge_images.py
# 역할: GUI 다이얼로그로 날짜 폴더를 여러 개 선택 → images/ 안 이미지를 하나로 합친다.
# 실행: python merge_images.py

import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

IMAGES_SUBDIR = "frames"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def select_source_dirs() -> list[Path]:
    """폴더 선택 다이얼로그를 반복 호출해 여러 폴더를 선택받는다."""
    root = tk.Tk()
    root.withdraw()                                    # 메인 창 숨김

    selected = []
    messagebox.showinfo("폴더 선택",
                        "합칠 날짜 폴더를 하나씩 선택하세요.\n"
                        "모두 선택했으면 '취소'를 누르세요.")

    while True:
        folder = filedialog.askdirectory(title=f"날짜 폴더 선택 (현재 {len(selected)}개 선택됨 — 완료시 취소)")
        if not folder:                                 # 취소 누르면 종료
            break
        p = Path(folder)
        if p in selected:
            messagebox.showwarning("중복", f"이미 선택된 폴더입니다:\n{p.name}")
        else:
            selected.append(p)
            print(f"  + {p.name}")

    root.destroy()
    return selected


def select_output_dir() -> Path | None:
    """출력 폴더를 선택받는다."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="이미지를 합칠 출력 폴더 선택")
    root.destroy()
    return Path(folder) if folder else None


def main():
    print("=" * 50)
    print(" 이미지 합치기 도구")
    print("=" * 50)

    # ── 1) 소스 폴더 선택 ──────────────────────────────
    print("\n[1단계] 합칠 날짜 폴더 선택 (여러 개 가능)")
    source_dirs = select_source_dirs()

    if not source_dirs:
        print("선택된 폴더가 없습니다. 종료합니다.")
        return

    print(f"\n선택된 폴더 {len(source_dirs)}개:")
    for d in source_dirs:
        print(f"  - {d}")

    # ── 2) 출력 폴더 선택 ──────────────────────────────
    print("\n[2단계] 출력 폴더 선택")
    output_dir = select_output_dir()

    if not output_dir:
        print("출력 폴더가 선택되지 않았습니다. 종료합니다.")
        return

    print(f"출력 폴더: {output_dir}\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 3) 이미지 복사 ─────────────────────────────────
    total = 0

    for date_dir in source_dirs:
        images_dir = date_dir / IMAGES_SUBDIR
        if not images_dir.exists():
            print(f"  [스킵] {date_dir.name} — images/ 폴더 없음")
            continue

        files = sorted([f for f in images_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])

        if not files:
            print(f"  [스킵] {date_dir.name}/images — 이미지 없음")
            continue

        print(f"  {date_dir.name}/images — {len(files)}개 복사 중...")

        for src in files:
            dst = output_dir / f"{date_dir.name}_{src.name}"
            if dst.exists():                           # 충돌 시 번호 추가
                idx = 1
                while dst.exists():
                    dst = output_dir / f"{date_dir.name}_{src.stem}_{idx}{src.suffix}"
                    idx += 1
            shutil.copy2(src, dst)
            total += 1

    print(f"\n완료: 총 {total}개 이미지 → {output_dir}")


if __name__ == "__main__":
    main()
