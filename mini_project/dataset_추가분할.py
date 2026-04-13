# 파일 경로: C:\final_pj\mini_project\dataset_추가분할.py
# 역할: 새로 라벨링한 이미지를 기존 train/val/test 폴더에 추가 분할
#        - 기존 split 폴더에 이미 있는 파일은 건너뜀 (중복 방지)
#        - 새 파일만 같은 비율(70/15/15)로 셔플 후 추가
# 실행: python dataset_추가분할.py

import shutil
import random
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

# ==================== 설정 ====================
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ==================== 폴더 선택 ====================
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

messagebox.showinfo("안내",
    "1단계: 새 이미지가 있는 소스 폴더 선택\n"
    "(images/ 와 labels/ 가 들어있는 상위 폴더)")
src_root = Path(filedialog.askdirectory(title="소스 폴더 선택 (images/ + labels/ 포함)"))

if not src_root or not src_root.exists():
    print("소스 폴더 선택 취소.")
    root.destroy()
    exit()

messagebox.showinfo("안내",
    "2단계: 기존 train/val/test 폴더가 있는 데이터셋 루트 선택")
dst_root = Path(filedialog.askdirectory(title="데이터셋 루트 선택 (train/val/test 폴더 있는 곳)"))

root.destroy()

if not dst_root or not dst_root.exists():
    print("데이터셋 루트 선택 취소.")
    exit()

# ==================== 경로 확인 ====================
src_img_dir = src_root / "images"
src_lbl_dir = src_root / "labels"

print(f"\n소스 폴더  : {src_root}")
print(f"데이터셋   : {dst_root}")

if not src_img_dir.exists() or not src_lbl_dir.exists():
    print(f"❌ 소스 폴더 안에 images/ 또는 labels/ 가 없습니다.")
    exit()

# ==================== 기존 split에 있는 파일명 수집 ====================
existing_stems = set()
for split in ['train', 'val', 'test']:
    img_dir = dst_root / split / 'images'
    if img_dir.exists():
        for f in img_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                existing_stems.add(f.stem)

print(f"\n기존 split 파일 수: {len(existing_stems)}개")

# ==================== 새 파일 필터링 ====================
# 소스에서 라벨이 있고 기존 split에 없는 이미지만 선택
src_label_stems = {
    f.stem for f in src_lbl_dir.iterdir()
    if f.is_file() and f.suffix == '.txt'
}

new_images = sorted([
    f for f in src_img_dir.iterdir()
    if f.is_file()
    and f.suffix.lower() in IMG_EXTS
    and f.stem in src_label_stems      # 라벨 있는 것만
    and f.stem not in existing_stems   # 기존 split에 없는 것만
])

print(f"소스 이미지 수     : {len(list(f for f in src_img_dir.iterdir() if f.suffix.lower() in IMG_EXTS))}개")
print(f"새로 추가할 이미지 : {len(new_images)}개")

if len(new_images) == 0:
    print("\n추가할 새 이미지가 없습니다. 종료합니다.")
    exit()

# ==================== 셔플 & 분할 ====================
random.seed(RANDOM_SEED)
random.shuffle(new_images)

total     = len(new_images)
train_end = int(total * TRAIN_RATIO)
val_end   = int(total * (TRAIN_RATIO + VAL_RATIO))

splits = {
    'train': new_images[:train_end],
    'val':   new_images[train_end:val_end],
    'test':  new_images[val_end:],
}

print(f"\n분할 결과 (새 이미지 {total}장):")
print(f"  train : {len(splits['train'])}장")
print(f"  val   : {len(splits['val'])}장")
print(f"  test  : {len(splits['test'])}장")

# ==================== split 폴더 생성 ====================
for split in ['train', 'val', 'test']:
    (dst_root / split / 'images').mkdir(parents=True, exist_ok=True)
    (dst_root / split / 'labels').mkdir(parents=True, exist_ok=True)

# ==================== 복사 ====================
print("\n📦 복사 시작...")
total_ok   = 0
total_fail = 0

for split_name, img_list in splits.items():
    ok = 0
    fail = 0
    for img in img_list:
        try:
            # 이미지 복사
            shutil.copy2(str(img),
                         str(dst_root / split_name / 'images' / img.name))
            # 라벨 복사
            lbl_src = src_lbl_dir / (img.stem + '.txt')
            shutil.copy2(str(lbl_src),
                         str(dst_root / split_name / 'labels' / (img.stem + '.txt')))
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  ⚠️ 실패: {img.name} → {e}")

    print(f"  {split_name}: {ok}장 추가 완료" + (f", {fail}장 실패" if fail else ""))
    total_ok   += ok
    total_fail += fail

# ==================== 최종 확인 ====================
print("\n📊 최종 split 현황:")
for split in ['train', 'val', 'test']:
    img_cnt = len([f for f in (dst_root / split / 'images').iterdir()
                   if f.suffix.lower() in IMG_EXTS])
    lbl_cnt = len(list((dst_root / split / 'labels').glob('*.txt')))
    match = "✅" if img_cnt == lbl_cnt else "⚠️ 불일치"
    print(f"  {split:5s}: 이미지 {img_cnt:4d}개 / 라벨 {lbl_cnt:4d}개 {match}")

print(f"\n✅ 완료: {total_ok}장 추가" + (f" ({total_fail}장 실패)" if total_fail else ""))
