import os
import shutil
import random
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


def split_dataset(DATA_ROOT, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """images/, labels/ → train/val/test 자동 분할"""

    src_img = DATA_ROOT / "images"
    src_lab = DATA_ROOT / "labels"

    # 경로 존재 확인
    if not src_img.exists():
        print(f"❌ images 폴더 없음: {src_img}")
        return False
    if not src_lab.exists():
        print(f"❌ labels 폴더 없음: {src_lab}")
        return False

    # 이미 분할된 경우 스킵
    if (DATA_ROOT / "train" / "images").exists():
        existing = len(list((DATA_ROOT / "train" / "images").glob("*.*")))
        if existing > 0:
            print(f"✅ 이미 분할됨 (train: {existing}개). 스킵합니다.")
            return True

    # 이미지 확장자 필터링
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_dict = {}
    for f in os.listdir(str(src_img)):                      # ✅ 수정: os.listdir(str())
        ext = os.path.splitext(f)[1].lower()
        stem = os.path.splitext(f)[0]
        if ext in img_exts:
            img_dict[stem] = f

    lab_stems = set()
    for f in os.listdir(str(src_lab)):                      # ✅ 수정: os.listdir(str())
        if f.endswith('.txt'):
            lab_stems.add(os.path.splitext(f)[0])

    # 매칭
    matched = sorted([s for s in img_dict if s in lab_stems])
    print(f"🔍 이미지: {len(img_dict)}, 라벨: {len(lab_stems)}, 매칭: {len(matched)}")

    if len(matched) == 0:
        print("❌ 매칭되는 이미지-라벨 쌍이 없습니다!")
        print(f"  이미지 샘플: {list(img_dict.keys())[:5]}")
        print(f"  라벨 샘플:  {list(lab_stems)[:5]}")
        return False

    # 셔플 & 분할
    random.seed(42)
    random.shuffle(matched)

    n = len(matched)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))

    splits = {
        'train': matched[:t_end],
        'val':   matched[t_end:v_end],
        'test':  matched[v_end:]
    }

    # 복사
    for split_name, stem_list in splits.items():
        img_dst = DATA_ROOT / split_name / "images"
        lab_dst = DATA_ROOT / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lab_dst.mkdir(parents=True, exist_ok=True)

        success = 0
        for stem in stem_list:
            try:
                img_file = img_dict[stem]
                # ✅ 수정: str()로 감싸서 한글+& 경로 문제 해결
                shutil.copy2(
                    str(src_img / img_file),
                    str(img_dst / img_file)
                )
                shutil.copy2(
                    str(src_lab / f"{stem}.txt"),
                    str(lab_dst / f"{stem}.txt")
                )
                success += 1
            except Exception as e:
                if success == 0:  # 첫 에러만 출력
                    print(f"    ⚠️ 복사 실패: {stem} → {e}")

        print(f"  📁 {split_name}: {success}/{len(stem_list)}개 복사 완료")

    print(f"\n✅ 데이터 분할 완료!\n")
    return True


def print_results(results, title="평가 결과"):
    """성능 출력"""
    print(f"\n{'='*60}")
    print(f"   📊 {title}")
    print(f"{'='*60}")
    box = results.box
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*36}")
    print(f"  {'mAP50':<25} {box.map50:>10.4f}")
    print(f"  {'mAP50-95':<25} {box.map:>10.4f}")
    print(f"  {'Precision':<25} {box.mp:>10.4f}")
    print(f"  {'Recall':<25} {box.mr:>10.4f}")
    f1 = 2 * (box.mp * box.mr) / (box.mp + box.mr + 1e-8)
    print(f"  {'F1 Score':<25} {f1:>10.4f}")

    if hasattr(results, 'names') and results.names:
        try:
            print(f"\n  📋 클래스별:")
            print(f"  {'Class':<15} {'mAP50':>8} {'mAP50-95':>10} {'P':>7} {'R':>7}")
            print(f"  {'-'*47}")
            for i, name in results.names.items():
                print(f"  {name:<15} {box.ap50()[i]:>8.4f} {box.ap()[i]:>10.4f} "
                      f"{box.p[i]:>7.4f} {box.r[i]:>7.4f}")
        except:
            pass
    print(f"{'='*60}\n")


def main():
    DATA_ROOT    = Path(r"N:\개인\대원&수빈\최종 프로젝트\capture4")
    PROJECT_ROOT = Path(r"N:\개인\대원&수빈\최종 프로젝트")

    # ✅ 수정: GPU 자동 감지
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'

    print(f"🖥️ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   CPU 모드로 실행됩니다")
    print(f"   Device: {DEVICE}")

    # ============================================================
    # 1단계: 데이터 분할
    # ============================================================
    print("\n📦 데이터 분할 시작...")
    if not split_dataset(DATA_ROOT, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        print("❌ 분할 실패! 종료합니다.")
        return

    # 수량 확인
    train_cnt = len(list((DATA_ROOT / "train" / "images").glob("*.*")))
    val_cnt   = len(list((DATA_ROOT / "val"   / "images").glob("*.*")))
    test_cnt  = len(list((DATA_ROOT / "test"  / "images").glob("*.*")))
    print(f"📊 Train: {train_cnt} / Val: {val_cnt} / Test: {test_cnt}")

    if train_cnt == 0:
        print("❌ Train 데이터가 없습니다!")
        return

    # ============================================================
    # 2단계: data.yaml 생성
    # ============================================================
    data_yaml = {
        'path':  str(DATA_ROOT),
        'train': 'train/images',
        'val':   'val/images',
        'test':  'test/images',
        'nc': 1,
        'names': ['vehicle']
    }
    yaml_path = DATA_ROOT / 'data.yaml'
    with open(str(yaml_path), 'w', encoding='utf-8') as f:   # ✅ str() 감싸기
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ data.yaml 생성: {yaml_path}")

    # ============================================================
    # 3단계: 학습
    # ============================================================
    print("\n" + "=" * 60)
    print("   🚀 YOLO11n 학습 시작")
    print("=" * 60)

    model = YOLO("yolo11n.pt")
    train_results = model.train(
        data=str(yaml_path),
        # ---- 기본 설정 ----
        epochs=150,                 # 최대 에폭 수 (최대 200까지, patience로 조기 종료)
        imgsz=640,                  # 학습/검증에 사용할 입력 이미지 크기 (640x640)
        batch=16,                    # 배치 크기 (한 번에 올리는 이미지 수)
        patience=20,                # 30에폭 동안 성능 향상 없으면 조기 종료
        device=DEVICE,              # GPU 번호 (자동 감지된 DEVICE 사용)
        workers=2,                  # DataLoader 병렬 작업자 수 (RAM/OOM 고려해서 2)
        exist_ok=True,              # 같은 이름의 run 폴더가 있어도 덮어쓰기 허용
        seed=42,                    # 랜덤 시드 고정 (재현성 확보)

        # ---- Optimizer / 학습 스케줄 ----
        optimizer="AdamW",          # 옵티마이저 종류 (AdamW 사용)
        lr0=0.001,                  # 초기 학습률
        lrf=0.01,                   # 최종 학습률 비율 (lr0 * lrf까지 감소)
        cos_lr=True,                # 코사인 러닝레이트 스케줄 사용 여부
        warmup_epochs=5,            # 초반 5에폭은 천천히 LR 올리면서 안정화

        # ---- 증강(Augmentation) ----
        close_mosaic=20,            # 마지막 20에폭 동안은 Mosaic 끄기 (정제 학습)
        mixup=0.1,                  # Mixup 증강 비율 (이미지/라벨 섞기)
        copy_paste=0.1,             # Copy-Paste 증강 비율 (객체를 잘라 다른 이미지에 붙이기)
        degrees=0.0,                # 회전 각도 범위 (CCTV 고정이므로 0)
        perspective=0.0,            # 원근 변환 강도 (CCTV 자체가 원근이라 추가 변환 X)
        # scale=0.5,
        # erasing=0.1,

        project=str(PROJECT_ROOT / "runs"),
        name="yolo11n_v6",
    )

    # ============================================================
    # 4단계: Validation 성능
    # ============================================================
    print_results(train_results, title="학습 완료 - Validation 성능")

    # ============================================================
    # 5단계: Test 평가
    # ============================================================
    best_pt = PROJECT_ROOT / "runs" / "yolo11n_v6" / "weights" / "best.pt"

    if not best_pt.exists():
        candidates = list((PROJECT_ROOT / "runs").rglob("best.pt"))
        if candidates:
            best_pt = candidates[-1]
            print(f"⚠️  best.pt 자동 탐색: {best_pt}")
        else:
            print("❌ best.pt 없음!")
            return

    print(f"🏆 Test 평가 모델: {best_pt}")
    test_results = YOLO(str(best_pt)).val(
        data=str(yaml_path),
        split='test',
        device=DEVICE               # ✅ 수정: device=0 → DEVICE
    )
    print_results(test_results, title="Test Set 최종 성능")

    # ============================================================
    # 6단계: 최종 비교표
    # ============================================================
    val_f1  = 2 * (train_results.box.mp * train_results.box.mr) / (train_results.box.mp + train_results.box.mr + 1e-8)
    test_f1 = 2 * (test_results.box.mp * test_results.box.mr) / (test_results.box.mp + test_results.box.mr + 1e-8)

    print("=" * 60)
    print("   📋 최종 성능 비교 (Val vs Test)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Validation':>12} {'Test':>12}")
    print(f"  {'-'*44}")
    print(f"  {'mAP50':<20} {train_results.box.map50:>12.4f} {test_results.box.map50:>12.4f}")
    print(f"  {'mAP50-95':<20} {train_results.box.map:>12.4f} {test_results.box.map:>12.4f}")
    print(f"  {'Precision':<20} {train_results.box.mp:>12.4f} {test_results.box.mp:>12.4f}")
    print(f"  {'Recall':<20} {train_results.box.mr:>12.4f} {test_results.box.mr:>12.4f}")
    print(f"  {'F1 Score':<20} {val_f1:>12.4f} {test_f1:>12.4f}")
    print("=" * 60)
    print("\n✅ 완료! 🎉")


if __name__ == '__main__':
    main()