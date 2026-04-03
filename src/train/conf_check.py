from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

IMAGE_PATH = r"C:\Users\User\Pictures\Screenshots\스크린샷 2026-04-02 100700.png"
CONF = 0.25

models = {
    "YOLO11n": YOLO(r"N:\개인\대원&수빈\최종 프로젝트\runs\yolo11n_v1\weights\best.pt"),
    "YOLO26n": YOLO(r"N:\개인\대원&수빈\최종 프로젝트\runs\yolo26n_v1\weights\best.pt"),
}

print(f"{'='*65}")
print(f"  📷 이미지: {Path(IMAGE_PATH).name}")
print(f"{'='*65}")

annotated_imgs = []

for name, model in models.items():
    results = model(IMAGE_PATH, conf=CONF)
    boxes = results[0].boxes
    confs = [float(b.conf[0]) for b in boxes]
    
    print(f"\n  🏷️ {name}: {len(boxes)}개 탐지")
    if confs:
        print(f"     평균 신뢰도: {sum(confs)/len(confs):.4f}")
        print(f"     최대: {max(confs):.4f} / 최소: {min(confs):.4f}")
        for i, box in enumerate(boxes):
            c = float(box.conf[0])
            n = results[0].names[int(box.cls[0])]
            print(f"     [{i+1}] {n}: {c:.4f} ({c*100:.1f}%)")
    
    # 이미지에 모델 이름 표시
    img = results[0].plot()
    cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    annotated_imgs.append(img)

# 나란히 비교 이미지 저장
if len(annotated_imgs) == 2:
    h1, w1 = annotated_imgs[0].shape[:2]
    h2, w2 = annotated_imgs[1].shape[:2]
    h = max(h1, h2)
    
    img1 = cv2.copyMakeBorder(annotated_imgs[0], 0, h-h1, 0, 0, cv2.BORDER_CONSTANT)
    img2 = cv2.copyMakeBorder(annotated_imgs[1], 0, h-h2, 0, 0, cv2.BORDER_CONSTANT)
    
    combined = np.hstack([img1, img2])
    save_path = str(Path(IMAGE_PATH).parent / "comparison.jpg")
    cv2.imwrite(save_path, combined)
    print(f"\n✅ 비교 이미지 저장: {save_path}")