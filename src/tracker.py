# YOLO + ByteTrack 래퍼
# model.track() 한 번 호출로 결과를 정리된 dict 리스트로 반환

import cv2
import numpy as np
from ultralytics import YOLO


class YoloTracker:
    def __init__(self, model_path, conf, target_classes=None, night_enhance: bool = True):
        self.model = YOLO(str(model_path))   # YOLO 모델 로드
        self.conf = conf                     # 객체 검출 신뢰도(confidence) 임계값
        self.target_classes = target_classes  # 추적할 클래스 인덱스 리스트 (None이면 모든 클래스)
        self.night_enhance = night_enhance    # 야간 저조도 전처리 여부

        # CLAHE: 대비 제한 적응형 히스토그램 평활화 — 야간 저조도·노이즈 보정
        # clipLimit=2.0: 과도한 노이즈 증폭 방지 / tileGridSize: 8×8 블록 단위 적용
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 모델에서 클래스 이름 자동 로드 (ex: {0: 'car', 1: 'bus', ...})
        self.class_names = getattr(self.model, "names", {})

    def track(self, frame):
        """
        YOLO 추적 실행 후 결과를 정리된 dict 리스트로 반환

        Returns:
            list of dict: [{id, x1, y1, x2, y2, cx, cy}, ...]
            빈 리스트면 감지된 객체 없음
        """
        # YOLO 추적 호출
        track_kwargs = {
            "tracker": "bytetrack.yaml",   # ByteTrack 설정
            "persist": True,               # 트랙 ID 유지
            "verbose": False,              # YOLO 로그 최소화
            "conf": self.conf,             # 신뢰도 임계값
        }
        # target_classes가 지정되어 있으면 해당 클래스만 추적
        if self.target_classes is not None:
            track_kwargs["classes"] = self.target_classes

        # ── 야간 저조도 전처리 (night_enhance=True 시) ───────────────
        # 평균 밝기가 80 이하일 때만 CLAHE 적용 — 낮에는 건너뜀
        if self.night_enhance:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 80:                         # 저조도 판단 (0~255 중 80 이하)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)   # LAB 색공간 변환
                l, a, b = cv2.split(lab)                       # L(밝기) 채널 분리
                l = self._clahe.apply(l)                       # L 채널에만 CLAHE 적용
                lab = cv2.merge((l, a, b))                     # 채널 병합
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)   # BGR로 복원

        results = self.model.track(frame, **track_kwargs)  # YOLO 추적 실행
        boxes = results[0].boxes

        if boxes.id is None:  # 감지된 객체(트랙)가 없으면
            return []

        tracks = []
        for i, tid in enumerate(boxes.id.int().tolist()):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()  # 바운딩 박스 좌상/우하 좌표
            tracks.append({
                "id": tid,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2,  # 중심 좌표 x
                "cy": (y1 + y2) / 2,  # 중심 좌표 y
            })
        return tracks