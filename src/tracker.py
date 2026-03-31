# YOLO + ByteTrack 래퍼
# model.track() 한 번 호출로 결과를 정리된 dict 리스트로 반환

from ultralytics import YOLO


class YoloTracker:
    def __init__(self, model_path, conf, target_classes=None):
        self.model = YOLO(str(model_path))   # YOLO 모델 로드
        self.conf = conf                     # 객체 검출 신뢰도(confidence) 임계값
        self.target_classes = target_classes  # 추적할 클래스 인덱스 리스트 (None이면 모든 클래스)

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