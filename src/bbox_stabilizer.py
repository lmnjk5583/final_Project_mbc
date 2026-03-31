# 트래킹 ID별 바운딩 박스 좌표(x1,y1,x2,y2)가 프레임마다 흔들리는(jitter) 현상을
# EMA(지수 이동 평균)로 부드럽게 만들어주는 모듈

import numpy as np


class BBoxStabilizer:
    """
    [역할]
    - 트래킹 ID별 바운딩 박스 좌표(x1, y1, x2, y2)에 EMA(Exponential Moving Average, 지수이동평균)를 적용하여
      프레임마다 미세하게 튀는(bbox jitter) 좌표를 부드럽게 만들어주는 클래스

    [효과]
    - 바운딩 박스 흔들림 감소 → 중심점(cx, cy) 궤적 안정화
    - 정지 차량이 '움직이는 것처럼' 보이는 현상(phantom motion) 감소
    - 속도/방향 기반 후처리(역주행 판별 등)의 오탐 감소
    """

    def __init__(self, alpha=0.5):
        """
        [파라미터]
        alpha: 새 관측값(현재 프레임 bbox)을 반영하는 비율 (0~1)
               EMA 공식: smoothed = alpha * current + (1-alpha) * previous

               - 0.3 → 매우 부드러움(노이즈 억제 강함) / 반응 느림(실제 움직임 따라가기 느릴 수 있음)
               - 0.5 → 균형(권장)
               - 0.7 → 빠른 반응(실제 움직임 반영 빠름) / 스무딩 약함(노이즈 억제 약함)
        """
        self.alpha = alpha

        # track_id별로 "스무딩된 bbox" 좌표를 저장하는 딕셔너리
        # 예) { 5: (sx1, sy1, sx2, sy2), 12: (sx1, sy1, sx2, sy2), ... }
        self.smoothed = {}

        # track_id가 마지막으로 관측된 프레임 번호를 저장(디버그/관리용)
        # 예) { 5: 123, 12: 124, ... }
        self.last_seen = {}

    def stabilize(self, track_id, bbox, frame_num=0):
        """
        [기능]
        - 원본 bbox(현재 프레임에서 얻은 좌표)를 입력받아
          EMA로 스무딩된 bbox를 반환

        [입력]
        - track_id: ByteTrack/추적기가 부여한 고유 ID
        - bbox: (x1, y1, x2, y2) 형태의 원본 바운딩박스 좌표 (xyxy)
        - frame_num: 현재 프레임 번호(마지막 관측 프레임 기록용)

        [출력]
        - (x1, y1, x2, y2, cx, cy)
          스무딩된 bbox 좌표(정수) + 스무딩된 중심점(cx, cy) 좌표(float)
        """
        # 원본 bbox 좌표 언패킹
        x1, y1, x2, y2 = bbox

        # 이 track_id가 마지막으로 관측된 프레임 번호 기록
        self.last_seen[track_id] = frame_num

        # (1) 처음 등장한 track_id라면 스무딩 기준값이 없으므로 그대로 저장 후 반환
        if track_id not in self.smoothed:
            # float로 저장(EMA 계산 시 소수점 유지)
            self.smoothed[track_id] = (float(x1), float(y1), float(x2), float(y2))

            # 중심점 계산(원본 기준)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # 최초 프레임에서는 원본 그대로 반환
            return x1, y1, x2, y2, cx, cy

        # (2) 이전 스무딩 값이 있으면 EMA 적용
        # 이전 스무딩 bbox 좌표(이전 프레임까지 누적된 값)
        px1, py1, px2, py2 = self.smoothed[track_id]

        # alpha(현재값 반영 비율)
        a = self.alpha

        # EMA 공식 적용: 새로운 스무딩 값 = a*현재 + (1-a)*이전
        sx1 = a * x1 + (1 - a) * px1
        sy1 = a * y1 + (1 - a) * py1
        sx2 = a * x2 + (1 - a) * px2
        sy2 = a * y2 + (1 - a) * py2

        # 갱신된 스무딩 bbox를 저장(다음 프레임에서 이전값으로 사용)
        self.smoothed[track_id] = (sx1, sy1, sx2, sy2)

        # 스무딩된 bbox 기준 중심점 계산
        cx = (sx1 + sx2) / 2
        cy = (sy1 + sy2) / 2

        # bbox 좌표는 화면에 그릴 때 정수 픽셀로 쓰기 때문에 int로 변환해서 반환
        # 중심점은 추적/속도 계산 등에서 소수점이 유용할 수 있어 float 유지
        return int(sx1), int(sy1), int(sx2), int(sy2), cx, cy

    def cleanup(self, active_ids):
        """
        [기능]
        - 현재 프레임에서 관측된(active) ID만 남기고
          더 이상 나타나지 않는 ID의 스무딩 기록을 정리하여 메모리 누수 방지

        [입력]
        - active_ids: 이번 프레임에 실제로 존재하는(검출된) track_id들의 집합(set)
        """
        # smoothed 딕셔너리에 있지만 active_ids에는 없는 ID들은 "죽은 트랙"으로 판단
        dead = [tid for tid in self.smoothed if tid not in active_ids]

        # 죽은 트랙 ID들에 대해 저장된 정보 삭제
        for tid in dead:
            del self.smoothed[tid]          # 스무딩 bbox 제거
            self.last_seen.pop(tid, None)   # last_seen에서도 제거(없어도 에러 안 나게 pop)