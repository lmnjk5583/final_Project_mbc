# 프레임 축소(160x90 grayscale) 기반 장면/카메라 전환 감지
# 차량 방향 개략 체크 + 에지 구조 비교 유틸

import cv2
import numpy as np


class CameraSwitchDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_small = None          # 이전 프레임 축소본 (인접 프레임 차이 계산용)
        self.reference_frame = None     # 기준 배경 프레임 (장면 전환 감지용)
        self.diff_history = []          # 인접 프레임 차이값 히스토리
        self.confirm_count = 0          # 카메라 전환 감지 누적 카운트

    def reset_history(self):
        """히스토리/카운트 초기화"""
        self.diff_history.clear()
        self.confirm_count = 0
        self.reference_frame = None

    def check(self, frame, frame_num, cooldown_until):
        """
        프레임 내용의 급격한 변화로 카메라 전환/장면 전환 여부 감지
        전환 감지 시 True 반환
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              # 그레이스케일 변환
        small = cv2.resize(gray, (160, 90)).astype(np.float32)      # 축소본 생성 (160x90, 연산량 줄이기)

        # ── 재학습 직후 쿨다운 (일정 시간 동안 전환 감지 비활성) ──
        if frame_num < cooldown_until:
            # 쿨다운 중에도 diff_history는 계속 쌓아서 기준 통계 유지
            if self.prev_small is not None:
                adj_diff = np.mean(np.abs(self.prev_small - small))  # 이전 프레임과의 차이 평균
                self.diff_history.append(adj_diff)                   # 차이 히스토리에 추가
                if len(self.diff_history) > 90:                      # 최근 90개까지만 유지
                    self.diff_history.pop(0)
            self.prev_small = small  # 현재 축소본을 이전 프레임으로 저장
            return False             # 쿨다운 중에는 전환 아님

        # ── 기준 프레임이 없으면 (프로그램 시작 직후/전환 직후) ──
        if self.reference_frame is None:
            self.reference_frame = small.copy()  # 현재 프레임을 기준 프레임으로 저장
            self.prev_small = small.copy()       # 이전 프레임으로도 저장
            return False                         # 아직은 전환 아님

        # ── 바로 이전 프레임과의 차이 (프레임 간 변화량) ──
        adj_diff = np.mean(np.abs(self.prev_small - small))
        self.diff_history.append(adj_diff)       # 변화량 히스토리에 추가
        if len(self.diff_history) > 90:
            self.diff_history.pop(0)

        # ── 기준 프레임(참조 배경)과의 차이 (장면 자체가 달라졌는지) ──
        ref_diff = np.mean(np.abs(self.reference_frame - small))

        self.prev_small = small.copy()  # 현재 축소본을 다음 프레임을 위한 prev로 저장

        # 디버그용 출력: 30프레임마다 한 번씩 현재 상태 출력
        if frame_num % 30 == 0:
            avg_adj = np.mean(self.diff_history) if self.diff_history else 0
            print(f"[F:{frame_num}] adj:{adj_diff:.1f} avg:{avg_adj:.1f} ref:{ref_diff:.1f}")

        triggered = False  # 전환 의심 플래그

        # 1) 인접 프레임 차이가 '평소' 평균보다 갑자기 커졌는지 확인 (급격한 변화)
        if len(self.diff_history) >= 10:
            avg_diff = np.mean(self.diff_history[:-1])          # 마지막 하나를 제외한 최근 평균
            # 평소 평균이 충분히 크고(>2.0), 이번 값이 그 5배 이상이면 전환 의심
            if avg_diff > 2.0 and adj_diff / (avg_diff + 1e-6) > 5.0:
                triggered = True

        # 2) 기준 프레임과의 차이가 매우 크면(>50), 장면 자체가 완전히 바뀌었다고 판단
        if ref_diff > 50:
            triggered = True

        if triggered:
            self.confirm_count += 1  # 전환 의심 카운트 +1
            # 연속으로 일정 횟수 이상 의심되면 '전환 확정'
            if self.confirm_count >= self.cfg.switch_confirm_needed:
                print(f"\n📷 카메라 전환 확정! (adj:{adj_diff:.1f}, ref:{ref_diff:.1f})")
                self.confirm_count = 0         # 카운트 초기화
                self.reference_frame = None    # 기준 프레임 제거 (다음 프레임을 새 기준으로)
                self.diff_history.clear()      # diff 히스토리도 초기화
                return True                    # 전환 감지됨
        else:
            # 이번 프레임에서 의심이 안 되면 카운트 1 감소 (0 미만으로는 안 감)
            self.confirm_count = max(0, self.confirm_count - 1)

            # ref_diff가 작고, 일정 주기마다 기준 프레임을 새로 갱신 → 조명 변화 등에 적응
            if ref_diff < 20 and frame_num % 150 == 0:
                self.reference_frame = small.copy()

        return False  # 여기까지 왔으면 이번 프레임에서는 전환 아님

    def set_reference(self, frame):
        """재학습 완료 후 새 기준 프레임 설정"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90)).astype(np.float32)
        self.reference_frame = small.copy()  # 기준 프레임 설정
        self.prev_small = small.copy()       # prev_small도 초기화 (다음 check()에서 None 오류 방지)
        self.diff_history.clear()

    # ==================== 유틸: 차량 방향 개략 체크 ====================
    @staticmethod
    def check_vehicles_direction(trajectories, flow_map, cos_threshold):
        """현재 추적 중인 차량들이 flow_map과 반대로 가는지 개략적으로 체크"""

        wrong_count = 0  # 역방향으로 가는 차량 수

        for track_id, traj in trajectories.items():  # 각 차량 궤적 순회
            if len(traj) < 5:
                continue                              # 궤적이 너무 짧으면 패스

            vdx = traj[-1][0] - traj[0][0]           # 시작~끝 x 이동량
            vdy = traj[-1][1] - traj[0][1]           # 시작~끝 y 이동량
            mag = np.sqrt(vdx ** 2 + vdy ** 2)       # 벡터 크기
            if mag < 3:
                continue                              # 거의 안 움직이면 패스

            ndx, ndy = vdx / mag, vdy / mag          # 단위 방향 벡터
            cx, cy = traj[-1]                         # 현재 위치 (마지막 점)

            flow_v = flow_map.get_interpolated(cx, cy)  # 해당 위치의 flow_map 방향
            if flow_v is not None:
                cos_sim = ndx * flow_v[0] + ndy * flow_v[1]  # 코사인 유사도
                if cos_sim < cos_threshold:                   # 역방향이면 카운트
                    wrong_count += 1

        return wrong_count

    # ==================== 유틸: 에지 구조 비교 ====================
    @staticmethod
    def compare_edge_structure(edges1, edges2):
        """두 에지(윤곽선) 영상의 블록별 에지 밀도를 비교해 구조 유사도 계산"""

        blocks = 6                           # 가로/세로 블록 수
        h, w = edges1.shape                  # 영상 높이, 너비
        bh, bw = h // blocks, w // blocks    # 블록 하나의 높이, 너비
        similarities = []                    # 각 블록의 유사도 목록

        for r in range(blocks):
            for c in range(blocks):
                block1 = edges1[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]  # 첫 영상 블록
                block2 = edges2[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]  # 둘째 영상 블록
                density1 = np.mean(block1 > 0)  # 블록 내 에지 비율
                density2 = np.mean(block2 > 0)  # 블록 내 에지 비율
                if density1 + density2 > 0:
                    # 밀도 차이 기반 유사도
                    sim = 1.0 - abs(density1 - density2) / max(density1, density2, 0.01)
                else:
                    sim = 1.0  # 둘 다 에지가 없으면 유사도 1
                similarities.append(sim)  # 유사도 리스트에 추가

        return np.mean(similarities)  # 전체 평균 유사도 반환