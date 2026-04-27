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
        self.last_adj_diff = 0.0        # 직전 프레임 인접 diff (안정 판정용 — 외부에서 읽음)

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
        self.last_adj_diff = adj_diff            # 외부 안정 판정용으로 노출
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

        # 2) 기준 프레임과의 차이가 매우 크면(>30), 장면 자체가 완전히 바뀌었다고 판단
        if ref_diff > 30:
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

