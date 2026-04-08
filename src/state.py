# 프레임마다 변하는 런타임 상태를 한 곳에 모아 관리하는 클래스

from collections import defaultdict


class DetectorState:
    def __init__(self):
        # ==================== 기본 상태 ====================
        self.frame_num = 0          # 현재까지 처리한 프레임 번호
        self.frame_w = 0            # 영상 너비
        self.frame_h = 0            # 영상 높이
        self.video_fps = 30.0       # run()에서 실제 FPS로 갱신

        # ==================== 학습/재학습 플래그 ====================
        self.is_learning = True         # 현재 학습 모드인지 여부 (True면 학습 중)
        self.relearning = False        # 재학습 모드 여부 (초기 학습은 is_learning으로 관리)
        self.relearn_start_frame = 0    # 재학습 시작 프레임 번호
        self.cooldown_until = 0         # 이 프레임까지 카메라 전환 감지 비활성 (쿨다운 끝나는 프레임)

        # ==================== 상태 변수(트래킹/역주행 카운트) ====================
        self.trajectories = defaultdict(list)     # 각 ID별 궤적 저장 {track_id: [(cx, cy), ...]}
        self.wrong_way_count = defaultdict(int)   # 각 ID별 역주행 의심 누적 횟수
        self.wrong_way_ids = set()                # 역주행으로 확정된 차량 ID들
        self.last_cos_values = defaultdict(list)  # 각 ID별 최근 내적(cos) 값 목록 (디버그용)

        # ==================== ID 매핑(역주행 차량 라벨 관리) ====================
        self.wrong_way_last_pos = {}   # 역주행 차량 마지막 위치 {track_id: (cx, cy, frame_num)}
        self.display_id_map = {}       # 표시용 라벨 매핑 {현재 ID: 'W1' 같은 라벨}
        self.next_wrong_way_label = 1  # 새 역주행 차량에 부여할 다음 라벨 번호 (W1, W2, ...)

        # ==================== 탐지 소요시간 통계 ====================
        self.first_seen_frame = {}     # {track_id: 처음 등장한 프레임 번호}
        self.first_suspect_frame = {}  # {track_id: 처음 역주행 의심 시작 프레임}
        self.detection_stats = {}      # {display_label: {... 통계 정보 ...}}

        # ==================== 트랙 정리용 ====================
        self._stale_counter = defaultdict(int)  # 각 ID가 "몇 프레임째 안 보이는 중인지" 세는 카운터

        # ==================== 방향 급변 필터용 ====================
        self.last_velocity = {}      # {track_id: (ndx, ndy)} 직전 프레임 단위 방향 벡터

        # ==================== 정상 주행 이력 필터용 ====================
        self.last_correct_frame = {}  # {track_id: frame_num} 마지막으로 정상 주행 판정된 프레임

        # ==================== 방향 급변 가드용 ====================
        self.stable_velocity = {}        # {track_id: (ndx, ndy)} 마지막 정상 투표 시 방향 (급변 기준점)
        self.direction_change_frame = {} # {track_id: frame_num} stable_velocity 대비 급변 감지된 시점
        self.direction_was_stable = {}   # {track_id: bool} 직전 프레임 방향 안정 여부 (edge 감지용)


    def reset_for_relearn(self):
        """카메라 전환 감지 후 추적/역주행 관련 상태 초기화, 재학습 모드 진입"""

        print("=" * 50)
        print("🔄 flow_map 초기화 → 재학습 시작")
        print("=" * 50 + "\n")

        self.wrong_way_ids.clear()          # 역주행 ID 목록 초기화
        self.wrong_way_count.clear()        # 역주행 카운트 초기화
        self.wrong_way_last_pos.clear()     # 역주행 마지막 위치 초기화
        self.display_id_map.clear()         # 표시 라벨 매핑 초기화
        self.trajectories.clear()           # 궤적 정보 초기화
        self.last_cos_values.clear()        # 내적값 히스토리 초기화
        self._stale_counter.clear()         # 스테일 카운터 초기화

        self.last_velocity.clear()          # 방향 벡터 이력 초기화
        self.last_correct_frame.clear()    # 정상 판정 프레임 초기화
        self.stable_velocity.clear()       # 안정 방향 기준점 초기화
        self.direction_change_frame.clear()  # 급변 감지 시점 초기화
        self.direction_was_stable.clear()  # 안정 상태 플래그 초기화

        self.relearning = True              # 재학습 모드 진입
        self.relearn_start_frame = self.frame_num  # 재학습 시작 프레임 기록