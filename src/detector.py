# 파일 경로: 최종 프로젝트/src/detector.py
# 역할: 모든 모듈을 조립하고 run() 루프를 실행하는 메인 오케스트레이터.
#        TrafficAnalyzer + PassageTracker 연동 추가 (정체 탐지 결과 매 프레임 갱신)

import cv2                                          # OpenCV — 영상 입출력·시각화
import numpy as np                                  # 수치 계산
import time                                         # FPS 측정용 타이머
from copy import copy                               # baseline 방향별 독립 복사용

from .config import DetectorConfig                  # 모든 파라미터가 담긴 설정 클래스
from .state import DetectorState                    # 프레임 번호·궤적·역주행 카운트 등 런타임 상태
from .flow_map import FlowMap                       # 15×15 그리드 정상 흐름 벡터 학습/비교
from .tracker import YoloTracker                    # YOLO 검출 + ByteTrack 추적
from .judge import WrongWayJudge                    # 코사인 유사도 + 투표/히스테리시스 역주행 판정
from .id_manager import IDManager                   # W라벨 관리 + occlusion 재매칭 + 오래된 트랙 정리
from .camera_switch import CameraSwitchDetector     # 장면/카메라 전환 감지 (grayscale diff)
from .visualizer import Visualizer                  # 시각화(박스/궤적/패널/디버그)
from .logger import CSVLogger                       # 프레임/트랙/이벤트 CSV 로그 저장
from .bbox_stabilizer import BBoxStabilizer         # 바운딩박스 EMA 안정화
from .traffic_analyzer import TrafficAnalyzer, CongestionPredictor  # 정체 탐지 + 단기 예측
from passage_tracker import PassageTracker          # 차량 진입·퇴장 기록 + baseline 산출
from baseline_stats import BaselineStats           # fallback baseline 생성용

# GRUModule: PyTorch 없는 환경에서도 동작하도록 try/except
try:
    from gru_module import GRUModule                # Phase 2 GRU 예측 모듈
    _GRU_AVAILABLE = True                           # GRU 사용 가능 플래그
except ImportError:                                 # gru_module.py 없거나 torch 없으면
    _GRU_AVAILABLE = False                          # fallback 모드


class Detector:
    """역주행 탐지 + 정체 탐지를 수행하는 메인 오케스트레이터."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg                                              # 설정 객체 저장

        # ── 런타임 상태(state) + 모듈 생성 ──────────────────────────────
        self.state = DetectorState()                                # 프레임 번호, 궤적, 역주행 카운트 등
        self.flow = FlowMap(cfg.grid_size, cfg.alpha, cfg.min_samples)  # 정상 흐름 벡터 그리드
        self.tracker = YoloTracker(cfg.model_path, cfg.conf, cfg.target_classes)  # YOLO+ByteTrack
        self.judge = WrongWayJudge(cfg, self.flow, self.state)      # 역주행 판정기
        self.idm = IDManager(cfg, self.flow, self.state)            # ID 관리 + 재매칭
        self.switch = CameraSwitchDetector(cfg)                     # 카메라 전환 감지기
        self.vis = Visualizer(cfg, self.state, self.flow)           # 시각화 모듈
        self.logger = CSVLogger(cfg.log_dir) if cfg.log_dir else None  # CSV 로거 (log_dir 없으면 None)
        self.bbox_stab = BBoxStabilizer(alpha=0.5)                  # bbox EMA 안정화기
        # ── 메인 PassageTracker (학습 시 on_entry/on_exit 전담) ────────
        self.passage_tracker = PassageTracker(cfg, self.state)      # 차량 진입·퇴장 기록 관리

        # ── 방향별 PassageTracker (탐지 시 record_frame_stats 전담) ──
        self.passage_tracker_a = PassageTracker(cfg, self.state)    # A방향 PT (state 공유)
        self.passage_tracker_b = PassageTracker(cfg, self.state)    # B방향 PT (state 공유)

        # ── 방향별 GRU/TrafficAnalyzer/Predictor (run()에서 초기화) ──
        # frame 크기(fw, fh)와 fps는 run()에서 영상을 열어야 확정되므로
        # __init__ 시점에서는 None으로 두고 run() 진입 직후 초기화한다.
        self.gru_module_a = None                                    # A방향 GRU (run()에서 초기화)
        self.gru_module_b = None                                    # B방향 GRU (run()에서 초기화)
        self.traffic_analyzer_a = None                              # A방향 정체 탐지 (run()에서 초기화)
        self.traffic_analyzer_b = None                              # B방향 정체 탐지 (run()에서 초기화)
        self.predictor_a = None                                     # A방향 정체 예측 (run()에서 초기화)
        self.predictor_b = None                                     # B방향 정체 예측 (run()에서 초기화)

        # ── 방향 분류 기준 벡터 + 차량별 방향 매핑 ────────────────────
        self._ref_direction = None                                  # 전역 기준 방향 벡터 (학습 완료 시 계산)
        self._track_direction = {}                                  # {tid: 'a' or 'b'} 차량별 방향
        self._dir_label_a = "상행"                                  # A방향 표시 레이블 (기본값)
        self._dir_label_b = "하행"                                  # B방향 표시 레이블 (기본값)

        # ── flow_map 로드 (탐지 전용이라면 필수) ────────────────────
        self._saved_baseline = None                                 # 파일에서 불러온 baseline (없으면 None)

        if cfg.flow_map_path:                                       # flow_map 경로가 설정되어 있으면
            if not cfg.detect_only:                                 # 학습 모드이면 기존 파일 무시하고 재학습
                self.state.is_learning = True                       # 학습 모드로 전환
                print("detect_only=False → 기존 flow_map 무시, 처음부터 학습 시작")
            else:                                                   # 탐지 전용이면 기존 파일 로드
                loaded, self._saved_baseline = self.flow.load(     # 튜플 언패킹: (성공여부, baseline)
                    cfg.flow_map_path
                )
                if not loaded:                                      # 탐지 전용인데 로드 실패
                    raise FileNotFoundError(                        # 즉시 예외 → 잘못된 실험 방지
                        f"detect_only=True 인데 flow_map이 없습니다: {cfg.flow_map_path}"
                    )
                self.state.is_learning = False                      # 로드 성공 → 학습 모드 해제
        else:                                                       # flow_map_path가 None
            if cfg.detect_only:                                     # 탐지 전용인데 경로 자체가 없음
                raise ValueError(                                   # 설정 오류 → 예외
                    "detect_only=True 인데 flow_map_path가 None 입니다."
                )
            else:                                                   # 학습 모드인데 경로 없음
                self.state.is_learning = True                       # 학습 모드로 전환
                print("flow_map_path 없음 → 학습 모드 시작 (저장 안 됨)")

        # ── 결과 저장 폴더 ──
        if cfg.result_dir:                                          # 결과 저장 경로가 있으면
            cfg.result_dir.mkdir(parents=True, exist_ok=True)       # 폴더 생성 (이미 있으면 무시)

    # ==================== 방향 분류 기준 벡터 계산 ====================
    def _compute_ref_direction(self):
        """flow_map에서 가장 샘플이 많은 셀의 흐름 벡터를 기준 방향으로 설정한다."""
        grid = self.flow                                          # FlowMap 참조
        best_r, best_c = 0, 0                                    # 최다 샘플 셀 좌표
        best_count = 0                                            # 최다 샘플 수
        for r in range(grid.grid_size):                           # 행 순회
            for c in range(grid.grid_size):                       # 열 순회
                cnt = grid.count[r, c]                            # 해당 셀 샘플 수
                if cnt > best_count:                              # 더 많은 샘플 발견
                    best_count = cnt                              # 갱신
                    best_r, best_c = r, c                         # 좌표 갱신
        vx = float(grid.flow[best_r, best_c, 0])                  # 최다 셀 흐름 x (flow[r,c,0])
        vy = float(grid.flow[best_r, best_c, 1])                  # 최다 셀 흐름 y (flow[r,c,1])
        mag = np.sqrt(vx**2 + vy**2)                              # 벡터 크기
        if mag > 1e-6:                                            # 유효한 벡터이면
            self._ref_direction = (vx / mag, vy / mag)            # 단위 벡터로 저장
        else:                                                     # 무효 (빈 flow_map)
            self._ref_direction = (1.0, 0.0)                      # fallback: 오른쪽

        # ── UP/DOWN 레이블 자동 판별 ────────────────────────────────
        # 카메라 좌표계: 이미지 위 = y 감소(vy < 0) = 화면 상 위로 이동 = UP
        #               이미지 아래 = y 증가(vy > 0) = 화면 상 아래로 이동 = DOWN
        # vy 부호만으로 판별 — 카메라 설치 방향 무관하게 항상 동일하게 적용
        ref_vy = self._ref_direction[1]                           # A방향 y성분
        if ref_vy < 0:                                            # A가 이미지 위쪽으로 이동 → UP
            self._dir_label_a = "UP"                              # A = UP (화면 위 방향)
            self._dir_label_b = "DOWN"                            # B = DOWN (화면 아래 방향)
        else:                                                     # A가 이미지 아래쪽으로 이동 → DOWN
            self._dir_label_a = "DOWN"                            # A = DOWN (화면 아래 방향)
            self._dir_label_b = "UP"                              # B = UP (화면 위 방향)

        print(f"🧭 기준 방향: ({self._ref_direction[0]:.3f}, {self._ref_direction[1]:.3f})"
              f" [셀({best_r},{best_c}), 샘플={best_count}]"
              f" → A={self._dir_label_a}, B={self._dir_label_b}")

    # ==================== 차량 방향 분류 ====================
    def _classify_direction(self, fx, fy):
        """footpoint 위치의 flow_map 셀 방향과 기준 방향을 비교해 'a' 또는 'b' 반환."""
        if self._ref_direction is None:                           # 기준 방향 미설정
            return 'a'                                            # 기본값: A방향
        flow_v = self.flow.get_interpolated(fx, fy)               # 해당 위치 흐름 벡터
        if flow_v is None:                                        # 벡터 없으면 (경계·flow 없음)
            return 'a'                                            # fallback: A방향
        ref_x, ref_y = self._ref_direction                        # 기준 방향 분해
        cos_val = flow_v[0] * ref_x + flow_v[1] * ref_y          # 코사인 유사도
        return 'a' if cos_val >= self.cfg.lane_cos_threshold else 'b'  # 임계값 기준 분류

    # ==================== 기본 유틸 ====================
    def _get_next_filename(self, base="results", ext=".mp4"):
        """결과 파일명이 겹치지 않도록 뒤에 번호를 붙여서 새 파일명 생성"""
        idx = 1                                                     # 시작 번호
        while True:                                                 # 사용 가능한 번호 찾을 때까지
            p = self.cfg.result_dir / f"{base}_{idx}{ext}"          # 예: results_1.mp4
            if not p.exists():                                      # 해당 파일이 아직 없으면
                return p                                            # 이 이름 사용
            idx += 1                                                # 존재하면 번호 증가

    # ==================== 메인 루프 ====================
    def run(self, video_name):
        """영상 파일을 열어 프레임 단위로 처리하며 역주행·정체를 감지/표시/저장"""
        cfg = self.cfg                                              # 설정 단축 참조
        st = self.state                                             # 상태 단축 참조

        video_path = cfg.data_dir / video_name                      # 입력 비디오 경로
        if not video_path.exists():                                 # 파일 존재 확인
            print(f"파일 없음: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))                     # 비디오 캡처 객체 생성
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                 # 프레임 너비
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                # 프레임 높이
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0                     # 원본 FPS (없으면 30)

        st.frame_w, st.frame_h, st.video_fps = fw, fh, fps         # state에 저장
        self.flow.init_grid(fw, fh)                                 # 그리드 초기화

        # ── 방향별 GRUModule 초기화 (Phase 2 — PyTorch 없으면 None 유지) ──
        if _GRU_AVAILABLE:                                          # PyTorch·gru_module 사용 가능이면
            self.gru_module_a = GRUModule(cfg)                      # A방향 GRU 예측 모듈
            self.gru_module_b = GRUModule(cfg)                      # B방향 GRU 예측 모듈
            print("🧠 GRUModule ×2 초기화 완료 (방향별 Phase 2 모드)")
        else:                                                       # 없으면 Phase 1 모드로 동작
            print("ℹ️  GRUModule 없음 → Phase 1 모드로 동작")

        # ── 방향별 TrafficAnalyzer 초기화 ─────────────────────────────
        self.traffic_analyzer_a = TrafficAnalyzer(                  # A방향 정체 탐지
            cfg, frame_w=fw, frame_h=fh, fps=fps,
            passage_tracker=self.passage_tracker_a,                 # A방향 PT 연결
            gru_module=self.gru_module_a                            # A방향 GRU 연결
        )
        self.traffic_analyzer_a.set_state(self.state)               # state 주입

        self.traffic_analyzer_b = TrafficAnalyzer(                  # B방향 정체 탐지
            cfg, frame_w=fw, frame_h=fh, fps=fps,
            passage_tracker=self.passage_tracker_b,                 # B방향 PT 연결
            gru_module=self.gru_module_b                            # B방향 GRU 연결
        )
        self.traffic_analyzer_b.set_state(self.state)               # state 주입

        # ── 방향별 CongestionPredictor 초기화 ─────────────────────────
        self.predictor_a = CongestionPredictor(cfg, fps=fps)        # A방향 정체 예측
        self.predictor_b = CongestionPredictor(cfg, fps=fps)        # B방향 정체 예측

        # ── 정체 탐지 baseline 설정 — 항상 fallback 모드 (stop_ratio + density 기반) ──
        # norm_speed_ref 등 카메라 종속 값 불필요. LCS=default_lcs로 임계값만 보정.
        # flow_map은 역주행 탐지 전용으로만 사용하며 정체 판정 기준으로 쓰지 않는다.
        _fb = BaselineStats(                                        # fallback baseline 생성
            free_flow_dwell=45.0,                                  # 미사용 (fallback 모드에서 참조 안 됨)
            typical_dwell=90.0,                                    # 미사용
            norm_speed_ref=0.15,                                   # 미사용
            count_ref=15.0,                                        # 미사용
            bbox_slope=0.0,                                        # 미사용
            bbox_intercept=50.0,                                   # 미사용
            lcs=cfg.default_lcs,                                   # 한강 측정값 0.36 — 임계값 보정에만 사용
            quality_warning=False,
            passage_count=0,
            is_fallback=True,                                      # fallback → stop_ratio + density 기반 jam 계산
        )
        self.traffic_analyzer_a.set_baseline(_fb)                  # A방향 baseline 설정
        self.traffic_analyzer_b.set_baseline(copy(_fb))            # B방향 baseline 설정
        if not st.is_learning:                                     # 탐지 전용이면 flow_map 로드됨 → 기준 방향 계산
            self._compute_ref_direction()
        print(f"✅ 정체 탐지 baseline 설정 완료 (fallback 모드, LCS={cfg.default_lcs})")

        save_path = self._get_next_filename()                       # 결과 저장 파일명
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")                    # mp4 인코더 설정
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (fw, fh))  # 영상 라이터 생성
        print(f"📹 저장: {save_path}")

        prev_time = time.time()                                     # FPS 계산 시작 시간

        # ── 프레임 간 상태 추적용 변수 ──────────────────────────────
        prev_active_ids: set = set()                                # 이전 프레임 활성 ID 집합 (퇴장 감지용)
        last_footpoints: dict = {}                                  # {track_id: (fx, fy)} 마지막 footpoint
        self._gru_pretrain_pending_a = False                         # A방향 GRU pretrain 예약 플래그
        self._gru_pretrain_pending_b = False                         # B방향 GRU pretrain 예약 플래그
        self._gru_feature_history_a = []                             # A방향 GRU pretrain용 feature 누적
        self._gru_feature_history_b = []                             # B방향 GRU pretrain용 feature 누적

        # ── 학습 연장 상한 계산 ──────────────────────────────────────
        # learning_frames 도달 후에도 min_passages_required 미달이면 최대 학습 연장
        max_learning_frames = int(                                  # 최대 학습 프레임 수
            cfg.learning_frames * cfg.max_learning_extension        # 기본 × 1.5 = 750프레임
        )

        # ── 개선 2: 중간 평활화 플래그 (80%/95% 시점 정확히 1회씩) ─────
        _learn_smoothed_80 = False                                  # 초기 학습 80% 시점 smoothing 완료 플래그
        _learn_smoothed_95 = False                                  # 초기 학습 95% 시점 smoothing 완료 플래그
        _relearn_smoothed_80 = False                                # 재학습 80% 시점 smoothing 완료 플래그
        _relearn_smoothed_95 = False                                # 재학습 95% 시점 smoothing 완료 플래그

        # 키보드 단축키 안내
        print("\n" + "=" * 50)
        print("  [T] 궤적  [D] 방향  [F] 흐름장")
        print("  [S] 속도  [I] 패널  [V] 투표 디버그")
        print("  [P] 내적값  [C] 정체 패널  [Q] 종료")
        print("=" * 50 + "\n")

        while cap.isOpened():                                       # 비디오 스트림이 열려 있는 동안
            ret, frame = cap.read()                                 # 프레임 읽기
            if not ret:                                             # 더 이상 프레임 없으면
                break                                               # 종료

            st.frame_num += 1                                       # 프레임 번호 증가

            # ── YOLO 추적 ──
            tracks = self.tracker.track(frame)                      # [{id, x1, y1, x2, y2, cx, cy}, ...]
            active_ids = {t["id"] for t in tracks}                  # 현재 프레임에 보이는 ID들

            # 처음 등장한 프레임 기록 + on_entry 호출
            for t in tracks:                                        # 각 트랙 순회
                if t["id"] not in st.first_seen_frame:              # 처음 보는 ID이면
                    st.first_seen_frame[t["id"]] = st.frame_num     # 등장 프레임 기록
                    # footpoint: x 중심 = (x1+x2)/2, y = y2 (바운딩박스 하단)
                    entry_fx = (t["x1"] + t["x2"]) / 2             # 진입 footpoint x
                    entry_fy = t["y2"]                              # 진입 footpoint y (하단)
                    self.passage_tracker.on_entry(                  # 진입 기록
                        t["id"], entry_fx, entry_fy, st.frame_num
                    )

            # 현재 프레임 시간(초) = frame / fps
            time_sec = st.frame_num / st.video_fps                  # 영상 타임라인 시간

            # 역주행 확정 차량 수 (W라벨 기준으로 중복 제거)
            wrong_confirmed_count = len(set(st.display_id_map.values()))

            # 이번 프레임에 로그를 남길지 여부 (N프레임마다)
            log_this_frame = (self.logger is not None) and (st.frame_num % cfg.log_interval_frames == 0)

            # 영상 타임라인 기준 시간(초)
            time_sec = st.frame_num / st.video_fps                  # 중복 할당이지만 원본 유지

            # 프레임 단위 요약 로그 저장
            if log_this_frame:                                      # 로그 기록 프레임이면
                wrong_confirmed_count = len(set(st.display_id_map.values()))  # 최신 값 재계산
                _jam_a = (self.traffic_analyzer_a.get_jam_score()       # A방향 jam_score
                          if self.traffic_analyzer_a else 0.0)
                _jam_b = (self.traffic_analyzer_b.get_jam_score()       # B방향 jam_score
                          if self.traffic_analyzer_b else 0.0)
                _lvl_a = (self.traffic_analyzer_a.get_congestion_level()  # A방향 레벨
                          if self.traffic_analyzer_a else "SMOOTH")
                _lvl_b = (self.traffic_analyzer_b.get_congestion_level()  # B방향 레벨
                          if self.traffic_analyzer_b else "SMOOTH")
                _worst_jam = max(_jam_a, _jam_b)                    # 둘 중 높은 jam_score
                _lo = {"SMOOTH": 0, "SLOW": 1, "CONGESTED": 2}     # 레벨 순위
                _worst_lvl = (_lvl_a if _lo.get(_lvl_a, 0) >= _lo.get(_lvl_b, 0)  # 더 나쁜 레벨
                              else _lvl_b)
                self.logger.log_frame(                              # 프레임 로그 기록
                    frame_num=st.frame_num,
                    time_sec=time_sec,
                    active_tracks=len(tracks),
                    wrong_confirmed_count=wrong_confirmed_count,
                    flow_samples_total=self.flow.count.sum(),
                    camera_switch_triggered=False,
                    mode="DETECTING",
                    jam_score=_worst_jam,                           # worst-of-both jam_score
                    congestion_level=_worst_lvl                     # worst-of-both 레벨
                )

            # ── 카메라 전환 감지 ──
            if not st.is_learning and not st.relearning:            # 학습/재학습 중이 아닐 때만
                if self.switch.check(frame, st.frame_num, st.cooldown_until):  # 전환 감지되면
                    st.reset_for_relearn()                          # 상태 초기화
                    self.flow.reset()                               # flow_map 초기화
                    self.passage_tracker.reset()                    # 메인 PT 초기화 (재학습 시작)
                    self.passage_tracker_a.reset()                  # A방향 PT 초기화
                    self.passage_tracker_b.reset()                  # B방향 PT 초기화
                    self.traffic_analyzer_a.congestion_judge.reset()  # A방향 EMA·히스테리시스 초기화
                    self.traffic_analyzer_b.congestion_judge.reset()  # B방향 EMA·히스테리시스 초기화
                    if self.gru_module_a is not None:               # A방향 GRU 있으면
                        self.gru_module_a.reset()                   # 버퍼·hidden·warmup 초기화
                    if self.gru_module_b is not None:               # B방향 GRU 있으면
                        self.gru_module_b.reset()                   # 버퍼·hidden·warmup 초기화
                    self._gru_feature_history_a = []                # A방향 feature 이력 초기화
                    self._gru_feature_history_b = []                # B방향 feature 이력 초기화
                    self._gru_pretrain_pending_a = False             # A방향 pretrain 예약 초기화
                    self._gru_pretrain_pending_b = False             # B방향 pretrain 예약 초기화
                    self._ref_direction = None                      # 기준 방향 초기화 (재학습 후 재계산)
                    self._track_direction.clear()                   # 차량 방향 매핑 초기화
                    _relearn_smoothed_80 = False                    # 재학습 80% smoothing 플래그 리셋
                    _relearn_smoothed_95 = False                    # 재학습 95% smoothing 플래그 리셋

            # ── 초기 학습 완료 처리 ──
            if st.is_learning:                                      # 학습 모드일 때만 체크
                enough_passages = (                                 # 최소 passage 조건 충족 여부
                    self.passage_tracker.get_completed_count()
                    >= cfg.min_passages_required
                )
                # 조건: (기본 학습 프레임 도달 + passage 충분) 또는 최대 학습 프레임 도달
                learning_done = (
                    (st.frame_num >= cfg.learning_frames and enough_passages)
                    or st.frame_num >= max_learning_frames          # 강제 종료 조건
                )
                if learning_done:                                   # 학습 완료이면
                    self.flow.apply_spatial_smoothing(verbose=True) # 공간 보정 (상세 진단)
                    self.flow.apply_boundary_erosion()              # 경계 셀 제거 (오탐 차단)
                    baseline = self.passage_tracker.finalize_baseline()  # passage 통계 산출 (LCS 확인용)
                    # 정체 탐지 baseline은 default_lcs 기반 fallback 유지 — 교체하지 않음
                    self._compute_ref_direction()                   # 기준 방향 벡터 계산
                    if cfg.flow_map_path:                           # 저장 경로 있으면
                        self.flow.save(cfg.flow_map_path)           # flow_map만 저장 (baseline 미포함)
                    st.is_learning = False                          # 학습 모드 종료
                    print(f"학습 완료! (passage={self.passage_tracker.get_completed_count()}, "
                          f"lcs={baseline.lcs:.2f})")
                    # GRU pretrain: 버퍼에 쌓인 feature로 자기지도 사전학습
                    # (학습 완료 후 feature가 아직 없으므로 push()가 충분히 쌓이면 호출)
                    self._gru_pretrain_pending_a = True             # A방향 pretrain 예약
                    self._gru_pretrain_pending_b = True             # B방향 pretrain 예약

            # ── 재학습 모드 처리 ──
            if st.relearning:                                       # 재학습 중이면
                elapsed = st.frame_num - st.relearn_start_frame     # 재학습 경과 프레임
                relearn_max = int(                                  # 재학습 최대 프레임
                    cfg.relearn_frames * cfg.max_learning_extension
                )
                enough_passages = (                                 # passage 조건
                    self.passage_tracker.get_completed_count()
                    >= cfg.min_passages_required
                )
                relearn_done = (
                    (elapsed >= cfg.relearn_frames and enough_passages)
                    or elapsed >= relearn_max                       # 강제 종료
                )
                if relearn_done:                                    # 재학습 완료이면
                    self.flow.apply_spatial_smoothing(verbose=True) # 공간 보정 (상세 진단)
                    self.flow.apply_boundary_erosion()              # 경계 셀 제거 (오탐 차단)
                    baseline = self.passage_tracker.finalize_baseline()  # passage 통계 산출 (LCS 확인용)
                    # 정체 탐지 baseline은 default_lcs 기반 fallback 유지 — 교체하지 않음
                    self._compute_ref_direction()                   # 기준 방향 벡터 재계산
                    if cfg.flow_map_path:                           # 저장 경로 있으면
                        self.flow.save(cfg.flow_map_path)           # flow_map만 저장 (baseline 미포함)
                    st.relearning = False                           # 재학습 모드 종료
                    st.cooldown_until = st.frame_num + cfg.cooldown_frames  # 쿨다운 설정
                    self.switch.set_reference(frame)                # 새 기준 프레임 설정
                    print("재학습 완료! 쿨다운 시작")
                    self._gru_pretrain_pending_a = True             # A방향 GRU pretrain 예약
                    self._gru_pretrain_pending_b = True             # B방향 GRU pretrain 예약

            # ── 차량별 속도 딕셔너리 초기화 ──
            speeds = {}                                             # {tid: mag} — traffic_analyzer용

            # ── 차량별 처리 ──
            for t in tracks:                                        # 각 트랙 순회
                tid = t["id"]                                       # 트랙 ID
                # 바운딩박스 EMA 안정화 적용
                raw_bbox = (t["x1"], t["y1"], t["x2"], t["y2"])     # YOLO 원본 bbox
                x1, y1, x2, y2, cx, cy = self.bbox_stab.stabilize( # 안정화된 bbox·중심점
                    tid, raw_bbox, st.frame_num
                )

                # ── footpoint 계산 (원근 보정 기준점: 바운딩박스 하단 중심) ──
                fx = (x1 + x2) / 2                                  # footpoint x = bbox 가로 중심
                fy = y2                                             # footpoint y = bbox 하단

                # 마지막 footpoint 갱신 (퇴장 시 on_exit에 사용)
                last_footpoints[tid] = (fx, fy)                     # 최신 footpoint 저장

                # ── 방향 분류 (탐지 모드에서만, 기준 방향 설정 후) ─────
                if not st.is_learning and not st.relearning and self._ref_direction is not None:
                    self._track_direction[tid] = self._classify_direction(fx, fy)

                # ID 재매칭 시도 (학습 모드가 아닐 때만)
                if not st.is_learning and not st.relearning:        # 탐지 모드일 때만
                    self.idm.check_reappear(tid, cx, cy)            # 재매칭 시도

                # 궤적에 현재 위치 추가 (footpoint 기준 — bbox 크기 변화에 무관하게 안정적)
                st.trajectories[tid].append((fx, fy))               # cx,cy 대신 fx,fy 사용
                if len(st.trajectories[tid]) > cfg.trail_length:    # 최대 길이 초과 시
                    st.trajectories[tid].pop(0)                     # 오래된 궤적 제거

                traj = st.trajectories[tid]                         # 궤적 참조
                is_wrong = False                                    # 이번 프레임 역주행 여부
                speed = 0                                           # 속도 (초기값 0)
                ndx, ndy = 0, 0                                     # 단위 방향 벡터
                debug_info = {}                                     # 디버그 정보

                # speeds 기본값 없음 — 궤적 부족(신규) 차량은 speeds에 등록 안 함
                # feature_extractor에서 tid not in speeds → 신규 차량으로 판단해 제외

                # 궤적 길이가 velocity_window 이상일 때만 속도/방향 계산
                if len(traj) >= cfg.velocity_window:                # 충분한 궤적 있으면
                    vdx = traj[-1][0] - traj[-cfg.velocity_window][0]  # x 이동량
                    vdy = traj[-1][1] - traj[-cfg.velocity_window][1]  # y 이동량
                    mag = np.sqrt(vdx ** 2 + vdy ** 2)              # 속도 크기 (픽셀)

                    # 프레임당 평균 이동거리 계산 (떨림 필터)
                    avg_move = mag / cfg.velocity_window             # 프레임당 평균 이동

                    speeds[tid] = 0                                 # 궤적 확인 완료 — 기본 정지
                    # 누적 이동거리 AND 프레임당 이동거리 모두 충족해야 "움직이는 중"
                    if mag > cfg.min_move_distance and avg_move > cfg.min_move_per_frame:
                        ndx, ndy = vdx / mag, vdy / mag             # 단위 방향 벡터
                        speed = mag                                 # 속도 = 픽셀 이동량
                        speeds[tid] = speed                         # 속도 딕셔너리에 기록

                        if st.is_learning or st.relearning:         # 학습/재학습 모드
                            self.flow.learn_step(                   # 흐름장 업데이트 (footpoint 기준)
                                traj[-cfg.velocity_window][0],
                                traj[-cfg.velocity_window][1],
                                fx, fy, cfg.min_move_distance       # cx,cy → fx,fy (footpoint 일관성)
                            )
                        else:                                       # 감지 모드
                            # 역주행 여부 판단
                            is_wrong, _, debug_info = self.judge.check(
                                tid, traj, ndx, ndy, mag, cy
                            )

                            # 역주행 의심이 전혀 없으면 정상 흐름으로 온라인 학습
                            if (cfg.enable_online_flow_update and
                                    (not is_wrong) and
                                    (st.wrong_way_count[tid] == 0)):
                                self.flow.learn_step(               # 흐름장 업데이트 (footpoint 기준)
                                    traj[-cfg.velocity_window][0],
                                    traj[-cfg.velocity_window][1],
                                    fx, fy, cfg.min_move_distance  # cx,cy → fx,fy (footpoint 일관성)
                                )
                                # 방향별 SMOOTH + 최소 활성 차량 조건 충족 시 baseline 온라인 갱신
                                _dir = self._track_direction.get(tid, 'a')  # 해당 차량 방향
                                _ta = self.traffic_analyzer_a if _dir == 'a' else self.traffic_analyzer_b
                                _pt = self.passage_tracker_a if _dir == 'a' else self.passage_tracker_b
                                if (_ta is not None and
                                        len(tracks) >= cfg.min_active_for_baseline and
                                        _ta.get_congestion_level() == "SMOOTH"):
                                    bbox_h = max(y2 - y1, 1)        # 바운딩박스 높이 (0 방지)
                                    norm_mag = speed / bbox_h       # normalized_mag 계산
                                    _pt.update_baseline(             # 방향별 기준선 점진 갱신
                                        fx, fy, norm_mag
                                    )

                            # 역주행 확정 시 라벨 부여
                            if is_wrong and tid in st.wrong_way_ids:
                                self.idm.assign_label(tid)          # W1, W2... 라벨 배정
                else:
                    # 궤적이 짧더라도 이미 역주행 확정된 차량이면 그대로 표시
                    if tid in st.wrong_way_ids:                     # 이미 확정된 역주행 차량
                        is_wrong = True                             # 역주행 플래그 유지
                        debug_info = {"status": "CONFIRMED", "cos_values": []}

                # ── 트랙 단위 로그 생성 ────────────────────────────────────
                speed_thr = self.judge.get_speed_threshold(cy)      # 원근 기반 속도 임계값

                flow_v = None                                       # 정상 흐름 벡터
                cos_current = None                                  # 현재 코사인 유사도

                if speed > 0:                                       # 움직이는 중이면
                    flow_v = self.flow.get_interpolated(cx, cy)     # 보간된 흐름 벡터
                    if flow_v is not None:                          # 흐름 벡터 유효하면
                        cos_current = float(ndx * flow_v[0] + ndy * flow_v[1])  # 코사인 계산

                # debug_info에서 투표 정보 추출
                agree = debug_info.get("agree", "")                 # 정방향 투표 수
                disagree = debug_info.get("disagree", "")           # 역방향 투표 수
                skip = debug_info.get("skip", "")                   # 스킵 수
                total = debug_info.get("total", "")                 # 총 투표 수
                judge_status = debug_info.get("status", "")         # 판정 상태

                # 궤적이 짧아서 judge를 안 탄 경우 상태 표시
                if judge_status == "" and len(traj) < cfg.velocity_window:
                    judge_status = "short"                          # 궤적 부족 표시

                # disagree_ratio 계산 (가능할 때만)
                disagree_ratio = ""                                 # 기본값: 빈 문자열
                if isinstance(total, int) and total > 0 and isinstance(disagree, int):
                    disagree_ratio = round(disagree / total, 4)     # 역방향 비율

                # 사용자 표시 라벨 (W1, W2... 없으면 빈칸)
                display_label = self.idm.get_display_label(tid) or ""

                # 역주행 확정 여부 (0/1)
                is_confirmed = int(tid in st.wrong_way_ids)         # 확정이면 1

                # 최종 1행 저장
                if log_this_frame:                                  # 로그 기록 프레임이면
                    self.logger.log_track([                         # 트랙 로그 기록
                        st.frame_num, round(time_sec, 3), tid, display_label,
                        round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                        round(cx, 2), round(cy, 2),
                        round(speed, 3) if speed else "",
                        round(ndx, 5) if speed else "",
                        round(ndy, 5) if speed else "",
                        round(speed_thr, 3) if speed_thr else "",
                        round(float(flow_v[0]), 5) if flow_v is not None else "",
                        round(float(flow_v[1]), 5) if flow_v is not None else "",
                        round(cos_current, 5) if cos_current is not None else "",
                        judge_status, agree, disagree, skip, total,
                        disagree_ratio,
                        st.wrong_way_count.get(tid, 0),
                        is_confirmed
                    ])

                # 시각화용 플래그: 이미 확정된 역주행도 항상 붉게 표시
                is_wrong_display = is_wrong or (tid in st.wrong_way_ids)

                if is_wrong_display:                                # 역주행 차량이면
                    st.wrong_way_last_pos[tid] = (cx, cy, st.frame_num)  # 위치 기록

                # ── 시각화 ──
                if self.vis.show_trails:                            # 궤적 표시 활성화 시
                    self.vis.draw_trajectory(frame, tid, is_wrong_display)

                if self.vis.show_direction and speed > 3:           # 방향 화살표 활성화 시
                    self.vis.draw_direction_arrow(frame, cx, cy, ndx, ndy,
                                                 speed, is_wrong_display)

                if is_wrong_display:                                # 역주행 차량이면
                    self.vis.draw_wrong_way_alert(frame, tid, x1, y1, x2, y2)  # 경고 표시
                else:                                               # 정상 차량이면
                    self.vis.draw_normal_box(frame, tid, x1, y1, x2, y2)       # 초록 박스

                if self.vis.show_speed and speed > 3:               # 속도 표시 활성화 시
                    self.vis.draw_speed_label(frame, x1, y2, speed, cy, is_wrong_display)

                if self.vis.show_vote_debug and debug_info:         # 투표 디버그 활성화 시
                    self.vis.draw_vote_debug(frame, tid, debug_info, x2, y1)

                # 내적값(코사인) 히스토리 저장/표시
                cos_vals = debug_info.get("cos_values", [])         # 코사인 값 리스트
                if cos_vals:                                        # 값이 있으면
                    st.last_cos_values[tid] = cos_vals              # 히스토리 저장

                if self.vis.show_dot_product and st.last_cos_values.get(tid):  # 내적 표시 활성화 시
                    self.vis.draw_dot_product(frame, x2, y1,
                                             st.last_cos_values[tid],
                                             is_wrong_display)

            # ── 이번 프레임에서 사라진 차량 처리 (on_exit) ──────────────
            gone_ids = prev_active_ids - active_ids                 # 이전 프레임에 있었지만 지금 없는 ID
            exit_count = len(gone_ids)                              # 퇴장 차량 수 (전체)
            exit_count_a = 0                                        # A방향 퇴장 수
            exit_count_b = 0                                        # B방향 퇴장 수

            for gone_id in gone_ids:                                # 퇴장 ID 순회
                # 방향별 퇴장 수 집계 (방향 정보 가져오고 제거)
                _gone_dir = self._track_direction.pop(gone_id, 'a') # 방향 꺼내며 매핑 정리
                if _gone_dir == 'a':                                # A방향이면
                    exit_count_a += 1                               # A 퇴장 수 증가
                else:                                               # B방향이면
                    exit_count_b += 1                               # B 퇴장 수 증가

                fp = last_footpoints.pop(gone_id, None)             # 마지막 footpoint 꺼냄
                if fp is not None:                                  # footpoint 기록 있으면
                    self.passage_tracker.on_exit(                   # 정상 퇴장 기록 (메인 PT)
                        gone_id, fp[0], fp[1], st.frame_num, is_complete=True
                    )
                else:                                               # 기록 없으면 entry_positions 활용
                    ep = st.entry_positions.get(gone_id, (0.0, 0.0))  # 진입 위치 fallback
                    self.passage_tracker.on_exit(                   # 비정상 퇴장 기록 (메인 PT)
                        gone_id, ep[0], ep[1], st.frame_num, is_complete=False
                    )

            # ── PassageTracker 프레임별 통계 수집 (방향별 분리) ──────────
            norm_mags_all = []                                      # 전체 normalized_mag (학습용)
            cy_vals_all = []                                        # 전체 cy (학습용)
            bbox_h_vals_all = []                                    # 전체 bbox_h (학습용)
            norm_mags_a, cy_vals_a, bbox_h_vals_a = [], [], []      # A방향 통계
            norm_mags_b, cy_vals_b, bbox_h_vals_b = [], [], []      # B방향 통계
            tracks_a, speeds_a = [], {}                             # A방향 차량 목록·속도
            tracks_b, speeds_b = [], {}                             # B방향 차량 목록·속도

            for t in tracks:                                        # 활성 차량별 수집
                tid_val = t["id"]                                   # 트랙 ID
                mag_val = speeds.get(tid_val)                       # 속도 조회 (없으면 None = 신규)
                bbox_h = max(t["y2"] - t["y1"], 1)                  # 바운딩박스 높이 (0 방지)
                # normalized_mag: 신규(mag_val=None)이면 0으로만 통계 수집 (학습용)
                nm = (mag_val / bbox_h) if mag_val else 0.0         # normalized_mag
                cy_val = t["cy"]                                    # cy
                bh = float(bbox_h)                                  # bbox_h (float)

                # 전체 리스트 누적 (학습 모드용)
                norm_mags_all.append(nm)                            # 전체 norm_mag
                cy_vals_all.append(cy_val)                          # 전체 cy
                bbox_h_vals_all.append(bh)                          # 전체 bbox_h

                # 방향별 분리 (탐지 모드용)
                _dir = self._track_direction.get(tid_val, 'a')      # 차량 방향 조회
                if _dir == 'a':                                     # A방향이면
                    norm_mags_a.append(nm)                          # A방향 통계 추가
                    cy_vals_a.append(cy_val)
                    bbox_h_vals_a.append(bh)
                    tracks_a.append(t)                              # A방향 차량 목록
                    if mag_val is not None:                         # 신규 제외 — feature_extractor가 None=신규로 구분
                        speeds_a[tid_val] = mag_val                 # A방향 속도 딕셔너리
                else:                                               # B방향이면
                    norm_mags_b.append(nm)                          # B방향 통계 추가
                    cy_vals_b.append(cy_val)
                    bbox_h_vals_b.append(bh)
                    tracks_b.append(t)                              # B방향 차량 목록
                    if mag_val is not None:                         # 신규 제외
                        speeds_b[tid_val] = mag_val                 # B방향 속도 딕셔너리

            if st.is_learning or st.relearning:                     # 학습/재학습 모드
                self.passage_tracker.record_frame_stats(            # 메인 PT에 전체 통계
                    active_count=len(tracks), exit_count=exit_count,
                    norm_mags=norm_mags_all, cy_vals=cy_vals_all,
                    bbox_h_vals=bbox_h_vals_all
                )
            else:                                                   # 탐지 모드
                self.passage_tracker_a.record_frame_stats(          # A방향 PT에 A 통계
                    active_count=len(tracks_a), exit_count=exit_count_a,
                    norm_mags=norm_mags_a, cy_vals=cy_vals_a,
                    bbox_h_vals=bbox_h_vals_a
                )
                self.passage_tracker_b.record_frame_stats(          # B방향 PT에 B 통계
                    active_count=len(tracks_b), exit_count=exit_count_b,
                    norm_mags=norm_mags_b, cy_vals=cy_vals_b,
                    bbox_h_vals=bbox_h_vals_b
                )

            # ── 이전 프레임 활성 ID 갱신 ─────────────────────────────────
            prev_active_ids = active_ids.copy()                     # 다음 프레임 비교용으로 저장

            # ── 방향별 TrafficAnalyzer·GRU 갱신 ──────────────────────────
            if (self.traffic_analyzer_a is not None                 # 초기화 완료 확인
                    and not st.is_learning and not st.relearning):  # 탐지 모드일 때만
                # A방향 정체 탐지 갱신
                self.traffic_analyzer_a.update(tracks_a, speeds_a, st.frame_num)
                self.predictor_a.update(self.traffic_analyzer_a.get_avg_speed())
                # B방향 정체 탐지 갱신
                self.traffic_analyzer_b.update(tracks_b, speeds_b, st.frame_num)
                self.predictor_b.update(self.traffic_analyzer_b.get_avg_speed())

                # ── GRU pretrain용 feature 누적 (A방향) ──────────────────
                if self._gru_pretrain_pending_a and self.gru_module_a is not None:
                    feat_a = self.traffic_analyzer_a.get_last_feature()  # A방향 최신 feature
                    if feat_a is not None:
                        self._gru_feature_history_a.append(feat_a)

                # ── GRU pretrain용 feature 누적 (B방향) ──────────────────
                if self._gru_pretrain_pending_b and self.gru_module_b is not None:
                    feat_b = self.traffic_analyzer_b.get_last_feature()  # B방향 최신 feature
                    if feat_b is not None:
                        self._gru_feature_history_b.append(feat_b)

                # ── GRU pretrain 실행 (A방향) ────────────────────────────
                if (self._gru_pretrain_pending_a
                        and self.gru_module_a is not None
                        and len(self._gru_feature_history_a) >= cfg.gru_seq_len * 2):
                    losses_a = self.gru_module_a.pretrain(self._gru_feature_history_a)
                    if losses_a:
                        print(f"🧠 GRU-A pretrain 완료: loss {losses_a[0]:.4f}→{losses_a[-1]:.4f}")
                    self._gru_pretrain_pending_a = False
                    self._gru_feature_history_a = []

                # ── GRU pretrain 실행 (B방향) ────────────────────────────
                if (self._gru_pretrain_pending_b
                        and self.gru_module_b is not None
                        and len(self._gru_feature_history_b) >= cfg.gru_seq_len * 2):
                    losses_b = self.gru_module_b.pretrain(self._gru_feature_history_b)
                    if losses_b:
                        print(f"🧠 GRU-B pretrain 완료: loss {losses_b[0]:.4f}→{losses_b[-1]:.4f}")
                    self._gru_pretrain_pending_b = False
                    self._gru_feature_history_b = []

                # ── GRU online_step: 각 방향 SMOOTH일 때 학습 ────────────
                if self.gru_module_a is not None:                   # A방향 GRU 있으면
                    if self.traffic_analyzer_a.get_congestion_level() == "SMOOTH":
                        self.gru_module_a.online_step(label=0)      # SMOOTH=0 레이블
                if self.gru_module_b is not None:                   # B방향 GRU 있으면
                    if self.traffic_analyzer_b.get_congestion_level() == "SMOOTH":
                        self.gru_module_b.online_step(label=0)      # SMOOTH=0 레이블

            # ── 트랙 정리 ──
            if st.frame_num % 30 == 0:                              # 30프레임마다
                self.idm.cleanup(active_ids)                        # ID 관리자 정리
                self.bbox_stab.cleanup(active_ids)                  # bbox 안정화 캐시 정리

            # 흐름장(배경 화살표) 표시
            if self.vis.show_flow:                                  # 흐름장 활성화 시
                self.vis.draw_flow(frame)

            # 정보 패널 표시
            if self.vis.show_info_panel:                            # 패널 활성화 시
                self.vis.draw_info_panel(frame, len(tracks))

            # 탐지 소요시간 통계 패널
            if self.vis.show_detection_stats:                       # 통계 패널 활성화 시
                self.vis.draw_detection_stats(frame)

            # 학습 중이면 화면 상단에 학습 진행률 표시
            if st.is_learning:                                      # 초기 학습 모드이면
                progress = min(100, st.frame_num / max(cfg.learning_frames, 1) * 100)  # 진행률 (%)
                passages = self.passage_tracker.get_completed_count()  # 완성 passage 수
                cv2.putText(frame, f"LEARNING FLOW MAP: {progress:.0f}%  (passage={passages})",
                            (fw // 2 - 240, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2, cv2.LINE_AA)

            # 재학습 중이면 화면 상단에 상태 텍스트 표시
            elif st.relearning:                                     # 재학습 모드이면 (elif — 동시 표시 방지)
                elapsed = st.frame_num - st.relearn_start_frame     # 경과 프레임
                progress = min(100, elapsed / cfg.relearn_frames * 100)  # 진행률 (%)
                cv2.putText(frame, f"CAMERA SWITCHED - RE-LEARNING: {progress:.0f}%",
                            (fw // 2 - 220, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            # ── 방향별 정체 상태 패널 표시 (좌하단) ────────────────────
            if st.is_learning or st.relearning:                     # 학습/재학습 중이면
                self.vis.draw_learning_status(frame, st.frame_num)  # "데이터 수집 중" 패널 표시
            elif self.traffic_analyzer_a is not None:               # 탐지 모드 + 초기화 완료 확인
                level_a     = self.traffic_analyzer_a.get_congestion_level()  # A방향 정체 레벨
                jam_score_a = self.traffic_analyzer_a.get_jam_score()         # A방향 jam_score
                dur_sec_a   = self.traffic_analyzer_a.get_duration_sec()      # A방향 지속 시간
                level_b     = self.traffic_analyzer_b.get_congestion_level()  # B방향 정체 레벨
                jam_score_b = self.traffic_analyzer_b.get_jam_score()         # B방향 jam_score
                dur_sec_b   = self.traffic_analyzer_b.get_duration_sec()      # B방향 지속 시간
                if self.vis.show_congestion_panel:                  # C키로 패널 ON/OFF 가능
                    self.vis.draw_congestion_status(                # 방향별 정체 패널 그리기
                        frame,
                        level_a, jam_score_a, dur_sec_a,            # A방향 데이터
                        level_b, jam_score_b, dur_sec_b,            # B방향 데이터
                        label_a=self._dir_label_a,                  # "UP" 또는 "DOWN"
                        label_b=self._dir_label_b                   # "DOWN" 또는 "UP"
                    )

            # FPS 계산 및 표시
            curr_time = time.time()                                 # 현재 시간
            show_fps = 1 / (curr_time - prev_time + 1e-6)          # FPS 계산
            prev_time = curr_time                                   # 시간 갱신
            fps_y = 255 if self.vis.show_info_panel else 30         # 패널 유무에 따라 y 위치
            cv2.putText(frame, f"FPS: {show_fps:.1f}", (10, fps_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)                                     # 결과 영상 파일에 프레임 기록
            cv2.imshow("Highway Wrong-Way Detection", frame)        # 화면에 출력

            key = cv2.waitKey(1) & 0xFF                             # 키 입력 대기
            if key == ord("q"):                                     # q 누르면
                break                                               # 종료
            self.vis.handle_keys(key)                               # 시각화 옵션 토글 처리

            # ── 개선 2: 80%/95% 시점 중간 평활화 (플래그 방식 — 정확히 1회씩) ──
            # 너무 이른 보간 채움 방지: 데이터가 충분히 쌓인 후반에만 실행
            # 학습 완료 시 verbose=True smoothing이 최종 1회 추가 실행됨
            if st.is_learning:                                      # 초기 학습 모드
                progress = st.frame_num / max(cfg.learning_frames, 1)  # 학습 진행률 (0.0~1.0+)
                if progress >= 0.80 and not _learn_smoothed_80:     # 80% 도달 & 미실행
                    self.flow.apply_spatial_smoothing()             # 1차 중간 평활화
                    _learn_smoothed_80 = True                       # 플래그 설정 (재실행 방지)
                if progress >= 0.95 and not _learn_smoothed_95:     # 95% 도달 & 미실행
                    self.flow.apply_spatial_smoothing()             # 2차 중간 평활화
                    _learn_smoothed_95 = True                       # 플래그 설정 (재실행 방지)
            elif st.relearning:                                     # 재학습 모드
                elapsed = st.frame_num - st.relearn_start_frame    # 재학습 경과 프레임
                progress = elapsed / max(cfg.relearn_frames, 1)    # 재학습 진행률
                if progress >= 0.80 and not _relearn_smoothed_80:  # 80% 도달 & 미실행
                    self.flow.apply_spatial_smoothing()             # 1차 중간 평활화
                    _relearn_smoothed_80 = True                    # 플래그 설정
                if progress >= 0.95 and not _relearn_smoothed_95:  # 95% 도달 & 미실행
                    self.flow.apply_spatial_smoothing()             # 2차 중간 평활화
                    _relearn_smoothed_95 = True                    # 플래그 설정

        # ── 루프 종료 후 정리 ──
        cap.release()                                               # 비디오 캡처 해제
        writer.release()                                            # 비디오 라이터 해제
        cv2.destroyAllWindows()                                     # 모든 OpenCV 창 닫기

        # 학습이 완료되지 않은 채로 종료된 경우 마지막 flow_map 저장
        if not st.is_learning and cfg.flow_map_path:                # 학습 완료 상태이면
            self.flow.save(cfg.flow_map_path)                       # flow_map 저장 (baseline 없이)

        self._print_final_stats(save_path)                          # 최종 통계 출력

        # ── 이벤트 로그 저장(확정된 W1/W2 통계 덤프) ──
        if self.logger:                                             # 로거 활성화 상태이면
            self.logger.log_events_from_stats(st.detection_stats)   # 이벤트 로그 저장
            self.logger.close()                                     # 파일 닫기

    # ==================== 최종 통계 출력 ====================
    def _print_final_stats(self, save_path):
        """최종 탐지 소요시간 통계 콘솔 출력"""
        st = self.state                                             # 상태 단축 참조

        # detection_stats에 기록된 역주행 차량(라벨)의 수 = 최종 역주행 차량 수
        total_wrong = len(st.detection_stats)                       # 총 역주행 차량 수

        print(f"\n✅ 저장 완료: {save_path} ({st.frame_num} 프레임)")
        print(f"   총 역주행 차량: {total_wrong}대")

        # 각 역주행 라벨(W1, W2...)에 어떤 ID들이 쓰였는지 출력
        for label in sorted(set(st.display_id_map.values())):       # 라벨별 순회
            ids = [k for k, v in st.display_id_map.items() if v == label]  # 해당 라벨의 ID 목록
            print(f"   {label}: 사용된 ID {ids}")

        # detection_stats가 비어있지 않으면 상세 통계 출력
        if st.detection_stats:                                      # 역주행 1대 이상이면
            print(f"\n{'=' * 60}")
            print(f" 역주행 탐지 소요시간 통계 (FPS: {st.video_fps:.1f})")
            print(f"{'=' * 60}")
            print(f"  {'차량':>6} │ {'등장→확정':>18} │ {'의심→확정':>18}")
            print(f"  {'─' * 6}─┼─{'─' * 18}─┼─{'─' * 18}")

            all_appear = []                                         # 등장→확정 시간 리스트
            all_suspect = []                                        # 의심→확정 시간 리스트

            for label, s in sorted(st.detection_stats.items()):     # 라벨별 순회
                appear_str = f"{s['frames_from_appear']:>4}f ({s['seconds_from_appear']:>5.2f}s)"
                suspect_str = f"{s['frames_from_suspect']:>4}f ({s['seconds_from_suspect']:>5.2f}s)"

                print(f"  {label:>6} │ {appear_str:>18} │ {suspect_str:>18}")

                all_appear.append(s["seconds_from_appear"])         # 초 단위 값 저장
                all_suspect.append(s["seconds_from_suspect"])       # 초 단위 값 저장

            print(f"  {'─' * 6}─┼─{'─' * 18}─┼─{'─' * 18}")

            # 평균·최소·최대 계산
            avg_a = np.mean(all_appear)                             # 등장→확정 평균
            avg_s = np.mean(all_suspect)                            # 의심→확정 평균
            min_a = np.min(all_appear)                              # 등장→확정 최소
            max_a = np.max(all_appear)                              # 등장→확정 최대
            min_s = np.min(all_suspect)                             # 의심→확정 최소
            max_s = np.max(all_suspect)                             # 의심→확정 최대

            # 요약 통계 출력
            print(f"  {'평균':>6} │ {'':>13}{avg_a:>5.2f}s │ {'':>13}{avg_s:>5.2f}s")
            print(f"  {'최소':>6} │ {'':>13}{min_a:>5.2f}s │ {'':>13}{min_s:>5.2f}s")
            print(f"  {'최대':>6} │ {'':>13}{max_a:>5.2f}s │ {'':>13}{max_s:>5.2f}s")
            print(f"{'=' * 60}")
        else:                                                       # 역주행 없으면
            print("\n 역주행 차량이 감지되지 않았습니다.")
