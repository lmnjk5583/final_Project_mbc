# 파일 경로: 최종 프로젝트/src/detector.py
# 역할: 모든 모듈을 조립하고 run() 루프를 실행하는 메인 오케스트레이터.

import cv2                                          # OpenCV — 영상 입출력·시각화
import numpy as np                                  # 수치 계산
import time                                         # FPS 측정용 타이머
import threading                                    # URL 선제 갱신용 백그라운드 스레드

from .config import DetectorConfig                  # 모든 파라미터가 담긴 설정 클래스
from .state import DetectorState                    # 프레임 번호·궤적·역주행 카운트 등 런타임 상태
from .flow_map import FlowMap                       # 15×15 그리드 정상 흐름 벡터 학습/비교
from .tracker import YoloTracker                    # YOLO 검출 + ByteTrack 추적
from .judge import WrongWayJudge                    # 코사인 유사도 + 투표/히스테리시스 역주행 판정
from .id_manager import IDManager                   # W라벨 관리 + occlusion 재매칭 + 오래된 트랙 정리
from .camera_switch import CameraSwitchDetector     # 장면/카메라 전환 감지 (grayscale diff)
from .visualizer import Visualizer                  # 시각화(박스/궤적/패널/디버그)
from .logger import CSVLogger                       # 프레임/트랙/이벤트 CSV 로그 저장
from .traffic_analyzer import TrafficAnalyzer, CongestionPredictor  # 정체 탐지 + 단기 예측

# GRUModule: PyTorch 없는 환경에서도 동작하도록 try/except
try:
    from gru_module import GRUModule                # Phase 2 GRU 예측 모듈
    _GRU_AVAILABLE = True                           # GRU 사용 가능 플래그
except ImportError:                                 # gru_module.py 없거나 torch 없으면
    _GRU_AVAILABLE = False                          # fallback 모드

try:
    from flow_map_matcher import FlowMapMatcher, save_ref_frame  # flow_map 자동 매칭
    _MATCHER_AVAILABLE = True
except ImportError:
    _MATCHER_AVAILABLE = False


class Detector:
    """역주행 탐지 + 정체 탐지를 수행하는 메인 오케스트레이터."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg                                              # 설정 객체 저장

        # ── 런타임 상태(state) + 모듈 생성 ──────────────────────────────
        self.state = DetectorState()                                # 프레임 번호, 궤적, 역주행 카운트 등
        self.flow = FlowMap(                                            # 정상 흐름 벡터 그리드
            cfg.grid_size, cfg.alpha, cfg.min_samples,
            bbox_alpha_decay=getattr(cfg, "bbox_alpha_decay", 0.5),
            bbox_gating_alpha_ratio=getattr(cfg, "bbox_gating_alpha_ratio", 0.3),
            edge_margin=getattr(cfg, "flow_map_edge_margin", 1),
        )
        self.tracker = YoloTracker(cfg.model_path, cfg.conf, cfg.target_classes,
                                   night_enhance=getattr(cfg, "night_enhance", True))  # YOLO+ByteTrack
        self.judge = WrongWayJudge(cfg, self.flow, self.state)      # 역주행 판정기
        self.idm = IDManager(cfg, self.flow, self.state)            # ID 관리 + 재매칭
        self.switch = CameraSwitchDetector(cfg)                     # 카메라 전환 감지기
        self.vis = Visualizer(cfg, self.state, self.flow)           # 시각화 모듈
        self.logger = CSVLogger(cfg.log_dir) if cfg.log_dir else None  # CSV 로거 (log_dir 없으면 None)
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
        self._valid_cells_a: int = 1                                # A방향 유효 셀 수 (bbox_coverage 원근 보정용)
        self._valid_cells_b: int = 1                                # B방향 유효 셀 수

        # ── flow_map 로드 (탐지 전용이라면 필수) ────────────────────
        if cfg.flow_map_path:                                       # flow_map 경로가 설정되어 있으면
            if not cfg.detect_only:                                 # 학습 모드이면 기존 파일 무시하고 재학습
                self.state.is_learning = True                       # 학습 모드로 전환
                print("detect_only=False → 기존 flow_map 무시, 처음부터 학습 시작")
            else:                                                   # 탐지 전용이면 기존 파일 로드
                loaded = self.flow.load(cfg.flow_map_path)          # bool 반환
                if not loaded:                                      # 탐지 전용인데 로드 실패
                    raise FileNotFoundError(                        # 즉시 예외 → 잘못된 실험 방지
                        f"detect_only=True 인데 flow_map이 없습니다: {cfg.flow_map_path}"
                    )
                self.flow.speed_ref[:] = 0                          # 이전 세션 오염 방지 — 항상 리셋
                self.state.is_learning = False                      # 로드 성공 → 학습 모드 해제
                # ── 119차: v3 이하 npy 로드 후 양방향 채널 재구성 ──────────
                # v4 이상이면 load()에서 이미 채널 복원됨.
                # v3 이하(채널 없음)도 기존 global flow로부터 채널을 즉시 구성해
                # 중앙선 오염 방지·contamination-aware fallback을 활성화한다.
                self._compute_ref_direction()                       # 기준 방향 계산 (최다 샘플 셀)
                if self._ref_direction is not None:                 # 기준 방향이 있으면
                    self.flow.build_directional_channels(           # 양방향 채널 구성
                        *self._ref_direction
                    )
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

    # ==================== 방향별 유효 셀 수 계산 ====================
    def _compute_direction_cell_counts(self):
        """flow_map 유효 셀을 A/B방향으로 분류해 각 셀 수를 계산한다.

        bbox_coverage 계산 시 전체 road_area 대신 방향별 road_area를 사용하기 위해
        학습 완료 직후 _compute_ref_direction() 다음에 호출한다.

        결과를 _valid_cells_a/b에 저장 후 각 TrafficAnalyzer에 주입.
        """
        if self._ref_direction is None:                           # 기준 방향 미설정이면
            return                                                # 계산 불가 → 기본값 유지
        ref_x, ref_y = self._ref_direction                        # 기준 방향 벡터
        count_a, count_b = 0, 0                                   # 방향별 셀 카운터
        for r in range(self.flow.grid_size):                      # 행 순회
            for c in range(self.flow.grid_size):                  # 열 순회
                if self.flow.count[r, c] <= 0:                    # 미학습 셀 건너뜀
                    continue
                vx = float(self.flow.flow[r, c, 0])               # 셀 흐름 x
                vy = float(self.flow.flow[r, c, 1])               # 셀 흐름 y
                cos_val = vx * ref_x + vy * ref_y                 # 기준 방향과 코사인 유사도
                if cos_val >= self.cfg.lane_cos_threshold:         # A방향 기준 이상
                    count_a += 1                                  # A방향 셀 카운트
                else:                                             # B방향
                    count_b += 1                                  # B방향 셀 카운트
        self._valid_cells_a = max(count_a, 1)                     # 0 방지
        self._valid_cells_b = max(count_b, 1)                     # 0 방지
        print(f"📐 방향별 셀 수: A={self._valid_cells_a}, B={self._valid_cells_b}"
              f" (전체 유효={count_a + count_b})")
        # TrafficAnalyzer에 방향별 셀 수 주입
        if self.traffic_analyzer_a is not None:                   # A방향 analyzer 있으면
            self.traffic_analyzer_a.set_valid_cell_count(self._valid_cells_a)
        if self.traffic_analyzer_b is not None:                   # B방향 analyzer 있으면
            self.traffic_analyzer_b.set_valid_cell_count(self._valid_cells_b)

    # ==================== 차량 방향 분류 ====================
    def _classify_direction(self, fx, fy):
        """footpoint 위치의 flow_map 셀 방향과 기준 방향을 비교해 'a' 또는 'b' 반환.

        미학습 셀(flow_v=None)이면 nearest-neighbor로 가장 가까운 학습 셀 방향을 사용한다.
        flow_map 상단(rows 0~6)이 미학습인 환경에서 상행 차량이 A로 오분류되던 문제 해결.
        """
        if self._ref_direction is None:                           # 기준 방향 미설정
            return 'a'                                            # 기본값: A방향
        flow_v = self.flow.get_interpolated(fx, fy)               # 해당 위치 흐름 벡터
        if flow_v is None:                                        # 미학습 구역 → nearest-neighbor
            flow_v = self.flow.get_nearest_direction(fx, fy)      # 가장 가까운 학습 셀 벡터
        if flow_v is None:                                        # 학습 셀 자체가 없음
            return 'a'                                            # 최종 fallback: A방향
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
    def run(self, video_name, max_seconds: float | None = None,
            url_refresher=None, url_refresh_interval: float | None = None):
        """영상 파일 또는 스트림 URL을 열어 프레임 단위로 처리한다.

        Args:
            video_name: 파일명(str) 또는 스트림 URL(http/rtsp로 시작).
            max_seconds: 이 시간(초) 경과 후 루프 종료. None이면 영상 끝까지.
            url_refresher: () -> str | None 콜백. 스트림 단절 또는 선제 갱신 시 새 URL을 반환.
                           None이면 단절 시 루프 종료 (기존 동작).
                           Detector 상태(trajectories, flow_map 등)는 유지됨.
            url_refresh_interval: 이 주기(초)마다 루프 안에서 cap을 교체 (화면 멈춤 없음).
                           갱신 20초 전부터 백그라운드 스레드로 새 스트림을 미리 열어둠.
                           설정 시 max_seconds 기반 루프 종료는 비활성.
        """
        cfg = self.cfg                                              # 설정 단축 참조
        st = self.state                                             # 상태 단축 참조

        # ── 입력 소스 판단: URL이면 직접 열고, 파일명이면 data_dir과 조합 ──
        is_stream = str(video_name).startswith(("http", "rtsp"))    # URL 여부
        if is_stream:
            video_src = str(video_name)                             # URL 그대로 사용
        else:
            video_path = cfg.data_dir / video_name                  # 파일 경로 조합
            if not video_path.exists():
                print(f"파일 없음: {video_path}")
                return
            video_src = str(video_path)

        cap = cv2.VideoCapture(video_src)                           # 비디오/스트림 캡처 객체 생성
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                 # 프레임 너비
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                # 프레임 높이
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0                     # 원본 FPS (없으면 30)

        st.frame_w, st.frame_h, st.video_fps = fw, fh, fps         # state에 저장
        self.flow.init_grid(fw, fh)                                 # 그리드 초기화

        # ── 방향별 GRUModule 초기화 (Phase 2 — PyTorch 없으면 None 유지) ──
        if _GRU_AVAILABLE:                                          # PyTorch·gru_module 사용 가능이면
            self.gru_module_a = GRUModule(cfg, fps=fps)             # A방향 GRU — fps로 horizon 프레임 계산
            self.gru_module_b = GRUModule(cfg, fps=fps)             # B방향 GRU
            # ── 저장된 weights 로드 (flow_map 같은 폴더) ──────────────
            if cfg.flow_map_path:
                _gru_a_path = cfg.flow_map_path.parent / "gru_a.pt"
                _gru_b_path = cfg.flow_map_path.parent / "gru_b.pt"
                if _gru_a_path.exists() and self.gru_module_a.load(_gru_a_path):
                    print(f"🧠 GRU-A weights 로드 완료: {_gru_a_path}")
                if _gru_b_path.exists() and self.gru_module_b.load(_gru_b_path):
                    print(f"🧠 GRU-B weights 로드 완료: {_gru_b_path}")
            print("🧠 GRUModule ×2 초기화 완료 (방향별 Phase 2 모드)")
        else:                                                       # 없으면 Phase 1 모드로 동작
            print("ℹ️  GRUModule 없음 → Phase 1 모드로 동작")

        # ── 방향별 TrafficAnalyzer 초기화 ─────────────────────────────
        self.traffic_analyzer_a = TrafficAnalyzer(                  # A방향 정체 탐지
            cfg, frame_w=fw, frame_h=fh, fps=fps,
            flow_map=self.flow,                                     # flow_map 주입 (bbox_coverage 계산용)
            gru_module=self.gru_module_a                            # A방향 GRU 연결
        )
        self.traffic_analyzer_a.set_state(self.state)               # state 주입

        self.traffic_analyzer_b = TrafficAnalyzer(                  # B방향 정체 탐지
            cfg, frame_w=fw, frame_h=fh, fps=fps,
            flow_map=self.flow,                                     # flow_map 주입 (bbox_coverage 계산용)
            gru_module=self.gru_module_b                            # B방향 GRU 연결
        )
        self.traffic_analyzer_b.set_state(self.state)               # state 주입

        # ── 방향별 CongestionPredictor 초기화 ─────────────────────────
        self.predictor_a = CongestionPredictor(cfg, fps=fps)        # A방향 정체 예측
        self.predictor_b = CongestionPredictor(cfg, fps=fps)        # B방향 정체 예측

        # ── 정체 탐지 활성화 ──────────────────────────────────────────────
        self.traffic_analyzer_a.set_baseline()                     # A방향 FE+CJ 활성화
        self.traffic_analyzer_b.set_baseline()                     # B방향 FE+CJ 활성화
        if not st.is_learning:                                     # 탐지 전용이면 flow_map 로드됨 → 기준 방향 계산
            self._compute_ref_direction()
            self._compute_direction_cell_counts()                  # 방향별 셀 수 계산 → TA 주입
        print("✅ 정체 탐지 활성화 완료")

        save_path = None                                            # 녹화 경로 (없으면 None)
        if self.cfg.result_dir:                                     # result_dir 없으면 녹화 안 함
            save_path = self._get_next_filename()                   # 결과 저장 파일명
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")                # mp4 인코더 설정
            writer = cv2.VideoWriter(str(save_path), fourcc, fps, (fw, fh))  # 영상 라이터 생성
            print(f"📹 저장: {save_path}")
        else:
            writer = None                                           # 녹화 없음

        prev_time = time.time()                                     # FPS 계산 시작 시간

        # ── 프레임 간 상태 추적용 변수 ──────────────────────────────
        prev_active_ids: set = set()                                # 이전 프레임 활성 ID 집합 (퇴장 감지용)
        last_footpoints: dict = {}                                  # {track_id: (fx, fy)} 마지막 footpoint
        self._gru_pretrain_pending_a = False                         # A방향 GRU pretrain 예약 플래그
        self._gru_pretrain_pending_b = False                         # B방향 GRU pretrain 예약 플래그
        self._gru_feature_history_a = []                             # A방향 GRU pretrain용 feature 누적
        self._gru_feature_history_b = []                             # B방향 GRU pretrain용 feature 누적

        # ── 누적 로그 경로 (flow_map과 같은 폴더) ─────────────────────
        self._log_path_a = (cfg.flow_map_path.parent / "feature_log_a.pkl"
                            if cfg.flow_map_path else None)          # A방향 누적 로그
        self._log_path_b = (cfg.flow_map_path.parent / "feature_log_b.pkl"
                            if cfg.flow_map_path else None)          # B방향 누적 로그
        self._log_interval = getattr(cfg, "gru_log_interval", 3)     # 저장 주기 (프레임)

        # ── 재학습 주기 추적 ──────────────────────────────────────────
        _retrain_sec = getattr(cfg, "gru_retrain_interval_sec", 3600.0)
        self._retrain_interval_frames = int(_retrain_sec * fps)      # 초→프레임
        self._last_retrain_frame = 0                                 # 마지막 재학습 프레임

        # ── detect_only 모드: 기존 로그로 즉시 재학습 시도 ────────────
        if cfg.detect_only and _GRU_AVAILABLE:
            for gru_m, log_p, tag in [
                (self.gru_module_a, self._log_path_a, "A"),
                (self.gru_module_b, self._log_path_b, "B"),
            ]:
                if gru_m is not None and log_p is not None and not gru_m._is_direct_trained:
                    if gru_m.retrain_from_log(log_p):               # 로그에서 재학습
                        if cfg.flow_map_path:                        # weights 저장
                            _pt = cfg.flow_map_path.parent / f"gru_{tag.lower()}.pt"
                            if gru_m.save(_pt):
                                print(f"💾 GRU-{tag} 재학습 weights 저장: {_pt}")

        # ── 학습 연장 상한 계산 ──────────────────────────────────────
        max_learning_frames = int(                                  # 최대 학습 프레임 수
            cfg.learning_frames * cfg.max_learning_extension        # 기본 × 1.5
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

        # ── 출력 창 초기화 (크기 고정) ──────────────────────────────────
        _win_name = "Highway Wrong-Way Detection"               # 창 이름 (imshow와 동일)
        _disp_w = getattr(cfg, "display_width", 1280)           # 표시 너비 (기본 1280)
        _disp_h = getattr(cfg, "display_height", 720)           # 표시 높이 (기본 720)
        cv2.namedWindow(_win_name, cv2.WINDOW_NORMAL)           # 리사이즈 가능한 창 생성
        if _disp_w > 0 and _disp_h > 0:                        # 0이 아닌 경우에만 크기 고정
            cv2.resizeWindow(_win_name, _disp_w, _disp_h)      # 창 크기 고정

        _run_start_time  = time.time()                              # 루프 시작 시각 (max_seconds 계산용)
        _stream_fail_cnt = 0                                        # 연속 read 실패 횟수 (스트림 단절 감지)
        _STREAM_FAIL_MAX = 50                                       # 이 횟수 초과 시 스트림 단절로 판단

        # ── 선제 URL 갱신 상태 (url_refresh_interval 사용 시) ────────────
        _url_timer        = time.time()     # 현재 URL 사용 시작 시각
        _prefetch_lock    = threading.Lock()
        _next_cap         = [None]          # 백그라운드에서 미리 열어둔 새 cap
        _prefetch_started = [False]         # 백그라운드 스레드 실행 여부
        _PREFETCH_AHEAD   = 20.0            # 갱신 N초 전에 미리 열기 시작

        # ── 실제 처리 fps 측정 → jump 임계값 동적 스케일링 ──────────────
        # CCTV는 30fps이지만 CPU 처리 속도가 10fps이면 3프레임치 이동이
        # 1프레임에 한 번에 발생 → 정상 이동도 jump로 오감지됨.
        # 실측 처리fps 기반으로 jump_px를 (stream_fps / proc_fps) 비율로 확대.
        _proc_fps_samples   = []                                    # 최근 N프레임 처리 시간 샘플
        _proc_fps_win       = 30                                    # 측정 윈도우 크기
        _last_frame_time    = time.time()                           # 직전 프레임 처리 완료 시각
        _base_jump_px       = getattr(cfg, "frame_skip_jump_px", 80.0)  # 30fps 기준 jump 임계값
        _jump_thr_dynamic   = _base_jump_px                         # 실측 fps 반영 동적 임계값
        _freeze_frame_count = 0                                      # 연속 정지 프레임 수 (끊김 감지용)
        _adj_diff_history   = []                                     # adj_diff 롤링 평균용 (최근 90프레임)
        _prev_fleet_cos     = None                                   # 직전 프레임 fleet 평균 코사인 (함대 끊김 감지용)

        while cap.isOpened():                                       # 비디오 스트림이 열려 있는 동안
            # ── max_seconds 초과 시 루프 종료 (url_refresh_interval 미사용 시 fallback) ──
            if max_seconds is not None and url_refresh_interval is None:
                if time.time() - _run_start_time >= max_seconds:
                    print(f"[run] {max_seconds:.0f}초 경과 → 루프 종료")
                    break

            # ── 선제 URL 갱신 (url_refresh_interval 설정 시) ─────────────
            if url_refresh_interval and url_refresher and is_stream:
                _url_elapsed = time.time() - _url_timer

                # t = interval-20s: 백그라운드에서 새 URL 발급 + 새 cap 미리 열기
                if not _prefetch_started[0] and _url_elapsed >= url_refresh_interval - _PREFETCH_AHEAD:
                    _prefetch_started[0] = True
                    def _do_prefetch(_lock=_prefetch_lock, _nc_ref=_next_cap):
                        _nu = url_refresher()                   # 새 URL 발급
                        if _nu:
                            _nc = cv2.VideoCapture(_nu)         # 새 스트림 미리 열기
                            with _lock:
                                _nc_ref[0] = _nc if _nc.isOpened() else None
                        else:
                            with _lock:
                                _nc_ref[0] = None
                    threading.Thread(target=_do_prefetch, daemon=True).start()
                    print(f"[run] 새 스트림 사전 준비 시작 (잔여 ~{_PREFETCH_AHEAD:.0f}초)")

                # t = interval: 미리 열어둔 cap으로 교체
                if _prefetch_started[0] and _url_elapsed >= url_refresh_interval:
                    with _prefetch_lock:
                        _nc = _next_cap[0]
                    if _nc is not None:
                        cap.release()
                        cap = _nc
                        print(f"[run] ✅ 스트림 교체 완료 (사전 준비 성공 — 끊김 없음)")
                    else:
                        # 사전 준비 실패 시 동기 fallback
                        print(f"[run] ⚠️ 사전 준비 미완료 → 동기 교체")
                        _nu = url_refresher()
                        if _nu:
                            cap.release()
                            cap = cv2.VideoCapture(_nu)
                    _next_cap[0] = None
                    _prefetch_started[0] = False
                    _url_timer = time.time()                    # 타이머 리셋

            ret, frame = cap.read()                                 # 프레임 읽기
            if not ret:                                             # 프레임 읽기 실패
                if is_stream:
                    _stream_fail_cnt += 1
                    if _stream_fail_cnt >= _STREAM_FAIL_MAX:        # 연속 50회 실패 = 스트림 단절
                        if url_refresher is not None:               # 콜백 있으면 재연결 시도
                            print(f"[run] 스트림 단절 → URL 재발급 시도 (상태 유지)")
                            _new_url = url_refresher()              # 새 URL 발급
                            if _new_url:
                                cap.release()                       # 기존 cap 해제
                                cap = cv2.VideoCapture(_new_url)    # 새 URL로 재연결
                                _stream_fail_cnt = 0               # 카운터 리셋
                                print(f"[run] 재연결 완료 — Detector 상태 유지")
                                continue
                        print(f"[run] 스트림 단절 감지 ({_STREAM_FAIL_MAX}회 연속 실패) → 루프 종료")
                        break                                       # 콜백 없거나 실패 → 종료
                    time.sleep(0.02)
                    continue
                break                                               # 파일이면 종료
            _stream_fail_cnt = 0                                    # 성공 시 실패 카운터 초기화

            # ── 실제 처리 fps 측정 및 jump 임계값 갱신 ──────────────────
            _now = time.time()
            _proc_fps_samples.append(_now - _last_frame_time)
            _last_frame_time = _now
            if len(_proc_fps_samples) > _proc_fps_win:
                _proc_fps_samples.pop(0)
            if len(_proc_fps_samples) >= 5:                         # 5프레임 이상 쌓이면 측정
                _avg_interval = sum(_proc_fps_samples) / len(_proc_fps_samples)
                _proc_fps     = 1.0 / max(_avg_interval, 0.01)     # 실측 처리 fps
                # 스케일 = stream_fps / proc_fps: 10fps 처리 시 3배 확대
                _scale = max(1.0, fps / max(_proc_fps, 1.0))
                _jump_thr_dynamic = _base_jump_px * min(_scale, 5.0)  # 최대 5배 제한

            st.frame_num += 1                                       # 프레임 번호 증가

            # ── YOLO 추적 ──
            tracks = self.tracker.track(frame)                      # [{id, x1, y1, x2, y2, cx, cy}, ...]
            active_ids = {t["id"] for t in tracks}                  # 현재 프레임에 보이는 ID들

            # ── 프레임 스킵(신호 끊김) 감지 ──────────────────────────────
            # HLS 스트림 끊김 시 모든 차량이 동시에 큰 변위를 가짐
            # → 절반 이상이 jump_px 초과 시 해당 프레임의 궤적·학습·판정 전부 스킵
            _jump_thr   = _jump_thr_dynamic                             # 실측 fps 기반 동적 임계값
            _jump_ratio = getattr(cfg, "frame_skip_ratio",    0.5)
            _jump_count = 0
            _jump_total = 0
            for _t in tracks:
                _tid = _t["id"]
                _traj = st.trajectories.get(_tid)
                if _traj:                                           # 이전 위치 있는 차량만 비교
                    _dx = _t["cx"] - _traj[-1][0]
                    _dy = _t["cy"] - _traj[-1][1]
                    _dist = (_dx**2 + _dy**2) ** 0.5
                    _jump_total += 1
                    if _dist > _jump_thr:
                        _jump_count += 1
            _is_frame_skip = (
                _jump_total >= 2                                    # 비교 가능 차량 2대 이상 (1→2: 차량 적은 상황 대응)
                and _jump_count / _jump_total >= _jump_ratio        # 절반 이상 jump
            )
            if _is_frame_skip:
                print(f"[F:{st.frame_num}] ⚠️ 프레임 스킵 감지 ({_jump_count}/{_jump_total}대 jump) → 이 프레임 스킵")
                # 스킵 후 궤적 전체를 현재 위치로 채움
                # velocity 계산: traj[-velocity_window] → traj[-1] 구간을 사용하므로
                # 마지막 점만 바꿔도 나머지 점(jump 전)이 남아 다음 프레임에서도
                # "순간이동 벡터"가 계산되어 역주행 오탐이 이어짐.
                # → 궤적 전체를 현재 위치로 덮어써서 velocity 계산 기점을 초기화.
                st.post_reconnect_frame = st.frame_num              # 재연결 이벤트 기록 (새 차량 포함 보호)
                _freeze_frame_count = 0                             # 끊김 카운터 초기화
                for _t in tracks:
                    _tid = _t["id"]
                    if st.trajectories[_tid]:                       # 기존 궤적 있으면
                        _cur_pos = (_t["cx"], _t["cy"])
                        st.trajectories[_tid] = [_cur_pos] * len(st.trajectories[_tid])  # 전체를 현재 위치로 초기화
                    st.last_velocity.pop(_tid, None)                # 방향 벡터 리셋 (dir_jump 오감지 차단)
                    st.wrong_way_count[_tid] = 0                    # 누적 의심 카운트 리셋
                    st.direction_change_frame[_tid] = st.frame_num  # guard 발동: velocity_window 동안 판정 차단
                    st.wrong_way_ids.discard(_tid)                  # 혹시 스킵 직전에 확정됐으면 취소

            # 처음 등장한 프레임 기록
            for t in tracks:                                        # 각 트랙 순회
                if t["id"] not in st.first_seen_frame:              # 처음 보는 ID이면
                    st.first_seen_frame[t["id"]] = st.frame_num     # 등장 프레임 기록

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
                _lo = {"SMOOTH": 0, "SLOW": 1, "JAM": 2}     # 레벨 순위
                _worst_lvl = (_lvl_a if _lo.get(_lvl_a, 0) >= _lo.get(_lvl_b, 0)  # 더 나쁜 레벨
                              else _lvl_b)
                # worst-of-both 방향의 rule_jam / gru_score 선택 (jam 기준)
                _src = (self.traffic_analyzer_a                     # jam 높은 쪽 analyzer
                        if _jam_a >= _jam_b else self.traffic_analyzer_b)
                _rule_jam = _src.get_rule_jam_score() if _src else 0.0  # rule_jam (블렌딩 전)
                _gru_score = _src.get_gru_score() if _src else None     # gru_score (None 허용)
                self.logger.log_frame(                              # 프레임 로그 기록
                    frame_num=st.frame_num,
                    time_sec=time_sec,
                    active_tracks=len(tracks),
                    wrong_confirmed_count=wrong_confirmed_count,
                    flow_samples_total=self.flow.count.sum(),
                    camera_switch_triggered=False,
                    mode="DETECTING",
                    jam_score=_worst_jam,                           # worst-of-both jam_score
                    congestion_level=_worst_lvl,                    # worst-of-both 레벨
                    rule_jam_score=_rule_jam,                       # rule 기반 jam (블렌딩 전)
                    gru_score=_gru_score                            # GRU 예측값 (warmup 중 None)
                )

            # ── 카메라 전환 감지 및 안정 대기 상태 머신 ─────────────────
            # 상태: 탐지 중 → (전환 감지) → waiting_stable → (안정 확인) → 재학습 → 탐지 중
            # 재학습 중 또 흔들리면 → waiting_stable 복귀 (잘못된 흐름 학습 방지)

            _stability_required_frames = int(
                getattr(cfg, "stability_required_sec", 4.0) * fps)  # 안정 대기 프레임 수
            _stability_thr   = getattr(cfg, "stability_diff_threshold", 8.0)   # 안정 판정 diff 임계값
            _relearn_abort   = getattr(cfg, "relearn_abort_diff", 15.0)        # 재학습 중단 diff 임계값

            if not st.is_learning:
                # ── (A) 탐지 중: 전환 감지 → waiting_stable 진입 ──────────
                if not st.relearning and not st.waiting_stable:
                    if self.switch.check(frame, st.frame_num, st.cooldown_until):
                        print("📷 카메라 전환 감지 → 화면 안정 대기 중...")
                        st.waiting_stable = True
                        st.stable_since_frame = st.frame_num       # 안정 타이머 시작
                        # GRU만 즉시 리셋 (flow_map은 안정 후 재학습 시작 시 초기화)
                        if self.gru_module_a is not None:
                            self.gru_module_a.reset()
                        if self.gru_module_b is not None:
                            self.gru_module_b.reset()
                        self._track_direction.clear()

                # ── (B) 안정 대기 중: diff 모니터링 ────────────────────────
                elif st.waiting_stable:
                    # switch.check()을 호출해서 last_adj_diff를 업데이트
                    self.switch.check(frame, st.frame_num, st.cooldown_until)
                    _cur_diff = self.switch.last_adj_diff

                    if _cur_diff > _stability_thr:                 # 아직 불안정 → 타이머 리셋
                        st.stable_since_frame = st.frame_num
                        if st.frame_num % 30 == 0:
                            print(f"[대기] 아직 불안정 diff={_cur_diff:.1f} > {_stability_thr}")
                    else:
                        # 안정 지속 중 — 충분히 유지됐으면 재학습 시작
                        stable_frames = st.frame_num - st.stable_since_frame
                        if stable_frames >= _stability_required_frames:
                            print(f"✅ 화면 안정 확인 ({stable_frames}프레임) → 재학습 시작")
                            st.waiting_stable = False
                            st.reset_for_relearn()                 # 재학습 모드 진입
                            self.flow.reset()                      # flow_map 초기화
                            self.traffic_analyzer_a.congestion_judge.reset()
                            self.traffic_analyzer_b.congestion_judge.reset()
                            # 메모리 history를 pkl에 먼저 저장 — 전환 전 데이터도 보존
                            if self._log_path_a and self._gru_feature_history_a:
                                self.gru_module_a.append_feature_log(
                                    self._gru_feature_history_a, self._log_path_a)
                            if self._log_path_b and self._gru_feature_history_b:
                                self.gru_module_b.append_feature_log(
                                    self._gru_feature_history_b, self._log_path_b)
                            self._gru_feature_history_a = []
                            self._gru_feature_history_b = []
                            self._gru_pretrain_pending_a = False
                            self._gru_pretrain_pending_b = False
                            self._ref_direction = None
                            _relearn_smoothed_80 = False
                            _relearn_smoothed_95 = False

                # ── (C) 재학습 중: 또 흔들리면 중단 → 대기 복귀 ───────────
                elif st.relearning:
                    self.switch.check(frame, st.frame_num, st.cooldown_until)
                    _cur_diff = self.switch.last_adj_diff
                    if _cur_diff > _relearn_abort:
                        print(f"⚠️ 재학습 중 화면 불안정 (diff={_cur_diff:.1f}) → 재학습 중단, 안정 대기 복귀")
                        st.relearning = False
                        st.waiting_stable = True
                        st.stable_since_frame = st.frame_num
                        self.flow.reset()                          # 오염된 flow_map 초기화
                        _relearn_smoothed_80 = False
                        _relearn_smoothed_95 = False
                        # GRU feature 이력도 초기화 — 재학습 중 수집된 오염 데이터 제거
                        self._gru_feature_history_a = []
                        self._gru_feature_history_b = []
                        self._gru_pretrain_pending_a = False
                        self._gru_pretrain_pending_b = False
                        if self.gru_module_a is not None:
                            self.gru_module_a.reset()
                        if self.gru_module_b is not None:
                            self.gru_module_b.reset()

            # ── 프레임 freeze 감지 (끊김 재연결 감지) ─────────────────────────
            # adj_diff ≈ 0 이 연속되면 카메라 freeze → 이후 정상 복귀 시 재연결 이벤트.
            # _is_frame_skip(차량 displacement 기반)이 미탐지하는 짧은 끊김(1~10초)을 포착.
            # switch.check()는 탐지 모드(not is_learning/relearning/waiting_stable)에서만 호출됨.
            # → same 조건에서만 freeze 카운터를 업데이트해 일관성 유지.
            #
            # ★ 임계값: 상대값 사용 (고정값 사용 시 정체 구간 저속 diff를 freeze로 오감지)
            #   - 정상 흐름:  avg≈5.0 → freeze 임계 = 0.50  (5.0 × 10%)
            #   - 정체 구간:  avg≈0.5 → freeze 임계 = 0.05  (0.5 × 10%)
            #   - 진짜 freeze: adj≈0.00 → 어느 환경에서도 임계 이하
            if not st.is_learning and not st.relearning and not st.waiting_stable:
                _adj_diff_now = self.switch.last_adj_diff           # switch.check()에서 방금 갱신됨

                # 롤링 평균 계산 (최근 90프레임 — camera_switch diff_history와 동일 윈도우)
                _adj_diff_history.append(_adj_diff_now)
                if len(_adj_diff_history) > 90:
                    _adj_diff_history.pop(0)
                _avg_adj = sum(_adj_diff_history) / max(len(_adj_diff_history), 1)

                # 동적 임계값: 평균의 10%, 최소 0.05 (noise floor)
                _freeze_thr  = max(_avg_adj * 0.10, 0.05)
                _min_freeze  = getattr(cfg, "min_freeze_frames", 10)  # 기본 10프레임

                if _adj_diff_now < _freeze_thr:                     # 정지 프레임 (adj ≈ 0)
                    _freeze_frame_count += 1
                else:
                    if _freeze_frame_count >= _min_freeze:          # freeze 구간 종료 = 재연결
                        print(f"[F:{st.frame_num}] 📡 끊김 재연결 감지 "
                              f"({_freeze_frame_count}프레임 정지, avg={_avg_adj:.2f}, "
                              f"thr={_freeze_thr:.3f}) → 역주행 판정 차단 시작")
                        st.post_reconnect_frame = st.frame_num      # 재연결 이벤트 기록
                        # 모든 차량 궤적 초기화 (_is_frame_skip과 동일 처리)
                        for _t in tracks:
                            _tid = _t["id"]
                            if st.trajectories[_tid]:
                                _cur_pos = (_t["cx"], _t["cy"])
                                st.trajectories[_tid] = [_cur_pos] * len(st.trajectories[_tid])
                            st.last_velocity.pop(_tid, None)
                            st.wrong_way_count[_tid] = 0
                            st.direction_change_frame[_tid] = st.frame_num
                            st.wrong_way_ids.discard(_tid)
                    _freeze_frame_count = 0                         # 정상 프레임 → 카운터 초기화

            # ── 초기 학습 완료 처리 ──
            if st.is_learning:                                      # 학습 모드일 때만 체크
                # 학습 완료 조건: learning_frames 도달 또는 최대 프레임 강제 종료
                learning_done = (
                    st.frame_num >= cfg.learning_frames
                    or st.frame_num >= max_learning_frames
                )
                if learning_done:                                   # 학습 완료이면
                    self.flow.apply_spatial_smoothing(verbose=True) # ① 초기 공간 채움
                    self.flow.apply_overlap_erosion(               # ② bbox 겹침 경계 셀 제거
                        contra_threshold=cfg.bbox_contra_threshold
                    )
                    self.flow.apply_direction_repair()             # ③ 프레임 스킵 방향 오류 교정
                    self.flow.apply_spatial_smoothing()            # ④ 재채움 (중앙선 불가침)
                    self._compute_ref_direction()                   # 기준 방향 벡터 계산
                    self._compute_direction_cell_counts()           # 방향별 셀 수 계산 → TA 주입
                    # ── 양방향 채널 구축 (117차) ─────────────────────────────
                    if self._ref_direction is not None:
                        self.flow.build_directional_channels(*self._ref_direction)
                    if cfg.flow_map_path:                           # 저장 경로 있으면
                        self.flow.save(cfg.flow_map_path)           # flow_map만 저장
                        # ref_frame 저장 — 다음 실행 시 자동 매칭에 사용
                        if _MATCHER_AVAILABLE:
                            save_ref_frame(frame, cfg.flow_map_path.parent)
                    st.is_learning = False                          # 학습 모드 종료
                    print(f"학습 완료! (frame={st.frame_num})")
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
                relearn_done = (
                    elapsed >= cfg.relearn_frames                  # 프레임 수 기반 종료
                    or elapsed >= relearn_max                      # 강제 종료
                )
                if relearn_done:                                    # 재학습 완료이면
                    self.flow.apply_spatial_smoothing(verbose=True) # ① 초기 공간 채움
                    self.flow.apply_overlap_erosion(               # ② bbox 겹침 경계 셀 제거
                        contra_threshold=cfg.bbox_contra_threshold
                    )
                    self.flow.apply_direction_repair()             # ③ 프레임 스킵 방향 오류 교정
                    self.flow.apply_spatial_smoothing()            # ④ 재채움 (중앙선 불가침)
                    self._compute_ref_direction()                   # 기준 방향 벡터 재계산
                    self._compute_direction_cell_counts()           # 방향별 셀 수 재계산 → TA 주입
                    # ── 양방향 채널 재구축 (117차) ────────────────────────────
                    if self._ref_direction is not None:
                        self.flow.build_directional_channels(*self._ref_direction)
                    if cfg.flow_map_path:                           # 저장 경로 있으면
                        self.flow.save(cfg.flow_map_path)           # flow_map만 저장 (baseline 미포함)
                        # ref_frame 갱신 — 카메라 전환 후 새 화면으로 매칭 기준 교체
                        if _MATCHER_AVAILABLE:
                            save_ref_frame(frame, cfg.flow_map_path.parent)
                    st.relearning = False                           # 재학습 모드 종료
                    st.cooldown_until = st.frame_num + cfg.cooldown_frames  # 쿨다운 설정
                    self.switch.set_reference(frame)                # 새 기준 프레임 설정
                    print("재학습 완료! 쿨다운 시작")
                    self._gru_pretrain_pending_a = True             # A방향 GRU pretrain 예약
                    self._gru_pretrain_pending_b = True             # B방향 GRU pretrain 예약

            # ── 전차량 fleet cosine 선제 계산 (끊김 감지) ──────────────────────
            # 정상 주행 시: 모든 차량이 flow_map 방향과 cos > 0 → fleet 평균 양수
            # 화면 끊김 후: 차량 위치가 순간이동 → 속도 벡터가 무작위 방향 → fleet cos 급락
            # 이 이벤트를 _is_frame_skip(displacement 기반)과 독립적으로 포착
            #
            # 발동 조건: (1) fleet 평균 < 0.0  (대다수가 역방향 벡터)
            #            (2) 이전 프레임 대비 0.5 이상 급락
            #            (3) 유효 차량 수 >= 3  (소수 샘플 오탐 방지)
            _is_fleet_skip = False
            if not st.is_learning and not st.relearning and not st.waiting_stable:
                _fleet_cos_vals = []
                for _ft in tracks:
                    _ftid  = _ft["id"]
                    _ftraj = st.trajectories[_ftid]
                    if not _ftraj or len(_ftraj) < cfg.velocity_window:
                        continue                                     # 궤적 부족 → 건너뜀
                    _fw   = cfg.velocity_window
                    _fsi  = len(_ftraj) - _fw
                    _fpfx = [_ftraj[_fsi+i+1][0] - _ftraj[_fsi+i][0] for i in range(_fw-1)]
                    _fpfy = [_ftraj[_fsi+i+1][1] - _ftraj[_fsi+i][1] for i in range(_fw-1)]
                    _fvdx = float(np.median(_fpfx)) * (_fw - 1)
                    _fvdy = float(np.median(_fpfy)) * (_fw - 1)
                    _fmag = np.sqrt(_fvdx ** 2 + _fvdy ** 2)
                    _fbh  = max(_ft["y2"] - _ft["y1"], 1)
                    if _fmag / _fbh > cfg.norm_learn_threshold and _fmag > 1.0:
                        _fndx, _fndy = _fvdx / _fmag, _fvdy / _fmag
                        _ffv = self.flow.get_interpolated(_ft["cx"], _ft["cy"])
                        if _ffv is not None:
                            _fleet_cos_vals.append(              # cos(vehicle_vel, flow_map)
                                _fndx * _ffv[0] + _fndy * _ffv[1]
                            )

                if len(_fleet_cos_vals) >= 5:                       # 유효 차량 5대 이상 (소수 오탐 방지)
                    _fleet_cos_avg = sum(_fleet_cos_vals) / len(_fleet_cos_vals)
                    # ① 절대 조건: fleet 평균이 음수 (이전 값 무관 — 재건 중에도 감지 가능)
                    # ② 상대 조건: 직전 안정값 대비 급락 (첫 번째 이벤트용)
                    _fleet_drop    = ((_prev_fleet_cos - _fleet_cos_avg)
                                      if _prev_fleet_cos is not None else 0.0)
                    _fleet_trigger = (
                        (_fleet_cos_avg < -0.1)                     # ① 절대: 대다수 역방향 벡터
                        or (_fleet_drop >= 0.5 and _fleet_cos_avg < 0.2)  # ② 상대: 급락+아직 낮음
                    )
                    if _fleet_trigger:
                        _is_fleet_skip = True
                        _prev_str = f"{_prev_fleet_cos:.2f}" if _prev_fleet_cos is not None else "N/A"
                        print(f"[F:{st.frame_num}] 🚨 fleet cos 급락 "
                              f"({_prev_str} → {_fleet_cos_avg:.2f}, "
                              f"n={len(_fleet_cos_vals)}) → 전차량 궤적 초기화")
                        st.post_reconnect_frame = st.frame_num      # 판정 차단 시작
                        for _ft2 in tracks:
                            _ftid2 = _ft2["id"]
                            if st.trajectories[_ftid2]:
                                _cur = (_ft2["cx"], _ft2["cy"])
                                st.trajectories[_ftid2] = [_cur] * len(st.trajectories[_ftid2])
                            st.last_velocity.pop(_ftid2, None)
                            st.wrong_way_count[_ftid2] = 0
                            st.direction_change_frame[_ftid2] = st.frame_num
                            st.wrong_way_ids.discard(_ftid2)
                        _prev_fleet_cos = None                      # 리셋: 다음 이벤트가 fresh start
                    else:
                        _prev_fleet_cos = _fleet_cos_avg            # 정상 프레임만 갱신

            # ── 차량별 속도 딕셔너리 초기화 ──
            speeds = {}                                             # {tid: mag} — traffic_analyzer용

            # ── 차량별 처리 ──
            for t in tracks:                                        # 각 트랙 순회
                tid = t["id"]                                       # 트랙 ID
                # YOLO 원본 bbox 그대로 사용 (EMA 안정화 제거 — 방향 급변 가드로 jitter 대응)
                x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]  # 원본 bbox 좌표
                cx = (x1 + x2) / 2                                  # bbox 가로 중심
                cy = (y1 + y2) / 2                                  # bbox 세로 중심

                # ── footpoint 계산 (추적 기준점: bbox 중앙) ──────────────
                fx = cx                                              # footpoint x = bbox 중심 x
                fy = cy                                              # footpoint y = bbox 중심 y

                # 마지막 footpoint 갱신 (퇴장 시 on_exit에 사용)
                last_footpoints[tid] = (fx, fy)                     # 최신 footpoint 저장

                # ── 방향 분류 (탐지 모드에서만, 기준 방향 설정 후) ─────
                # 기존: flow_map 기반 → 미학습 셀(상단) 에서 nearest-neighbor가
                #       반대 방향 셀을 반환해 상행 차량을 A로 오분류
                # 개선: 궤적 3점 이상이면 velocity 벡터로 즉시 분류
                #       원거리 차량은 YOLO가 자주 끊겨 traj<20인 경우 많음 →
                #       velocity_window(20) 대기 없이 조기 정확 분류
                if not st.is_learning and not st.relearning and not st.waiting_stable and self._ref_direction is not None:
                    _traj_dir = st.trajectories[tid]                # 이번 프레임 추가 전 궤적
                    _DIR_WIN = min(cfg.velocity_window, len(_traj_dir))  # 가용 최대 window
                    if _DIR_WIN >= 3:                               # 3포인트 이상이면 velocity 사용
                        _ddx = _traj_dir[-1][0] - _traj_dir[-_DIR_WIN][0]  # x 변위
                        _ddy = _traj_dir[-1][1] - _traj_dir[-_DIR_WIN][1]  # y 변위
                        _dmag = np.sqrt(_ddx ** 2 + _ddy ** 2)     # 이동 거리
                        if _dmag > 1.0:                             # 움직임 확인
                            _ref_x, _ref_y = self._ref_direction    # 기준 방향
                            _cos_dir = ((_ddx / _dmag) * _ref_x    # 코사인 유사도
                                        + (_ddy / _dmag) * _ref_y)
                            self._track_direction[tid] = (
                                'a' if _cos_dir >= cfg.lane_cos_threshold else 'b'
                            )
                        else:                                       # 거의 정지 → flow_map fallback
                            self._track_direction[tid] = self._classify_direction(fx, fy)
                    else:                                           # 궤적 없음 → flow_map fallback
                        self._track_direction[tid] = self._classify_direction(fx, fy)

                # ID 재매칭 시도 (학습 모드가 아닐 때만)
                if not st.is_learning and not st.relearning and not st.waiting_stable:        # 탐지 모드일 때만
                    self.idm.check_reappear(tid, cx, cy)            # 재매칭 시도

                # 궤적에 현재 위치 추가 — 첫 등장 3프레임은 YOLO 초기 bbox가 불안정하므로 건너뜀
                # EMA 스무딩 적용: bbox jitter가 velocity 벡터에 미치는 영향 완화
                # alpha=0.4: 현재 40% + 직전 60% → 갑작스러운 위치 튐을 흡수
                # 프레임 스킵 시: 궤적 추가 자체를 건너뜀 → 이상 변위가 궤적에 남지 않음
                _age = st.frame_num - st.first_seen_frame.get(tid, st.frame_num)  # 트랙 경과 프레임

                # 개별 차량 단독 jump 체크 (프레임 스킵 미감지 시에도 1대가 튀는 경우)
                _solo_jump = False
                if st.trajectories[tid]:
                    _prev_fx, _prev_fy = st.trajectories[tid][-1]
                    _solo_dist = ((fx - _prev_fx)**2 + (fy - _prev_fy)**2) ** 0.5
                    if _solo_dist > _jump_thr * 1.2:                # 단독 jump 기준 (1.5→1.2: 더 민감하게)
                        _solo_jump = True
                        # traj 전체를 현재 위치로 덮어씀
                        # → velocity 계산 구간(traj[-window]→traj[-1]) 안에
                        #   jump 전 좌표가 남으면 다음 프레임에도 오탐 벡터가 계산됨
                        _cur_pos = (fx, fy)
                        st.trajectories[tid] = [_cur_pos] * len(st.trajectories[tid])
                        st.last_velocity.pop(tid, None)             # 방향 벡터 리셋
                        st.wrong_way_count[tid] = 0                 # 누적 카운트 리셋
                        st.direction_change_frame[tid] = st.frame_num  # guard 발동: velocity_window 동안 판정 차단
                        st.wrong_way_ids.discard(tid)               # 스킵 직전 확정됐으면 취소

                if _age >= 3 and not _is_frame_skip and not _solo_jump:
                    _traj_cur = st.trajectories[tid]
                    if _traj_cur:                                    # 이전 점이 있으면 EMA 스무딩
                        _px, _py = _traj_cur[-1]
                        fx = 0.4 * fx + 0.6 * _px                  # x EMA (현재 40% + 직전 60%)
                        fy = 0.4 * fy + 0.6 * _py                  # y EMA
                    st.trajectories[tid].append((fx, fy))           # 스무딩된 footpoint 추가
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
                    # ── 중앙값 속도 벡터 (117차) ─────────────────────────────
                    # endpoint-to-endpoint 대신 프레임별 변위의 중앙값 사용.
                    # 단일 프레임 끊김·순간이동(신호 지연 등)이 있어도 중앙값에는
                    # 영향 없음 → fleet_cos/solo_jump 의존 없이 방향 벡터가 견고해짐.
                    _w   = cfg.velocity_window
                    _si  = len(traj) - _w
                    _pfx = [traj[_si+i+1][0] - traj[_si+i][0] for i in range(_w-1)]
                    _pfy = [traj[_si+i+1][1] - traj[_si+i][1] for i in range(_w-1)]
                    vdx  = float(np.median(_pfx)) * (_w - 1)        # 중앙값 × 창 크기 (mag 단위 유지)
                    vdy  = float(np.median(_pfy)) * (_w - 1)
                    mag  = np.sqrt(vdx ** 2 + vdy ** 2)             # 속도 크기 (픽셀)

                    # 프레임당 평균 이동거리 계산 (떨림 필터)
                    avg_move = mag / cfg.velocity_window             # 프레임당 평균 이동

                    speeds[tid] = mag                               # 실제 이동량 기록 — feature_extractor에서 nm으로 정지 판정

                    # ── 속도 벡터 기반 방향 분류 override ───────────────────────
                    # flow_map 기반(_classify_direction)은 미학습 셀(상단)에서
                    # nearest-neighbor가 반대 방향 셀을 반환해 오분류 발생.
                    # velocity 벡터(vdx, vdy)는 실제 이동 방향 → 더 신뢰도 높음.
                    if (not st.is_learning and not st.relearning and not st.waiting_stable
                            and self._ref_direction is not None
                            and mag > 1.0):                         # 최소 이동 확인
                        _vn_x = vdx / mag                          # 정규화 속도 x
                        _vn_y = vdy / mag                          # 정규화 속도 y
                        _ref_x, _ref_y = self._ref_direction       # 기준 방향
                        _cos_v = _vn_x * _ref_x + _vn_y * _ref_y  # 코사인 유사도
                        self._track_direction[tid] = (             # flow_map 분류 덮어쓰기
                            'a' if _cos_v >= cfg.lane_cos_threshold else 'b'
                        )

                    # ── nm 기반 이동 조건 (학습·판정 임계값 분리) ────────────────
                    # 학습:  nm_move > norm_learn_threshold(0.05)  — 서행·정체 차량 방향도 학습
                    # 판정:  judge.check() 내부에서 nm > norm_speed_gate_threshold(0.15) 재검사
                    #        → 학습엔 진입했지만 판정은 내부 게이트에서 필터링
                    #   원거리(bbox_h=30): mag≥1.5px → 학습 가능 (서행 포함)
                    #   근거리(bbox_h=150): mag≥7.5px → 학습 가능
                    _bh = max(y2 - y1, cfg.min_bbox_h)              # bbox_h 클램프
                    _nm_move = mag / _bh                            # 원근 정규화 이동량
                    if _nm_move > cfg.norm_learn_threshold and mag > 1.0:  # 학습 임계값 기준
                        ndx, ndy = vdx / mag, vdy / mag             # 단위 방향 벡터
                        speed = mag                                 # 속도 = 픽셀 이동량
                        speeds[tid] = speed                         # 속도 딕셔너리 갱신 (이미 mag이나 명시적 유지)

                        _learn_min_mag = max(1.0, _bh * cfg.norm_learn_threshold)  # nm 역산 최소 mag (학습·온라인 공용)
                        if _is_frame_skip or _solo_jump or _is_fleet_skip:  # 프레임 스킵·단독 jump·fleet cos 급락 → 학습/판정 스킵
                            pass
                        elif st.is_learning or st.relearning:       # 학습/재학습 모드
                            # 궤적 전체 방향 계산 (traj[0] → 현재 위치)
                            # velocity_window 단기 벡터보다 안정적 → 중앙점 방향 오탐 방지
                            _traj_ndx, _traj_ndy = None, None
                            if len(traj) >= cfg.velocity_window:
                                _tx = fx - traj[0][0]              # 전체 이동 x
                                _ty = fy - traj[0][1]              # 전체 이동 y
                                _tmag = np.sqrt(_tx ** 2 + _ty ** 2)
                                if _tmag >= cfg.min_move_distance:  # 충분히 이동한 경우만
                                    _traj_ndx = _tx / _tmag         # 궤적 단위 방향 x
                                    _traj_ndy = _ty / _tmag         # 궤적 단위 방향 y

                            # ── bbox 수평 폭 클리핑 (중앙선 침범 방지) ────────
                            # max 반폭 = bbox_h × ratio → 차선 폭 범위 내로 제한
                            # 실제 bbox가 더 좁으면 클리핑 없음 (min으로 자연 처리)
                            _bh_learn = max(y2 - y1, 1)
                            _bcx_learn = (x1 + x2) / 2
                            _hw_limit = _bh_learn * getattr(cfg, 'bbox_learn_w_ratio', 0.8)
                            _x1_learn = max(x1, _bcx_learn - _hw_limit)
                            _x2_learn = min(x2, _bcx_learn + _hw_limit)

                            self.flow.learn_step(                   # 흐름장 업데이트
                                traj[-cfg.velocity_window][0],
                                traj[-cfg.velocity_window][1],
                                fx, fy, _learn_min_mag,
                                bbox=(_x1_learn, y1, _x2_learn, y2),  # 폭 제한된 bbox
                                traj_ndx=_traj_ndx,                # 궤적 방향 (중앙점 학습용)
                                traj_ndy=_traj_ndy
                            )
                        else:                                       # 감지 모드
                            # 역주행 여부 판단 (bbox_h 전달 — nm 기반 속도 게이트용)
                            _bbox_h = max(y2 - y1, 1)               # bbox 높이 (원근 정규화용)
                            is_wrong, _, debug_info = self.judge.check(
                                tid, traj, ndx, ndy, mag, cy, _bbox_h,
                                track_dir=self._track_direction.get(tid)
                            )

                            # ── 이웃 차량 방향 일치 확인 (118차 — neighbor_agreement_guard) ──
                            # 진짜 역주행: 이 차량만 반대 방향, 같은 분류 이웃은 정방향
                            # 오탐(flow map 오류·오염): 같은 분류 이웃 차량들도 같은 방향으로 이동 중
                            # → 이웃 N대 이상이 같은 방향이면 flow map이 틀린 것으로 판단 → 취소
                            _nbr_min   = getattr(cfg, "neighbor_guard_min_total", 3)
                            _nbr_agree = getattr(cfg, "neighbor_guard_agree",     2)
                            _sus_dir   = self._track_direction.get(tid)
                            if is_wrong and (ndx != 0.0 or ndy != 0.0) and _sus_dir is not None:
                                _same_dir  = 0
                                _total_nbr = 0
                                for _ov, _ovv in st.last_velocity.items():
                                    if _ov == tid or _ov in st.wrong_way_ids:
                                        continue
                                    if self._track_direction.get(_ov) != _sus_dir:
                                        continue            # 같은 방향 분류 차량만 비교
                                    _total_nbr += 1
                                    if float(ndx * _ovv[0] + ndy * _ovv[1]) > 0.5:
                                        _same_dir += 1
                                if _total_nbr >= _nbr_min and _same_dir >= _nbr_agree:
                                    st.wrong_way_ids.discard(tid)
                                    st.wrong_way_count[tid] = 0
                                    st.first_suspect_frame.pop(tid, None)
                                    is_wrong = False
                                    print(f"   ✅ ID:{tid} 이웃 {_same_dir}/{_total_nbr}대 "
                                          f"동방향 → 역주행 취소 (flow map 오탐 추정)")
                                    # ── 119차 cascade reset ────────────────────────────────
                                    # 이웃 중 의심 누적 중인 같은 방향 차량도 함께 초기화
                                    # 이유: W1 취소 직후 W2·W3이 연속 확정되는 패턴 방지
                                    for _ov2, _ovv2 in list(st.last_velocity.items()):
                                        if _ov2 == tid or _ov2 in st.wrong_way_ids:
                                            continue
                                        if self._track_direction.get(_ov2) != _sus_dir:
                                            continue
                                        if float(ndx * _ovv2[0] + ndy * _ovv2[1]) > 0.5:
                                            if st.wrong_way_count.get(_ov2, 0) > 0:
                                                st.wrong_way_count[_ov2] = 0
                                                st.first_suspect_frame.pop(_ov2, None)
                                                print(f"   ↩️  ID:{_ov2} 연쇄 의심 초기화 "
                                                      f"(cascade from ID:{tid})")

                            # 역주행 확정 시 라벨 부여
                            if is_wrong and tid in st.wrong_way_ids:
                                self.idm.assign_label(tid)          # W1, W2... 라벨 배정
                else:
                    # 궤적이 짧더라도 이미 역주행 확정된 차량이면 그대로 표시
                    if tid in st.wrong_way_ids:                     # 이미 확정된 역주행 차량
                        is_wrong = True                             # 역주행 플래그 유지
                        debug_info = {"status": "CONFIRMED", "cos_values": []}

                # ── 트랙 단위 로그 생성 ────────────────────────────────────
                _log_bh = max(y2 - y1, cfg.min_bbox_h)              # 로그용 bbox_h 클램프
                nm_spd = speed / _log_bh if _log_bh > 0 else 0     # 로그용 nm 속도
                speed_thr = cfg.norm_speed_gate_threshold           # nm 게이트 임계값 (로그용)

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

                if is_wrong_display:                                # 역주행 차량은 항상 경고 표시
                    self.vis.draw_wrong_way_alert(frame, tid, x1, y1, x2, y2)      # 경고 표시
                elif self.vis.show_bbox:                            # 정상 차량은 B키 ON일 때만
                    self.vis.draw_normal_box(frame, tid, x1, y1, x2, y2)           # 초록 박스

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
            exit_count_a = 0                                        # A방향 퇴장 수
            exit_count_b = 0                                        # B방향 퇴장 수

            for gone_id in gone_ids:                                # 퇴장 ID 순회
                # 방향별 퇴장 수 집계 (방향 정보 가져오고 제거)
                _gone_dir = self._track_direction.pop(gone_id, 'a') # 방향 꺼내며 매핑 정리
                if _gone_dir == 'a':                                # A방향이면
                    exit_count_a += 1                               # A 퇴장 수 증가
                else:                                               # B방향이면
                    exit_count_b += 1                               # B 퇴장 수 증가

                last_footpoints.pop(gone_id, None)                  # footpoint 기록 정리

            # ── 방향별 차량 분리 (정체 탐지용) ──────────────────────────
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
                    if mag_val is not None:                         # 속도 있으면
                        speeds_a[tid_val] = mag_val                 # A방향 속도 딕셔너리
                else:                                               # B방향이면
                    norm_mags_b.append(nm)                          # B방향 통계 추가
                    cy_vals_b.append(cy_val)
                    bbox_h_vals_b.append(bh)
                    tracks_b.append(t)                              # B방향 차량 목록
                    if mag_val is not None:                         # 속도 있으면
                        speeds_b[tid_val] = mag_val                 # B방향 속도 딕셔너리

            # ── 이전 프레임 활성 ID 갱신 ─────────────────────────────────
            prev_active_ids = active_ids.copy()                     # 다음 프레임 비교용으로 저장

            # ── 방향별 TrafficAnalyzer·GRU 갱신 ──────────────────────────
            if (self.traffic_analyzer_a is not None                 # 초기화 완료 확인
                    and not st.is_learning and not st.relearning and not st.waiting_stable  # 탐지 모드일 때만
                    and not _is_frame_skip):                        # 프레임 스킵(순간이동) 프레임은 jam_score 업데이트 차단
                # A방향 정체 탐지 갱신
                self.traffic_analyzer_a.update(tracks_a, speeds_a, st.frame_num)
                self.predictor_a.update(self.traffic_analyzer_a.get_avg_speed())
                # B방향 정체 탐지 갱신
                self.traffic_analyzer_b.update(tracks_b, speeds_b, st.frame_num)
                self.predictor_b.update(self.traffic_analyzer_b.get_avg_speed())

                # ── GRU 로그 수집 신뢰도 판단 ────────────────────────────
                # 탐지 차량이 너무 적거나(야간·안개) 프레임이 너무 어두우면
                # feature가 실제 교통 상황을 반영하지 못함 → 로그 스킵
                _min_veh = getattr(cfg, "gru_min_vehicles_for_log", 3)
                _min_bri = getattr(cfg, "gru_min_brightness_for_log", 0.0)
                _total_tracks = len(tracks_a) + len(tracks_b)       # 전체 탐지 차량 수
                _brightness_ok = True
                if _min_bri > 0:                                    # 밝기 필터 활성 시
                    import numpy as _np
                    _gray_mean = float(_np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    _brightness_ok = (_gray_mean >= _min_bri)
                _feature_reliable = (_total_tracks >= _min_veh and _brightness_ok)

                # ── feature 누적 (A방향) — pretrain·로그 공용 ─────────────
                # 신뢰도 필터 통과 시에만 수집 — 오학습 방지
                if self.gru_module_a is not None and st.frame_num % self._log_interval == 0:
                    feat_a = self.traffic_analyzer_a.get_last_feature()
                    if feat_a is not None and _feature_reliable:
                        self._gru_feature_history_a.append(feat_a)  # pretrain용 메모리 누적

                # ── feature 누적 (B방향) ──────────────────────────────────
                if self.gru_module_b is not None and st.frame_num % self._log_interval == 0:
                    feat_b = self.traffic_analyzer_b.get_last_feature()
                    if feat_b is not None and _feature_reliable:
                        self._gru_feature_history_b.append(feat_b)

                # ── GRU pretrain / 재학습 (A방향) ────────────────────────
                if self.gru_module_a is not None:
                    _hist_len_a = len(self._gru_feature_history_a)
                    _need_pretrain_a = (                             # 최초 pretrain 조건
                        self._gru_pretrain_pending_a
                        and _hist_len_a >= self.gru_module_a._pretrain_min_frames
                    )
                    _need_retrain_a = (                              # 주기적 재학습 조건
                        self.gru_module_a._is_direct_trained         # 이미 한 번 학습됨
                        and (st.frame_num - self._last_retrain_frame)
                            >= self._retrain_interval_frames         # 재학습 주기 도달
                        and self._log_path_a is not None
                    )
                    if _need_pretrain_a or _need_retrain_a:
                        # 디스크 로그에 이번 세션 누적분 먼저 저장
                        if self._log_path_a:
                            total_a = self.gru_module_a.append_feature_log(
                                self._gru_feature_history_a, self._log_path_a)
                            print(f"[GRU-A] 로그 저장: 이번세션 {_hist_len_a}개 / 누적 {total_a}개 "
                                  f"({total_a / max(fps, 1) / 60:.1f}분)")
                            # 전체 누적 로그로 학습 (이번 세션 + 과거 세션)
                            losses_a = self.gru_module_a.retrain_from_log(self._log_path_a)
                        else:
                            losses_a = self.gru_module_a.pretrain(self._gru_feature_history_a)
                            if losses_a:
                                print(f"🧠 GRU-A pretrain 완료: loss {losses_a[0]:.4f}→{losses_a[-1]:.4f}")
                        self._gru_feature_history_a = []             # 메모리 비우기
                        self._gru_pretrain_pending_a = False
                        self._last_retrain_frame = st.frame_num      # 재학습 시각 갱신
                        if cfg.flow_map_path:                        # weights 저장
                            _save_a = cfg.flow_map_path.parent / "gru_a.pt"
                            if self.gru_module_a.save(_save_a):
                                print(f"💾 GRU-A weights 저장: {_save_a}")

                # ── GRU pretrain / 재학습 (B방향) ────────────────────────
                if self.gru_module_b is not None:
                    _hist_len_b = len(self._gru_feature_history_b)
                    _need_pretrain_b = (
                        self._gru_pretrain_pending_b
                        and _hist_len_b >= self.gru_module_b._pretrain_min_frames
                    )
                    _need_retrain_b = (
                        self.gru_module_b._is_direct_trained
                        and (st.frame_num - self._last_retrain_frame)
                            >= self._retrain_interval_frames
                        and self._log_path_b is not None
                    )
                    if _need_pretrain_b or _need_retrain_b:
                        if self._log_path_b:
                            total_b = self.gru_module_b.append_feature_log(
                                self._gru_feature_history_b, self._log_path_b)
                            print(f"[GRU-B] 로그 저장: 이번세션 {_hist_len_b}개 / 누적 {total_b}개 "
                                  f"({total_b / max(fps, 1) / 60:.1f}분)")
                            losses_b = self.gru_module_b.retrain_from_log(self._log_path_b)
                        else:
                            losses_b = self.gru_module_b.pretrain(self._gru_feature_history_b)
                            if losses_b:
                                print(f"🧠 GRU-B pretrain 완료: loss {losses_b[0]:.4f}→{losses_b[-1]:.4f}")
                        self._gru_feature_history_b = []
                        self._gru_pretrain_pending_b = False
                        if cfg.flow_map_path:
                            _save_b = cfg.flow_map_path.parent / "gru_b.pt"
                            if self.gru_module_b.save(_save_b):
                                print(f"💾 GRU-B weights 저장: {_save_b}")

                # ── GRU online_step: 매 프레임 현재 레벨로 실시간 학습 ──────
                # SMOOTH만 학습하던 방식 → 전체 레벨 학습으로 확장
                # 이유: 하루종일 실행 시 아침 러시(JAM), 낮(SMOOTH), 저녁 러시(SLOW) 등
                #       다양한 패턴을 실시간으로 반영해야 예측 정확도가 올라감
                _level_map = {"SMOOTH": 0, "SLOW": 1, "JAM": 2}
                if self.gru_module_a is not None and _feature_reliable:  # 신뢰 구간만 학습
                    _lv_a = self.traffic_analyzer_a.get_congestion_level()
                    self.gru_module_a.online_step(label=_level_map[_lv_a])
                if self.gru_module_b is not None and _feature_reliable:  # 신뢰 구간만 학습
                    _lv_b = self.traffic_analyzer_b.get_congestion_level()
                    self.gru_module_b.online_step(label=_level_map[_lv_b])

                # ── flow_map speed_ref 온라인 학습 (SMOOTH 구간만) ────────
                # SMOOTH 구간의 nm을 셀별로 EMA 축적 → 위치별 정상속도 기준 확보
                # 이후 feature_extractor에서 velocity_deficit = 1 - nm/speed_ref 계산에 사용
                for t in tracks:                                    # 활성 차량 순회
                    _tid = t["id"]
                    _mag = speeds.get(_tid)                         # 속도 (없으면 None=신규)
                    if _mag is None or _mag <= 0:                   # 신규·정지 차량 제외
                        continue
                    _bh_ref = max(t["y2"] - t["y1"], cfg.min_bbox_h)  # bbox_h 클램프
                    _nm_ref = _mag / _bh_ref                        # normalized_mag
                    _dir_ref = self._track_direction.get(_tid, 'a') # 차량 방향
                    # 방향별 SMOOTH 레벨일 때만 학습
                    _lvl = (self.traffic_analyzer_a.get_congestion_level()
                            if _dir_ref == 'a'
                            else self.traffic_analyzer_b.get_congestion_level())
                    if _lvl == "SMOOTH":                            # SMOOTH 구간만 학습
                        _fx = t.get("fx", t["cx"])                  # footpoint x
                        _fy = t.get("fy", t["y2"])                  # footpoint y
                        self.flow.learn_baseline(_fx, _fy, _nm_ref) # 셀별 정상 속도 EMA 갱신

            # ── 트랙 정리 ──
            if st.frame_num % 30 == 0:                              # 30프레임마다
                self.idm.cleanup(active_ids)                        # ID 관리자 정리

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
                cv2.putText(frame, f"LEARNING FLOW MAP: {progress:.0f}%",
                            (fw // 2 - 240, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2, cv2.LINE_AA)

            # 안정 대기 중이면 화면 상단에 대기 텍스트 표시
            elif st.waiting_stable:                                 # 안정 대기 모드이면
                stable_frames = st.frame_num - st.stable_since_frame
                remain = max(0, _stability_required_frames - stable_frames)
                remain_sec = remain / max(fps, 1)
                cv2.putText(frame, f"CAMERA MOVED - WAITING STABLE: {remain_sec:.1f}s",
                            (fw // 2 - 240, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            # 재학습 중이면 화면 상단에 상태 텍스트 표시
            elif st.relearning:                                     # 재학습 모드이면
                elapsed = st.frame_num - st.relearn_start_frame     # 경과 프레임
                progress = min(100, elapsed / cfg.relearn_frames * 100)  # 진행률 (%)
                cv2.putText(frame, f"CAMERA SWITCHED - RE-LEARNING: {progress:.0f}%",
                            (fw // 2 - 220, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            # ── 방향별 정체 상태 패널 표시 (좌하단) ────────────────────
            if st.is_learning or st.relearning or st.waiting_stable:  # 학습·재학습·대기 중이면
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
                    # ── 미래 예측 패널 (1·3·5분 후) ─────────────────────
                    pred_a = self.traffic_analyzer_a.get_direct_prediction()
                    pred_b = self.traffic_analyzer_b.get_direct_prediction()
                    self.vis.draw_prediction_panel(
                        frame, pred_a, pred_b,
                        label_a=self._dir_label_a                   # A방향 레이블로 Down/Up 자동 배치
                    )

            # FPS 계산 및 표시
            curr_time = time.time()                                 # 현재 시간
            show_fps = 1 / (curr_time - prev_time + 1e-6)          # FPS 계산
            prev_time = curr_time                                   # 시간 갱신
            fps_y = 255 if self.vis.show_info_panel else 30         # 패널 유무에 따라 y 위치
            cv2.putText(frame, f"FPS: {show_fps:.1f}", (10, fps_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)                                 # 결과 영상 파일에 프레임 기록
            cv2.imshow(_win_name, frame)                            # 화면에 출력 (고정 창)

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
        if writer:
            writer.release()                                        # 비디오 라이터 해제
        if not is_stream:                                           # 파일 재생 완료 시에만 창 닫기
            cv2.destroyAllWindows()                                 # 스트림은 URL 갱신 후 재사용하므로 유지

        # 학습이 완료되지 않은 채로 종료된 경우 마지막 flow_map 저장
        if not st.is_learning and cfg.flow_map_path:                # 학습 완료 상태이면
            self.flow.save(cfg.flow_map_path)                       # flow_map 저장 (baseline 없이)

        # ── 세션 종료 시 미저장 feature 로그 flush ──────────────────
        # 재학습 주기에 도달하지 않은 채 종료되더라도 이번 세션 데이터를 보존
        for gru_m, hist, log_p, tag in [
            (self.gru_module_a, self._gru_feature_history_a, self._log_path_a, "A"),
            (self.gru_module_b, self._gru_feature_history_b, self._log_path_b, "B"),
        ]:
            if gru_m is not None and hist and log_p is not None:
                total = gru_m.append_feature_log(hist, log_p)
                print(f"[GRU-{tag}] 세션 종료 — "
                      f"이번 {len(hist)}개 저장 / 누적 {total}개 "
                      f"({total / max(fps, 1) / 60:.1f}분 / "
                      f"{gru_m._pretrain_min_frames / max(fps, 1) / 60:.1f}분 필요)")

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

        if save_path:
            print(f"\n✅ 저장 완료: {save_path} ({st.frame_num} 프레임)")
        else:
            print(f"\n✅ 완료 ({st.frame_num} 프레임, 녹화 없음)")
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
