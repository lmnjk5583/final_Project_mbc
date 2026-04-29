# 파일 경로: 최종 프로젝트/src/detector.py
# 역할: 모든 모듈을 조립하고 run() 루프를 실행하는 메인 오케스트레이터.

import cv2                                          # OpenCV — 영상 입출력·시각화
import numpy as np                                  # 수치 계산
import time                                         # FPS 측정용 타이머
import threading                                    # URL 선제 갱신용 백그라운드 스레드
from datetime import datetime                       # HistoricalPredictor 시각 슬롯용

from .config import DetectorConfig                  # 모든 파라미터가 담긴 설정 클래스
from .state import DetectorState                    # 프레임 번호·궤적·역주행 카운트 등 런타임 상태
from .flow_map import FlowMap                       # 20x20 그리드 정상 흐름 벡터 학습/비교
from .tracker import YoloTracker                    # YOLO 검출 + ByteTrack 추적
from .judge import WrongWayJudge                    # 코사인 유사도 + 투표/히스테리시스 역주행 판정
from .id_manager import IDManager                   # W라벨 관리 + occlusion 재매칭 + 오래된 트랙 정리
from .camera_switch import CameraSwitchDetector     # 장면/카메라 전환 감지 (grayscale diff)
from .visualizer import Visualizer                  # 시각화(박스/궤적/패널/디버그)
from .logger import CSVLogger                       # 프레임/트랙/이벤트 CSV 로그 저장
from .traffic_analyzer import TrafficAnalyzer, CongestionPredictor  # 정체 탐지 + 단기 예측
from .historical_predictor import HistoricalPredictor               # 시각별 과거 jam 이력 예측

try:
    from flow_map_matcher import save_flow_snapshot, find_best_snapshot, load_snapshot_meta  # flow_map 스냅샷
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
            max_cross_flow_cells=getattr(cfg, "max_cross_flow_cells", 1.2),
        )
        self.tracker = YoloTracker(cfg.model_path, cfg.conf, cfg.target_classes,
                                   night_enhance=getattr(cfg, "night_enhance", True))  # YOLO+ByteTrack
        self.judge = WrongWayJudge(cfg, self.flow, self.state)      # 역주행 판정기
        self.idm = IDManager(cfg, self.flow, self.state)            # ID 관리 + 재매칭
        self.switch = CameraSwitchDetector(cfg)                     # 카메라 전환 감지기
        self.vis = Visualizer(cfg, self.state, self.flow)           # 시각화 모듈
        self.logger = CSVLogger(cfg.log_dir) if cfg.log_dir else None  # CSV 로거 (log_dir 없으면 None)
        # ── 방향별 TrafficAnalyzer/Predictor (run()에서 초기화) ──────
        # frame 크기(fw, fh)와 fps는 run()에서 영상을 열어야 확정되므로
        # __init__ 시점에서는 None으로 두고 run() 진입 직후 초기화한다.
        self.traffic_analyzer_a = None                              # A방향 정체 탐지 (run()에서 초기화)
        self.traffic_analyzer_b = None                              # B방향 정체 탐지 (run()에서 초기화)
        self.predictor_a = None                                     # A방향 정체 예측 (run()에서 초기화)
        self.predictor_b = None                                     # B방향 정체 예측 (run()에서 초기화)

        # ── 방향 분류 기준 벡터 + 차량별 방향 매핑 ────────────────────
        self._ref_direction = None                                  # 전역 기준 방향 벡터 (학습 완료 시 계산)
        self._prev_ref_direction = None                             # 재학습 직전 방향 벡터 (방향 반전 감지용)
        self._prev_dir_label_a = None                               # 재학습 직전 A방향 라벨 (Up/Down 반전 감지용)
        self._track_direction = {}                                  # {tid: 'a' or 'b'} 차량별 방향
        self._wrongway_stable_until = 0                             # 재학습 후 역주행 판정 유예 종료 프레임
        self._dir_label_a = "상행"                                  # A방향 표시 레이블 (기본값)
        self._dir_label_b = "하행"                                  # B방향 표시 레이블 (기본값)
        self._valid_cells_a: int = 1                                # A방향 유효 셀 수 (flow_occupancy 정규화 분모)
        self._valid_cells_b: int = 1                                # B방향 유효 셀 수

        # ── flow_map 로드 (탐지 전용이라면 필수) ────────────────────
        if cfg.flow_map_path:                                       # flow_map 경로가 설정되어 있으면
            if not cfg.detect_only:                                 # 학습 모드 — run()에서 스냅샷 자동 매칭
                self.state.is_learning = True                       # 일단 학습 모드로 설정 (스냅샷 없으면 학습)
                print("detect_only=False → run() 시작 시 스냅샷 자동 매칭 시도")
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
                # ── 123차: 로드 후 중앙선 경계 셀 침식 + 재채움 ──────────────
                # _bbox_contra_count는 npy에 저장되지 않으므로 apply_overlap_erosion은
                # 로드 시 효과 없음. apply_boundary_erosion은 방향 벡터 일관성만으로
                # 경계를 판정하므로 저장된 npy에도 적용 가능.
                # 중앙선 인접 셀(코사인 급변 셀)을 삭제 → get_interpolated None 반환
                # → vote skip → 오탐 원천 차단.
                self.flow.apply_boundary_erosion()                  # 방향 불일치 경계 셀 제거
                self.flow.apply_spatial_smoothing()                 # 침식 후 빈 셀 재채움
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
        """flow_map 전체 셀의 샘플 수 가중 평균 벡터를 기준 방향으로 설정한다.

        기존 방식(최다 샘플 단일 셀)은 재학습마다 어느 차선이 바빴느냐에 따라
        결과가 달라져 UP/DOWN 레이블이 회차마다 뒤집히는 문제가 있었다.
        가중 평균은 왕복 차선 전체를 반영하므로 더 안정적이다.
        왕복 도로에서 양방향 벡터가 상쇄돼 합이 0에 가까울 경우
        단일 최다 셀을 fallback으로 사용한다.
        """
        grid = self.flow                                          # FlowMap 참조

        # ── 가중 평균 방향 계산 ──────────────────────────────────────
        _mask = grid.count > 3                                    # 의미 있는 셀만 (노이즈 제거)
        vx, vy = 0.0, 0.0
        _best_r, _best_c, _best_count = 0, 0, 0                  # fallback용 최다 샘플 셀
        if _mask.any():
            _vx_arr = grid.flow[_mask, 0]
            _vy_arr = grid.flow[_mask, 1]
            _w_arr  = grid.count[_mask].astype(float)
            vx = float(np.average(_vx_arr, weights=_w_arr))      # 샘플 수 가중 평균 x
            vy = float(np.average(_vy_arr, weights=_w_arr))      # 샘플 수 가중 평균 y
        # fallback: 최다 샘플 셀 (가중 평균이 0에 수렴할 경우)
        for r in range(grid.grid_size):
            for c in range(grid.grid_size):
                if grid.count[r, c] > _best_count:
                    _best_count = grid.count[r, c]
                    _best_r, _best_c = r, c

        mag = np.sqrt(vx**2 + vy**2)                              # 가중 평균 벡터 크기
        if mag > 0.1:                                             # 유효한 가중 평균이면 사용
            self._ref_direction = (vx / mag, vy / mag)
            _src = "가중평균"
        else:
            # 왕복 차선에서 벡터 상쇄 → 단일 최다 셀 fallback
            _fx = float(grid.flow[_best_r, _best_c, 0])
            _fy = float(grid.flow[_best_r, _best_c, 1])
            _fm = np.sqrt(_fx**2 + _fy**2)
            if _fm > 1e-6:
                self._ref_direction = (_fx / _fm, _fy / _fm)
            else:
                self._ref_direction = (1.0, 0.0)                  # 최후 fallback
            _src = f"단일셀[{_best_r},{_best_c}]"

        # ── UP/DOWN 레이블 자동 판별 ────────────────────────────────
        # 카메라 좌표계: 이미지 위 = y 감소(vy < 0) = 화면 상 위로 이동 = UP
        #               이미지 아래 = y 증가(vy > 0) = 화면 상 아래로 이동 = DOWN
        ref_vy = self._ref_direction[1]                           # A방향 y성분
        _auto_label_a = "UP" if ref_vy < 0 else "DOWN"

        self._dir_label_a = _auto_label_a
        self._dir_label_b = "DOWN" if _auto_label_a == "UP" else "UP"

        print(f"🧭 기준 방향: ({self._ref_direction[0]:.3f}, {self._ref_direction[1]:.3f})"
              f" [{_src}, 최다셀=({_best_r},{_best_c}) n={_best_count}]"
              f" → A={self._dir_label_a}, B={self._dir_label_b}")

    # ==================== 방향별 유효 셀 수 계산 ====================
    def _compute_direction_cell_counts(self):
        """flow_map 학습 셀을 A/B방향으로 분류해 방향별 유효 셀 수를 계산하고 TrafficAnalyzer에 주입한다.

        FeatureExtractor는 dwell_cell_ratio·flow_occupancy·cell_dwell_score 계산 시
        전체 그리드가 아닌 방향별 실제 도로 면적(셀 수)을 분모로 사용한다.
        예: 왕복 2차선에서 A방향 40셀·B방향 35셀이면 A방향 피처는 40을 분모로 정규화.
        이를 위해 _compute_ref_direction() 이후에 호출해 방향별 셀 수를 확정한다.

        분류 기준: 기준 방향(ref_direction)과의 코사인 유사도가
          lane_cos_threshold 이상이면 A방향, 미만이면 B방향으로 분류.
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

    # ==================== 스냅샷 디렉터리 ====================
    def _snapshot_dir(self) -> "Path | None":
        """camera_id가 설정된 경우 flow_map_path.parent/camera_id/ 를 반환.

        camera_id가 비어 있으면 flow_map_path.parent 를 그대로 반환.
        flow_map_path가 None이면 None.
        """
        if self.cfg.flow_map_path is None:
            return None
        base = self.cfg.flow_map_path.parent
        cam_id = getattr(self.cfg, "camera_id", "").strip()
        return base / cam_id if cam_id else base

    # ==================== 차량 격자 마스크 ====================
    @staticmethod
    def _make_vehicle_grid(tracks: list, frame_w: int, frame_h: int,
                           grid_size: int) -> "np.ndarray":
        """현재 프레임의 탐지 차량 bbox 중심점을 flow_map 격자(grid_size×grid_size)에
        투영한 bool 마스크를 반환한다.

        스냅샷 매칭(find_best_snapshot)에서 카메라 전환 여부를 판별하는 용도로 사용된다.
        같은 카메라라면 도로 구조가 동일하므로 차량이 매 프레임 비슷한 셀에 집중되고,
        저장된 스냅샷의 vehicle_grid와 IoU가 높게 유지된다.
        카메라가 전환되면 차량 위치 패턴이 달라져 IoU가 낮아지므로 매칭 후보에서 탈락한다.

        Args:
            tracks: 현재 프레임 탐지 결과 (각 항목에 x1·y1·x2·y2 키 포함).
            frame_w: 프레임 너비 (픽셀).
            frame_h: 프레임 높이 (픽셀).
            grid_size: flow_map 격자 한 변의 셀 수.

        Returns:
            (grid_size, grid_size) bool ndarray.
            차량 중심점이 투영된 셀은 True, 나머지는 False.
        """
        import numpy as _np
        mask = _np.zeros((grid_size, grid_size), dtype=bool)
        if not tracks or frame_w <= 0 or frame_h <= 0:
            return mask
        cw = frame_w / grid_size
        ch = frame_h / grid_size
        for t in tracks:
            cx = (t["x1"] + t["x2"]) / 2.0
            cy = (t["y1"] + t["y2"]) / 2.0
            r = int(min(cy / ch, grid_size - 1))
            c = int(min(cx / cw, grid_size - 1))
            mask[max(0, r), max(0, c)] = True
        return mask

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

        # ── 스냅샷 자동 매칭: 이전 학습 결과가 있으면 로드하고 학습 스킵 ──
        # detect_only=False 여도 타임스탬프 스냅샷이 존재하면 자동 로드해 학습을 건너뜀.
        # 첫 프레임을 peek해서 ref_frame 유사도를 비교 → 가장 유사한 npy 로드.
        # ★ 즉시 학습 스킵 하지 않음 — 루프 진입 후 45프레임 차량 흐름 검증 후 확정.
        _snap_dir = self._snapshot_dir()
        _startup_snap_npy = None                                     # 시작 스냅샷 후보 (검증 대기)
        if (_snap_dir is not None and not cfg.detect_only
                and st.is_learning and _MATCHER_AVAILABLE):
            # 스트림 첫 프레임은 버퍼링/I-frame 미수신으로 품질이 낮을 수 있음
            # → 최대 10프레임 읽어 마지막으로 성공한 프레임을 매칭에 사용
            # prev_frame도 함께 보관 → optical flow 기반 방향 추정 보조
            _peek_frame = None
            _peek_prev_frame = None
            for _ in range(10):
                _ret_peek, _f = cap.read()
                if _ret_peek and _f is not None:
                    _peek_prev_frame = _peek_frame
                    _peek_frame = _f
                else:
                    break
            if _peek_frame is not None:
                _cam_label = getattr(cfg, "camera_id", "").strip() or _snap_dir.name
                print(f"[스냅샷] 이전 학습 스냅샷 검색 중... (범위: {_cam_label})")
                _best_npy, _snap_score = find_best_snapshot(
                    _peek_frame, _snap_dir, min_score=0.75,
                    prev_frame=_peek_prev_frame
                )
                if _best_npy is not None and self.flow.load(_best_npy):
                    self.flow.speed_ref[:] = 0
                    self.flow.apply_boundary_erosion()
                    self.flow.apply_spatial_smoothing()
                    self._compute_ref_direction()
                    self._compute_direction_cell_counts()
                    if self._ref_direction is not None:
                        self.flow.build_directional_channels(*self._ref_direction)
                    # is_learning=False 는 루프 진입 후 차량 흐름 검증 통과 시 확정
                    # (메타 방향 반전 체크도 hist_pred 초기화 후인 루프에서 수행)
                    _startup_snap_npy = _best_npy
                    print(f"[스냅샷] 후보 로드: {_best_npy.name} "
                          f"(score={_snap_score:.3f}) → 차량 흐름 검증 대기")
                elif _best_npy is None:
                    print("[스냅샷] 저장된 스냅샷 없음 → 새로 학습 후 저장")
            if not is_stream:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)                 # 파일이면 첫 프레임으로 되감기 (스트림은 이미 소비된 프레임 포기)

        # ── 방향별 TrafficAnalyzer 초기화 ─────────────────────────────
        self.traffic_analyzer_a = TrafficAnalyzer(                  # A방향 정체 탐지
            cfg, frame_w=fw, frame_h=fh, fps=fps,
            flow_map=self.flow,                                     # flow_map 주입 (flow_occupancy 셀 수 계산용)
        )
        self.traffic_analyzer_a.set_state(self.state)               # state 주입

        self.traffic_analyzer_b = TrafficAnalyzer(                  # B방향 정체 탐지
            cfg, frame_w=fw, frame_h=fh, fps=fps,
            flow_map=self.flow,                                     # flow_map 주입 (flow_occupancy 셀 수 계산용)
        )
        self.traffic_analyzer_b.set_state(self.state)               # state 주입

        # ── 방향별 CongestionPredictor 초기화 ─────────────────────────
        self.predictor_a = CongestionPredictor(cfg, fps=fps)        # A방향 정체 예측
        self.predictor_b = CongestionPredictor(cfg, fps=fps)        # B방향 정체 예측

        # ── 방향별 HistoricalPredictor 초기화 (시각별 jam 이력 기반 예측) ─
        _hist_dir = cfg.flow_map_path.parent if cfg.flow_map_path else None
        if _hist_dir is not None:
            self._hist_pred_a = HistoricalPredictor(
                csv_path=_hist_dir / "hist_jam_a.csv",
                smooth_threshold=cfg.smooth_jam_threshold,
                slow_threshold=cfg.slow_jam_threshold,
            )
            self._hist_pred_b = HistoricalPredictor(
                csv_path=_hist_dir / "hist_jam_b.csv",
                smooth_threshold=cfg.smooth_jam_threshold,
                slow_threshold=cfg.slow_jam_threshold,
            )
            print(f"📊 HistoricalPredictor ×2 초기화 완료 → {_hist_dir}")
        else:
            self._hist_pred_a = None                                # flow_map_path 없으면 비활성
            self._hist_pred_b = None
            print("ℹ️  HistoricalPredictor 비활성 (flow_map_path 없음)")

        # ── 정체 탐지 활성화 ──────────────────────────────────────────────
        self.traffic_analyzer_a.set_baseline()                     # A방향 FeatureExtractor(피처 계산) + CongestionJudge(정체 판정) 활성화
        self.traffic_analyzer_b.set_baseline()                     # B방향 FeatureExtractor(피처 계산) + CongestionJudge(정체 판정) 활성화
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
        _last_skip_frame: int = -9999                               # 마지막 프레임 스킵 감지 프레임 번호
        _post_skip_grace = getattr(cfg, "post_skip_grace_frames", 30)  # 재연결 후 차단 프레임 수

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
        _prev_cap_ts_ms     = None                                   # 직전 프레임의 스트림 타임스탬프 (ms)
        _is_time_gap        = False                                  # 이번 프레임이 타임스탬프 갭 직후인지

        # ── 카메라 전환 / 시작 스냅샷 차량 흐름 검증 ──────────────────────────
        # 스냅샷 로드 후 45프레임 동안 차량 속도 벡터와 flow_map 방향의
        # |cos| 평균으로 "같은 도로" 여부 확정.
        _sw_verifying         = False    # 검증 진행 중
        _sw_verify_npy        = None     # 검증 중 스냅샷 경로
        _sw_verify_cos_sum    = 0.0      # 누적 |cos| 합
        _sw_verify_vehicle_n  = 0        # 누적 유효 차량 수
        _sw_verify_frame_n    = 0        # 검증 경과 프레임 수
        _sw_verify_prev_label = None     # 전환 전 dir_label_a (검증 실패 시 복원용)
        _sw_verify_is_startup = False    # True: 시작 스냅샷 검증 (실패 시 is_learning 복귀)
        _SW_VERIFY_FRAMES     = 45       # 검증 기간 (약 1.5초 @ 30fps)
        _SW_VERIFY_COS_THR    = 0.40     # 같은 도로 판정 임계값 (avg |cos|)
        _SW_MATCH_MIN_SCORE   = 0.70     # 전환 후 시각 매칭 최소 점수
        _prev_frame_for_hint  = None     # 1프레임 전 버퍼 (optical flow 방향 추정용)

        # 시작 스냅샷이 로드됐으면 즉시 검증 모드로 진입
        # (hist_pred·traffic_analyzer 초기화 완료 후 이 시점에서 is_learning=False 설정)
        if _startup_snap_npy is not None:
            _sw_verifying         = True
            _sw_verify_npy        = _startup_snap_npy
            _sw_verify_cos_sum    = 0.0
            _sw_verify_vehicle_n  = 0
            _sw_verify_frame_n    = 0
            _sw_verify_prev_label = self._dir_label_a
            _sw_verify_is_startup = True
            st.is_learning        = False   # 탐지 모드로 전환 (검증 통과 시 확정)
            print(f"🔍 시작 스냅샷 차량 흐름 검증 시작 "
                  f"({_SW_VERIFY_FRAMES}프레임) → {_startup_snap_npy.name}")

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
            _hint_prev = _prev_frame_for_hint                       # 이번 이터레이션용 이전 프레임 보관
            _prev_frame_for_hint = frame                            # 다음 이터레이션을 위해 현재 프레임 저장

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

            # ── 타임스탬프 갭 감지 (partial drop 포착) ──────────────────────
            # freeze 감지: adj_diff ≈ 0 → 동결(반복) 프레임만 감지.
            # displacement: 다수 차량 동시 jump → 전체 끊김만 감지.
            # 이 감지: 스트림 타임스탬프 간격으로 "프레임이 빠진" 상황을 포착.
            # CAP_PROP_POS_MSEC: 파일이면 디코드 위치, 스트림이면 재생 시점.
            # 1~3프레임 partial drop 시: adj_diff는 정상(다른 장면이므로),
            # 차량 displacement는 정상 범위 내일 수 있지만 시간 갭이 발생.
            # 갭 감지 시: solo_jump 임계값을 시간 비율로 확대하여 정상 이동을
            # jump로 오판하는 것을 방지 + 속도 벡터에 갭 보정 적용.
            _cur_cap_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)             # 현재 프레임 타임스탬프 (ms)
            _time_gap_ratio = 1.0                                       # 기본: 갭 없음
            _is_time_gap = False
            if _prev_cap_ts_ms is not None and _cur_cap_ts_ms > 0:
                _ts_delta_ms = _cur_cap_ts_ms - _prev_cap_ts_ms        # 프레임 간 시간 차이 (ms)
                _expected_ms = 1000.0 / max(fps, 1.0)                  # 예상 프레임 간격 (ms)
                if _expected_ms > 0 and _ts_delta_ms > 0:
                    _time_gap_ratio = _ts_delta_ms / _expected_ms      # 실제/예상 비율
                    if _time_gap_ratio > 2.5 and _ts_delta_ms > 400:   # 2.5배 + 0.4초 이상
                        _is_time_gap = True
                        print(f"[F:{st.frame_num}] ⏱ 타임스탬프 갭 감지 "
                              f"(간격={_ts_delta_ms:.0f}ms, "
                              f"예상={_expected_ms:.0f}ms, "
                              f"비율={_time_gap_ratio:.1f}x)")
            _prev_cap_ts_ms = _cur_cap_ts_ms                           # 다음 프레임용 저장

            # ── YOLO 추적 ──
            tracks = self.tracker.track(frame)                      # [{id, x1, y1, x2, y2, cx, cy}, ...]
            active_ids = {t["id"] for t in tracks}                  # 현재 프레임에 보이는 ID들

            # ── 프레임 스킵(신호 끊김) 감지 ──────────────────────────────
            # HLS 스트림 끊김 시 모든 차량이 동시에 큰 변위를 가짐
            # → 절반 이상이 jump_px 초과 시 해당 프레임의 궤적·학습·판정 전부 스킵
            # 타임스탬프 갭 감지 시: jump 임계값을 시간 비율만큼 확대
            # → partial drop 동안의 정상 이동을 jump로 오판하지 않음
            _jump_thr   = _jump_thr_dynamic * min(_time_gap_ratio, 5.0) if _is_time_gap else _jump_thr_dynamic
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
                _last_skip_frame = st.frame_num                     # grace period 타이머 갱신
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

            # ── 타임스탬프 갭 기반 궤적 초기화 (displacement 미탐지 partial drop) ──
            # _is_frame_skip(displacement 기반)이 미탐지했으나 타임스탬프 갭이 감지된 경우:
            # 1~3프레임 partial drop에서 차량 displacement가 jump 임계값 미만이지만
            # 시간적으로 갭이 있어 velocity 벡터가 갭 구간을 포함하게 되는 경우.
            # 궤적을 초기화하여 갭 구간의 변위가 velocity 계산에 개입하는 것을 차단.
            # _is_frame_skip과 달리 wrong_way_ids는 유지 (이미 확정된 역주행은 취소 불필요).
            if _is_time_gap and not _is_frame_skip:
                _last_skip_frame = st.frame_num                     # grace period 타이머 갱신
                st.post_reconnect_frame = st.frame_num              # 재연결 이벤트 기록
                for _t in tracks:
                    _tid = _t["id"]
                    if st.trajectories[_tid]:
                        _cur_pos = (_t["cx"], _t["cy"])
                        st.trajectories[_tid] = [_cur_pos] * len(st.trajectories[_tid])
                    st.last_velocity.pop(_tid, None)
                    st.wrong_way_count[_tid] = 0
                    st.direction_change_frame[_tid] = st.frame_num

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
                # worst-of-both 방향의 jam_score 선택
                _src = (self.traffic_analyzer_a                     # jam 높은 쪽 analyzer
                        if _jam_a >= _jam_b else self.traffic_analyzer_b)
                _rule_jam = _src.get_rule_jam_score() if _src else 0.0  # rule_jam (로그용)
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
                    rule_jam_score=_rule_jam,                       # rule 기반 jam
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
                        st.waiting_stable_entered_frame = st.frame_num  # 최초 진입 시점 기록
                        self._track_direction.clear()

                # ── (B) 안정 대기 중: diff 모니터링 ────────────────────────
                elif st.waiting_stable:
                    # switch.check()을 호출해서 last_adj_diff를 업데이트
                    self.switch.check(frame, st.frame_num, st.cooldown_until)
                    _cur_diff = self.switch.last_adj_diff

                    _max_wait_frames = int(
                        getattr(cfg, "waiting_stable_max_sec", 30.0) * fps
                    )
                    _total_waited = st.frame_num - st.waiting_stable_entered_frame
                    _force_relearn = _total_waited >= _max_wait_frames

                    if _force_relearn:
                        # 최대 대기 시간 초과 — 불안정해도 강제 재학습
                        print(f"⏱️ 안정 대기 최대 시간 초과 ({_total_waited}프레임) → 강제 재학습 시작")

                    if _cur_diff > _stability_thr and not _force_relearn:
                        # 아직 불안정 → 안정 타이머 리셋 (단, 최대 대기 미초과 시에만)
                        st.stable_since_frame = st.frame_num
                        if st.frame_num % 30 == 0:
                            print(f"[대기] 아직 불안정 diff={_cur_diff:.1f} > {_stability_thr} "
                                  f"(총 대기 {_total_waited}/{_max_wait_frames}프레임)")
                    else:
                        # 안정 지속 중 or 강제 재학습 — 충분히 유지됐으면 재학습 시작
                        stable_frames = st.frame_num - st.stable_since_frame
                        if stable_frames >= _stability_required_frames or _force_relearn:
                            _prev_label_sw = self._dir_label_a   # 전환 전 라벨 보존

                            # ── 스냅샷 재매칭 시도 (강제 재학습 제외) ───────────────
                            # 시각 점수 0.70 이상 후보 → 45프레임 차량 흐름 검증 후 확정.
                            # 검증 통과(avg|cos|≥0.40) → 재학습 생략.
                            # 검증 실패 or 후보 없음 → 전체 재학습.
                            _sw_matched = False
                            if (not _force_relearn
                                    and _snap_dir is not None
                                    and _MATCHER_AVAILABLE
                                    and not _sw_verifying):
                                _sw_vgrid = self._make_vehicle_grid(
                                    tracks, fw, fh, self.cfg.grid_size
                                ) if tracks else None
                                _sw_cand, _sw_score = find_best_snapshot(
                                    frame, _snap_dir, min_score=_SW_MATCH_MIN_SCORE,
                                    prev_frame=_hint_prev,
                                    vehicle_grid=_sw_vgrid
                                )
                                if _sw_cand is not None and self.flow.load(_sw_cand):
                                    self.flow.speed_ref[:] = 0
                                    self.flow.apply_boundary_erosion()
                                    self.flow.apply_spatial_smoothing()
                                    self._compute_ref_direction()
                                    self._compute_direction_cell_counts()
                                    if self._ref_direction is not None:
                                        self.flow.build_directional_channels(
                                            *self._ref_direction
                                        )
                                    # 검증 상태 초기화
                                    _sw_verifying         = True
                                    _sw_verify_npy        = _sw_cand
                                    _sw_verify_cos_sum    = 0.0
                                    _sw_verify_vehicle_n  = 0
                                    _sw_verify_frame_n    = 0
                                    _sw_verify_prev_label = _prev_label_sw
                                    st.waiting_stable     = False
                                    _sw_matched           = True
                                    print(f"✅ 화면 안정 확인 ({stable_frames}프레임) "
                                          f"→ 스냅샷 후보 {_sw_cand.name} "
                                          f"(score={_sw_score:.3f}) → 차량 흐름 검증 시작")

                            if not _sw_matched:
                                # 후보 없음 or 강제 재학습 → 전체 재학습
                                if not _force_relearn:
                                    print(f"✅ 화면 안정 확인 ({stable_frames}프레임) → 재학습 시작")
                                st.waiting_stable        = False
                                self._prev_ref_direction = self._ref_direction
                                self._prev_dir_label_a   = _prev_label_sw
                                st.reset_for_relearn()         # 재학습 모드 진입
                                self.flow.reset()              # flow_map 초기화
                                self.traffic_analyzer_a.congestion_judge.reset()
                                self.traffic_analyzer_b.congestion_judge.reset()
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
                        st.waiting_stable_entered_frame = st.frame_num  # 최대 대기 타이머 재시작
                        self.flow.reset()                          # 오염된 flow_map 초기화
                        _relearn_smoothed_80 = False
                        _relearn_smoothed_95 = False

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
                        _last_skip_frame = st.frame_num             # grace period 타이머 갱신
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
                    _sdir = self._snapshot_dir()
                    if _sdir is not None:                           # 저장 경로 있으면
                        if _MATCHER_AVAILABLE:                      # 스냅샷으로 저장 (camera_id 서브폴더)
                            save_flow_snapshot(frame, self.flow, _sdir,
                                               dir_label_a=self._dir_label_a)
                        else:                                       # matcher 없으면 단일 파일 fallback
                            self.flow.save(cfg.flow_map_path)
                    st.is_learning = False                          # 학습 모드 종료
                    print(f"학습 완료! (frame={st.frame_num})")

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
                    # ── HistoricalPredictor 슬롯 스왑 비활성 ──────────────────
                    # 자동 스왑은 카메라 전환·새벽 저교통량·물리적 회전 등을
                    # 서로 구별할 수 없어 오히려 오작동을 유발한다.
                    # ref_direction을 가중평균으로 안정화했으므로 재학습 후에도
                    # 방향이 뒤집히는 일이 드물고, 설령 일시적으로 방향이 바뀌어도
                    # hist CSV는 누적 평균 구조라 새 데이터가 쌓이면 자가 교정된다.
                    self._prev_ref_direction = None                 # 사용 후 초기화
                    self._prev_dir_label_a   = None                 # 사용 후 초기화
                    _sdir = self._snapshot_dir()
                    if _sdir is not None:                           # 저장 경로 있으면
                        if _MATCHER_AVAILABLE:                      # 스냅샷으로 저장 (camera_id 서브폴더)
                            save_flow_snapshot(frame, self.flow, _sdir,
                                               dir_label_a=self._dir_label_a)
                        else:                                       # matcher 없으면 단일 파일 fallback
                            self.flow.save(cfg.flow_map_path)
                    st.relearning = False                           # 재학습 모드 종료
                    st.cooldown_until = st.frame_num + cfg.cooldown_frames  # 쿨다운 설정
                    # 역주행 판정 안정화 대기: 재학습 직후 _track_direction 미확정 차량이
                    # judge.check()에 진입하지 않도록 별도 유예 프레임 설정
                    # (cooldown_frames는 카메라 전환 감지 억제용 — 역주행 판정과 분리)
                    self._wrongway_stable_until = (
                        st.frame_num
                        + getattr(cfg, "wrongway_relearn_grace_frames",
                                  cfg.cooldown_frames)              # 기본값: cooldown_frames 재사용
                    )
                    self.switch.set_reference(frame)                # 새 기준 프레임 설정
                    print("재학습 완료! 쿨다운 시작")

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

            # ── 카메라 전환 스냅샷 검증: 차량 흐름 vs flow_map |cos| 누적 ────
            # _sw_verifying 구간 동안 각 차량의 속도 벡터와 flow_map 방향의
            # |cos| 를 누적. _SW_VERIFY_FRAMES 도달 시 평균으로 같은 도로 판정.
            if _sw_verifying:
                _fw_vw = cfg.velocity_window                        # 속도 계산 윈도우
                for _vt in tracks:
                    _vtraj = st.trajectories[_vt["id"]]
                    if not _vtraj or len(_vtraj) < _fw_vw:
                        continue                                    # 궤적 부족 → 건너뜀
                    _vsi  = len(_vtraj) - _fw_vw
                    _vpfx = [_vtraj[_vsi+i+1][0] - _vtraj[_vsi+i][0] for i in range(_fw_vw-1)]
                    _vpfy = [_vtraj[_vsi+i+1][1] - _vtraj[_vsi+i][1] for i in range(_fw_vw-1)]
                    _vvdx = float(np.median(_vpfx)) * (_fw_vw - 1)
                    _vvdy = float(np.median(_vpfy)) * (_fw_vw - 1)
                    _vmag = np.sqrt(_vvdx**2 + _vvdy**2)
                    _vbh  = max(_vt["y2"] - _vt["y1"], 1)
                    if _vmag / _vbh > cfg.norm_learn_threshold and _vmag > 1.0:
                        _vndx, _vndy = _vvdx / _vmag, _vvdy / _vmag
                        _vfv = self.flow.get_interpolated(_vt["cx"], _vt["cy"])
                        if _vfv is not None:
                            _sw_verify_cos_sum   += abs(_vndx * _vfv[0] + _vndy * _vfv[1])
                            _sw_verify_vehicle_n += 1
                _sw_verify_frame_n += 1

                if _sw_verify_frame_n >= _SW_VERIFY_FRAMES:
                    _sw_avg_cos = (_sw_verify_cos_sum / max(_sw_verify_vehicle_n, 1))
                    if _sw_avg_cos >= _SW_VERIFY_COS_THR:
                        # ── 검증 통과: 로드된 스냅샷 유지 → 재학습 생략 ──────
                        print(f"✅ 차량 흐름 검증 통과 "
                              f"(avg|cos|={_sw_avg_cos:.3f}, "
                              f"n={_sw_verify_vehicle_n}) → 스냅샷 재사용")
                        st.is_learning = False
                    else:
                        # ── 검증 실패: 다른 도로 ─────────────────────────────
                        print(f"⚠️ 차량 흐름 검증 실패 "
                              f"(avg|cos|={_sw_avg_cos:.3f} < {_SW_VERIFY_COS_THR}) → 재학습")
                        self.flow.reset()
                        self.traffic_analyzer_a.congestion_judge.reset()
                        self.traffic_analyzer_b.congestion_judge.reset()
                        self._ref_direction = None
                        if _sw_verify_is_startup:
                            # 시작 스냅샷 실패 → 신규 초기 학습으로 복귀
                            st.is_learning   = True
                            _learn_smoothed_80 = False
                            _learn_smoothed_95 = False
                        else:
                            # 카메라 전환 스냅샷 실패 → 재학습 모드
                            self._prev_ref_direction = self._ref_direction
                            self._prev_dir_label_a   = _sw_verify_prev_label
                            st.reset_for_relearn()
                            _relearn_smoothed_80 = False
                            _relearn_smoothed_95 = False
                    _sw_verifying         = False                   # 검증 종료
                    _sw_verify_is_startup = False

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
                #       원거리 차량은 YOLO가 자주 끊겨 traj<10인 경우 많음 →
                #       velocity_window(10) 대기 없이 조기 정확 분류
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
                # 타임스탬프 갭 감지 시: _jump_thr이 이미 시간 비율로 확대되어 있으므로
                # 갭 동안의 정상 이동은 jump로 오판되지 않음.
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
                    # ── 중앙값 속도 벡터 (117차) + IQR 이상치 필터 (124차) ──────
                    # endpoint-to-endpoint 대신 프레임별 변위의 중앙값 사용.
                    # 단일 프레임 끊김·순간이동(신호 지연 등)이 있어도 중앙값에는
                    # 영향 없음 → fleet_cos/solo_jump 의존 없이 방향 벡터가 견고해짐.
                    # [124차] IQR 이상치 필터 추가: 프레임 드롭 후 궤적에 남은
                    # 이상 변위(갭 구간 포함)를 제거한 후 중앙값 계산.
                    # 중앙값만으로도 50% 미만 이상치는 필터되지만, 2~3프레임 연속
                    # 드롭 시 velocity_window(10f) 내 이상치 비율 20~30%가 돼
                    # 중앙값이 이상치 쪽으로 편향될 수 있음. IQR은 이를 방어.
                    _w   = cfg.velocity_window
                    _si  = len(traj) - _w
                    _pfx = [traj[_si+i+1][0] - traj[_si+i][0] for i in range(_w-1)]
                    _pfy = [traj[_si+i+1][1] - traj[_si+i][1] for i in range(_w-1)]

                    # IQR 이상치 필터: per-frame 변위 크기의 Q1~Q3 범위 밖 제거
                    _pf_mags = [(_pfx[i]**2 + _pfy[i]**2)**0.5 for i in range(len(_pfx))]
                    if len(_pf_mags) >= 5:                           # 충분한 샘플 시에만
                        _q1 = float(np.percentile(_pf_mags, 25))
                        _q3 = float(np.percentile(_pf_mags, 75))
                        _iqr = _q3 - _q1
                        _upper = _q3 + 2.0 * _iqr                   # 상한 (2×IQR — 보수적)
                        if _upper > 0:                               # 유효한 상한이 있으면
                            _keep = [i for i in range(len(_pfx)) if _pf_mags[i] <= _upper]
                            if len(_keep) >= 3:                      # 필터 후 최소 3개 남아야
                                _pfx = [_pfx[i] for i in _keep]
                                _pfy = [_pfy[i] for i in _keep]

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
                            # ── 재학습 후 유예 기간 또는 track_dir 미확정 → 역주행 판정 차단 ──
                            # ① 재학습 완료 직후: _track_direction이 재확정되기 전
                            #    플로우맵이 막 완성된 상태에서 old 궤적 기반 오탐 방지
                            # ② track_dir=None: _ref_direction=None 기간 중 방향 분류 스킵된 차량
                            #    → flow 채널 특정 불가 + 이웃 가드(_sus_dir=None) 무력화
                            _wrongway_blocked = (
                                st.frame_num <= self._wrongway_stable_until
                                or self._track_direction.get(tid) is None
                            )
                            # judge.check() 전에 이미 확정된 차량인지 기록
                            # → 이웃 가드는 이번 프레임에 새로 확정된 차량에만 적용
                            _was_confirmed_before = (tid in st.wrong_way_ids)
                            if _wrongway_blocked:
                                debug_info = {"status": "dir_unclassified", "cos_values": []}
                            else:
                                # 역주행 여부 판단 (bbox_h 전달 — nm 기반 속도 게이트용)
                                _bbox_h = max(y2 - y1, 1)           # bbox 높이 (원근 정규화용)
                                is_wrong, _, debug_info = self.judge.check(
                                    tid, traj, ndx, ndy, mag, cy, _bbox_h,
                                    track_dir=self._track_direction.get(tid)
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

            # ── 방향별 TrafficAnalyzer 갱신 ─────────────────────────────
            _in_grace = (st.frame_num - _last_skip_frame) <= _post_skip_grace
            if (self.traffic_analyzer_a is not None                 # 초기화 완료 확인
                    and not st.is_learning and not st.relearning and not st.waiting_stable  # 탐지 모드일 때만
                    and not _is_frame_skip                          # 프레임 스킵 프레임 차단
                    and not _in_grace):                             # 재연결 후 grace period 차단 (속도 이력 재구성 대기)
                # A방향 정체 탐지 갱신
                self.traffic_analyzer_a.update(tracks_a, speeds_a, st.frame_num)
                self.predictor_a.update(self.traffic_analyzer_a.get_avg_speed())
                # B방향 정체 탐지 갱신
                self.traffic_analyzer_b.update(tracks_b, speeds_b, st.frame_num)
                self.predictor_b.update(self.traffic_analyzer_b.get_avg_speed())

                # ── HistoricalPredictor: 현재 jam_score를 5분 창에 누적 ──
                # 슬롯 경계(5분) 도달 시 중앙값 계산 후 CSV에 자동 flush
                # 검증 구간(_sw_verifying)에는 기록 금지 — 잘못된 도로 스냅샷 오염 방지
                if self._hist_pred_a is not None and not _sw_verifying:
                    _now_dt = datetime.now()
                    self._hist_pred_a.record(
                        self.traffic_analyzer_a.get_jam_score(), dt=_now_dt
                    )
                    self._hist_pred_b.record(
                        self.traffic_analyzer_b.get_jam_score(), dt=_now_dt
                    )

                # ── flow_map speed_ref 온라인 학습 (SMOOTH 구간만) ────────
                # SMOOTH 구간의 nm을 셀별로 EMA 축적 → 위치별 정상속도 기준 확보
                # 이후 feature_extractor에서 velocity_deficit = 1 - nm/speed_ref 계산에 사용
                # 검증 구간(_sw_verifying)에는 학습 금지 — 잘못된 도로 속도 기준 오염 방지
                if not _sw_verifying:
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

            # 스냅샷 차량 흐름 검증 중이면 화면 상단에 검증 텍스트 표시
            elif _sw_verifying:
                _sv_remain = max(0, _SW_VERIFY_FRAMES - _sw_verify_frame_n)
                _sv_avg = (_sw_verify_cos_sum / max(_sw_verify_vehicle_n, 1))
                cv2.putText(frame,
                            f"VERIFYING SNAPSHOT: {_sv_remain}f remain "
                            f"| avg|cos|={_sv_avg:.2f}",
                            (fw // 2 - 280, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

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
                    # ── 5분 후 예측 패널 — HistoricalPredictor ──────────────
                    # 데이터 없으면 predict()=None → "Training..." 표시
                    if self._hist_pred_a is not None:
                        _pred_now = datetime.now()
                        pred_a = self._hist_pred_a.predict(_pred_now)
                        pred_b = self._hist_pred_b.predict(_pred_now)
                    else:
                        pred_a = pred_b = None
                    self.vis.draw_prediction_panel(
                        frame, pred_a, pred_b,
                        label_a=self._dir_label_a
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

        # ── HistoricalPredictor: 마지막 미완성 5분 창 flush + 통계 출력 ─
        if self._hist_pred_a is not None:
            self._hist_pred_a.flush_current()
            self._hist_pred_b.flush_current()
            print(f"📊 HistoricalPredictor 저장 완료"
                  f" (A: {self._hist_pred_a.get_total_windows()}창"
                  f" / B: {self._hist_pred_b.get_total_windows()}창)")

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
