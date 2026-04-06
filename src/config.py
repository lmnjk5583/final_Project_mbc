# 모든 고정 파라미터를 한 곳에서 관리하는 설정 클래스

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectorConfig:
    # ==================== 모델/기본 설정 ====================
    model_path: str | Path = ""       # YOLO 모델 경로
    conf: float = 0.3                 # 객체 검출 신뢰도(confidence) 임계값
    target_classes: list | None = None  # 추적할 클래스 인덱스 리스트 (None이면 모든 클래스)

    # ==================== 흐름 그리드(Flow Map) 설정 ====================
    grid_size: int = 20               # 흐름 맵을 나눌 격자 크기 (N x N) — 20은 셀 수 과다→셀당 샘플 부족→노이즈

    # ==================== 학습(learning) 관련 설정 ====================
    learning_frames: int = 1800        # 초기 학습에 사용할 프레임 수 — 원래 500, 충분한 셀 커버리지 확보
    alpha: float = 0.1                # EMA 학습 속도 (새 데이터 반영 비율 10%)
    min_samples: int = 5              # 셀당 최소 학습 샘플 수 (이하이면 공간 보정에 사용)
    enable_online_flow_update: bool = False # 정상 흐름 학습에 사용(True)

    # ==================== 역주행 탐지 관련 설정 ====================
    velocity_window: int = 20         # 속도/방향 계산 시 사용하는 프레임 간격 (이전 위치~현재 위치 거리) — 15→20 (bbox jitter의 방향 벡터 영향 완화)
    base_speed_threshold: float = 7.0 # 기본 속도 임계값 (원근에 따라 가중을 곱해 사용)
    cos_threshold: float = -0.75      # 코사인 유사도 임계값 (원본값 복원 — smoothing 오염 방지로 오탐 차단)
    wrong_count_threshold: int = 12   # 역주행 확정까지 필요한 연속 의심 횟수 — 8→12 (오탐 감소)
    vote_threshold: float = 0.7       # 투표 시 역방향 비율 임계값 (원본값 복원 — 60% 이상이면 역주행 의심)
    min_move_distance: float = 10.0    # 최소 누적 이동 거리 (이하면 정지로 판단) — 원래 20.0, 원거리 CCTV 대응 완화
    min_move_per_frame: float = 0.4   # 프레임당 평균 이동거리 (이하면 정지) — 원래 1.5, 원거리 CCTV 대응 완화
    direction_change_guard_frames: int = 120 # 방향 급변 이후 가드 기간 (프레임 수) — 급변 감지 시점부터 이 프레임 동안 의심 카운트 차단
    direction_change_cos_threshold: float = 0.0  # 급변 감지 임계값 (cos 기준, 0.0=90°+, 0.5=60°+) — stable 방향 대비 이 이상 벗어나면 급변으로 기록

    # ==================== ID 매핑 관련 ====================
    id_match_distance: int = 120      # ID 재매칭 허용 거리 (픽셀 단위, 이전 ID와 새 ID 위치 비교)
    trail_length: int = 30            # 차량 궤적 최대 길이 (리스트에 최근 몇 점까지 보관할지)
    stale_threshold: int = 90         # 안 보이는 ID를 삭제하기까지의 프레임 수
    reappear_frame_limit: int = 45    # ID 재매칭 시 사라진 지 최대 몇 프레임까지 허용
    last_pos_expire: int = 60         # 역주행 마지막 위치 기록 만료 프레임

    # ==================== 카메라 전환 감지 관련 ====================
    relearn_frames: int = 300         # 재학습에 사용할 프레임 수
    cooldown_frames: int = 150        # 재학습 후 전환 감지 비활성 프레임 수
    switch_confirm_needed: int = 4    # 전환으로 확정하기 위해 필요한 연속 감지 횟수

    # ==================== 경로 관련 ====================
    flow_map_path: Path = None        # flow_map 저장/로드 파일 경로
    result_dir: Path = None           # 결과 영상 저장 폴더 경로
    data_dir: Path = None             # 입력 데이터(영상) 폴더 경로
    
    # ==================== 로깅/실행 모드 ====================
    detect_only: bool = True            # True면 flow_map 필수(학습 안 함)
    log_dir: Path | None = None         # 로그 저장 폴더
    log_interval_frames: int = 5        # 트랙/프레임 로그를 N프레임마다 기록

    # ==================== 정체 탐지 파라미터 (OLD — CongestionPredictor 호환용) ====================
    free_flow_speed: float = 100.0        # 자유 흐름 속도 기준 (km/h) — 고속도로 기본값
    pixels_per_meter: float = 8.0         # 1미터 = 몇 픽셀 (카메라·해상도 따라 보정 필요)
    congestion_hysteresis_sec: float = 3.0   # 정체 레벨 전환 유지 시간 (초) — 5→3으로 단축 (서행 반응속도 향상)
    prediction_history_window: int = 30   # CongestionPredictor 속도 히스토리 창 (프레임 수)
    prediction_horizon: int = 5           # 정체 예측 시간 범위 (분)

    # ==================== Phase 1 정체 탐지 파라미터 ====================
    min_active_for_baseline: int   = 2      # 온라인 학습 baseline 갱신에 필요한 최소 활성 차량 수
    min_passage_dist:        float = 100.0  # 유효 passage 최소 진입-퇴장 픽셀 거리
    min_passages_required:   int   = 5      # 학습 종료에 필요한 최소 완성 passage 수
    stop_mag_threshold:      float = 3.0    # (구버전 호환용, 미사용) 절대 픽셀 정지 임계값
    norm_stop_threshold:     float = 0.06   # bbox_h 대비 정지 임계값 (mag/bbox_h < 이 값 → 정지)
                                            # 0.10→0.06: 원거리 차량 bbox_h 클램프(30px) 시 nm≈0.07~0.10 오판 방지
                                            # 완전 정지는 nm≈0~0.03, 서행은 nm≈0.07+로 충분히 구분 가능
    norm_learn_threshold:      float = 0.10 # 플로우맵 학습 진입 nm 임계값 (nm_move < 이 값 → 기록 스킵)
                                            # norm_speed_gate_threshold(0.15)보다 낮게 — 서행 차량 방향도 학습 허용
                                            # nm=0.10: bbox_h=30 기준 mag≥3px (방향 신뢰 최솟값)
                                            # 0.05는 1.5px 변위 허용 → 방향 벡터가 랜덤 노이즈 수준 → 오염 유발
    norm_speed_gate_threshold: float = 0.15 # 역주행 판정 진입 nm 임계값 (nm_speed < 이 값 → 방향 불명확, 판정 스킵)
                                            # nm = mag / max(bbox_h, min_bbox_h) — 원근 정규화 속도
                                            # cy 기반 raw 속도 임계값(1~2 단위 변화)을 대체:
                                            # 근거리(bbox_h=150) nm=0.15 → mag=22px 필요
                                            # 원거리(bbox_h=30)  nm=0.15 → mag=4.5px 필요 (비례 보정)
    min_bbox_h:              float = 30.0   # bbox_h 최솟값 보정 (이 값 미만이면 30px로 클램프)
                                            # 원거리 차량 bbox_h≈15px → nm이 과대 계산되어 원활 오판 방지
    exit_rate_window:        int   = 30     # exit_rate 계산 슬라이딩 윈도우 (프레임)
    grace_period_sec:        float = 60.0   # 카메라 전환 후 판정 유예 시간 (초)

    # ==================== jam_score 임계값 ====================
    smooth_jam_threshold:    float = 0.30   # jam_score 이 값 미만 → SMOOTH — 0.25→0.30 (원활 차량 dwell/density 기여 흡수)
    slow_jam_threshold:      float = 0.60   # jam_score 이 값 미만 → SLOW, 이상 → CONGESTED
    density_max_vehicles:   float = 40.0   # density 정규화 기준 차량 수 — 이 값 이상이면 density=1.0 (포화) — 20→40 (10대 원활 시 density 50%→25% 보정)
    default_lcs:             float = 0.36   # 한강대교 베이스라인 학습 결과 — 모든 카메라 임계값 보정에 사용

    # ==================== jam_score EMA 스무딩 ====================
    # 비대칭 EMA: 악화(올라갈 때)는 빠르게, 호전(내려갈 때)은 느리게
    # 실제 교통 특성 반영 — 정체는 순식간에 쌓이지만 해소는 수분 이상 걸림
    jam_ema_alpha_up:   float = 0.10  # 악화 방향 EMA 속도 (새 값 10% 반영) — 약 10프레임에 걸쳐 반응 (0.15→0.10 급등 완화)
    jam_ema_alpha_down: float = 0.04  # 호전 방향 EMA 속도 (새 값 4% 반영)  — 약 25프레임에 걸쳐 반응

    # ==================== 학습 연장 ====================
    max_learning_extension:  float = 1.5    # learning_frames × 이 값 = 최대 학습 프레임 수

    # ==================== Phase 2 GRU 파라미터 ====================
    gru_hidden: int = 64                    # GRU hidden state 크기
    gru_layers: int = 2                     # GRU 레이어 수
    gru_seq_len: int = 30                   # 입력 시퀀스 길이 (프레임)
    gru_blend_ratio: float = 0.40           # GRU 기여 비율 (1 - 이 값 = rule 비율)
    gru_warmup_frames: int = 30             # camera_switch 후 GRU 사용 금지 프레임
    gru_replay_size: int = 200              # replay_buffer 최대 크기
    gru_online_interval: int = 10           # 온라인 학습 gradient step 주기 (프레임)
    gru_lr: float = 1e-3                    # Adam optimizer 학습률
    gru_forecast_steps: int = 150           # 미래 예측 자기회귀 스텝 수 (150프레임 ≈ 5초@30fps)

    # ==================== 화면 표시 설정 ====================
    display_width: int = 1280              # 화면 출력 창 너비 (픽셀). 0이면 원본 해상도 그대로
    display_height: int = 720              # 화면 출력 창 높이 (픽셀). 0이면 원본 해상도 그대로

    # ==================== 정체 탐지 slow_ratio 파라미터 ====================
    slow_upper_nm: float = 0.50            # 서행 판정 상한 nm (이 값 미만 = 서행, 이상 = 정상 주행)
                                           # norm_speed_gate_threshold(0.15)와 분리 — 역주행 게이트와 무관
                                           # nm≈0.50: 속도 40~50km/h 수준 (카메라 화각에 따라 다름)
                                           # 원활(80km/h nm≈0.90): slow 미해당, 서행(40km/h nm≈0.44): slow 해당

    # ==================== 방향별 차선 분리 파라미터 ====================
    lane_cos_threshold: float = 0.0        # 방향 분류 코사인 임계값 (≥ 이면 A방향, < 이면 B방향)