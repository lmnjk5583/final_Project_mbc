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
    # ==================== 역주행 탐지 관련 설정 ====================
    velocity_window: int = 10         # 속도/방향 계산 프레임 간격 — 원거리 상행 차량 ID 수명 짧아 10으로 설정
    base_speed_threshold: float = 7.0 # 기본 속도 임계값 (원근에 따라 가중을 곱해 사용)
    cos_threshold: float = -0.75      # 코사인 유사도 임계값 (원본값 복원 — smoothing 오염 방지로 오탐 차단)
    wrong_count_threshold: int = 20   # 역주행 확정까지 필요한 연속 의심 횟수 — 25→20 (fast-track이 명확한 경우 처리, normal path는 오탐 방지용 보수적 유지)
    min_wrongway_track_age: int = 20  # 역주행 판정 시작까지 최소 추적 프레임 수 — 45→20 (빠른 역주행 차량 age gate 병목 해소)
    vote_threshold: float = 0.7       # 투표 시 역방향 비율 임계값 (원본값 복원 — 60% 이상이면 역주행 의심)
    min_move_distance: float = 10.0    # 최소 누적 이동 거리 (이하면 정지로 판단) — 원래 20.0, 원거리 CCTV 대응 완화
    min_move_per_frame: float = 0.4   # 프레임당 평균 이동거리 (이하면 정지) — 원래 1.5, 원거리 CCTV 대응 완화
    direction_change_guard_frames: int = 120 # 방향 급변 이후 가드 기간 (프레임 수) — 급변 감지 시점부터 이 프레임 동안 의심 카운트 차단
    direction_change_cos_threshold: float = 0.3  # 급변 감지 임계값 (cos 기준) — 0.0(90°)→0.3(72°): bbox jitter 45~60° 편차도 가드 트리거

    # ── 고신뢰 즉시 확정 (fast-track) ─────────────────────────────────
    # 투표 결과가 압도적이고 속도가 충분하면 wrong_count 누적 없이 바로 확정
    # 조건: disagree_ratio >= fast_confirm_ratio AND nm_speed >= fast_confirm_speed
    #       AND 처음부터 역방향(lcf==0) — 갑자기 방향 바뀐 차량은 일반 경로 유지
    fast_confirm_ratio: float = 0.95  # 단기 투표 역방향 비율 임계 (8포인트 중 ~모두 역방향)
    fast_confirm_speed: float = 0.20  # nm_speed 임계 (중속 이상 역주행 즉시 확정 — 0.40→0.20: 중속 역주행 차량도 fast-track 적용)
    fast_confirm_min_age: int = 45    # fast-track 발동 최소 트랙 나이 (프레임)
                                      # 새 CCTV 뷰·오염 flow map 셀에서 정상 차량이 lcf=0으로
                                      # 보이는 오탐 방지. 45f = 6fps 기준 약 7.5초 관찰 후 확정.
                                      # 진짜 역주행: 45f 후에도 consistently wrong → 여전히 확정
                                      # 정상 차량: 45f 동안 최소 1회 정상 판정 → lcf 갱신 → fast-track 차단
    post_slow_guard_frames: int = 30  # 서행→가속 직후 fast-track 오탐 방지 최소 의심 유지 프레임

    # ── 이웃 차량 방향 일치 가드 (neighbor_agreement_guard) ──────────────
    # 역주행 확정 직전, 같은 방향 분류(A/B)의 이웃 차량들이 같은 방향으로
    # 이동 중이면 flow map 오탐으로 판단해 확정 취소.
    # 진짜 역주행: 이 차량만 반대방향 → 이웃 동방향 차량 없음 → 가드 비발동
    # flow map 오류 오탐: 여러 정상 차량이 일괄 flagging → 이웃도 같은 방향 → 취소
    neighbor_guard_min_total: int = 2   # 이웃 가드 발동 최소 같은분류 차량 수 (미만이면 비적용) — 3→2: W2+W3 2대만 있어도 발동
    neighbor_guard_agree: int = 1        # 이 수 이상의 이웃이 같은 방향이면 오탐으로 취소 — 2→1: 1대만 동방향이어도 취소

    # ── 구역 확정 쿨다운 (120차) ──────────────────────────────────────────
    # 같은 grid cell에서 역주행이 확정된 후 이 프레임 수 이내에
    # 동일 구역에서 새 확정이 발생하면 flow map 에코 오탐으로 간주해 취소.
    # W1 확정(1차 경보) → W2/W3 같은 구역 통과 → 에코 차단
    # 진짜 역주행: 쿨다운 종료 후엔 재발동 가능 / 다른 구역은 즉시 감지
    wrong_zone_cooldown_frames: int = 900  # 30fps 기준 30초 — 동일 구역 연속 에코 차단 기간
                                      # nm < norm_speed_gate_threshold 서행 구간이 이 값 이상 지속된 후
                                      # 의심 시작 시, fast-track 발동 전 최소 이 만큼 의심 지속 요구.
                                      # ID:1327 패턴: ~150f 서행(nm<0.15) → 가속 순간 lcf=0 → 즉시 fast-track
                                      # → 이 가드가 30f 의심 유지 후에만 fast-track 허용

    # ── bbox 겹침 기반 경계 침식 ───────────────────────────────────────
    # 학습 중 반대 차선 차량의 bbox가 이 셀을 N회 이상 밟았으면 중앙선 경계로 판정 → 제거
    bbox_contra_threshold: int = 8      # 반대방향 bbox 방문 횟수 임계값 (이상이면 경계 셀로 삭제) — 3→8: bbox 풋프린트 확대 후 중앙선 인접 셀 과잉 침식 방지

    # ── bbox 거리 기반 alpha 감쇠 ────────────────────────────────────────
    # bbox 중심에서 멀어질수록 alpha를 감쇠 → 중앙점 우선 학습.
    # edge 셀(방향 게이팅 임계 미만)은 잠기지 않아 반대 차선이 덮어쓸 수 있음.
    # dist=0: alpha×1.0 / dist=1: alpha×decay / dist=2: alpha×decay² / ...
    bbox_alpha_decay: float = 0.5       # 거리 1셀당 alpha 감쇠율 (0.5 → 2셀 거리에서 alpha/4)
    bbox_gating_alpha_ratio: float = 0.3  # 이 비율(decay^dist) 미만이면 방향 게이팅·count 증가 비적용
    max_cross_flow_cells: float = 1.2    # 횡방향(차선 횡단) 최대 확산 셀 수 (이동 방향에 수직)
                                         # 1.2 = 수직 방향 1셀 + 약간의 여유 (대각선 허용)
                                         # 이동 방향과 평행(전후방)은 무제한 → 차선 내 커버리지 유지
                                         # 중앙선 annotation 없이 반대 차선 오염 구조 차단

    # ── 프레임 freeze 감지 (끊김 재연결 감지) ────────────────────────────
    # adj_diff(인접 프레임 차이)가 rolling_avg × 10% 이하로 min_freeze_frames 이상 지속되면
    # 카메라 freeze로 간주. 이후 adj_diff 복귀 시 재연결 이벤트 → fast-track 차단.
    # ★ 임계값: 상대값(avg × 10%) — 정체 구간 저속 diff 오감지 방지
    #   정상(avg≈5.0): freeze 임계 0.50 / 정체(avg≈0.5): freeze 임계 0.05
    #   진짜 freeze: adj≈0.00 → 항상 감지됨
    min_freeze_frames: int = 10         # 이 이상 정지 프레임이 연속되면 freeze로 확정 (6fps=1.7초)

    # ── FlowMap 가장자리 학습 제외 마진 ──────────────────────────────────
    # 그리드 외곽 N줄은 학습하지 않음.
    # 이유: 차량이 막 진입/퇴장하는 가장자리는 궤적이 짧아 traj 방향 미확립 상태.
    #       이 상태에서 반대 차선 bbox가 가장자리 셀을 오염시키면 수정 불가.
    # grid_size=20 기준 1셀 마진 → 외곽 76셀 미학습 (약 19%)
    flow_map_edge_margin: int = 1       # 학습 제외 가장자리 셀 수 (그리드 외곽 N줄)

    # ── bbox 수평 폭 제한 (중앙선 침범 방지) ─────────────────────────────
    # 학습 시 bbox의 수평(x) 폭을 bbox_height × ratio 로 제한.
    # 근거: 고속도로 CCTV에서 차선 폭 ≈ 차량 높이(원근 기준).
    #       ratio=0.8 → max 반폭 = 0.8×bbox_h → 차선 내에서만 학습.
    #       실제 bbox가 더 좁으면 클리핑 없음 (min 적용).
    # 효과:  넓은 트럭/버스 bbox나 카메라 각도로 인해 bbox가 반대 차선까지 뻗어도
    #        학습 대상에서 제외 → 중앙선 침범 방지.
    # 공백 우려: dist=0·1은 클리핑 후에도 중심 ±1셀 커버 → 학습 충분.
    bbox_learn_w_ratio: float = 0.8     # 학습 bbox 반폭 제한 = bbox_height × 이 값

    # ==================== ID 매핑 관련 ====================
    id_match_distance: int = 120      # ID 재매칭 허용 거리 (픽셀 단위, 이전 ID와 새 ID 위치 비교)
    trail_length: int = 30            # 차량 궤적 최대 길이 (리스트에 최근 몇 점까지 보관할지)
    stale_threshold: int = 90         # 안 보이는 ID를 삭제하기까지의 프레임 수
    reappear_frame_limit: int = 45    # ID 재매칭 시 사라진 지 최대 몇 프레임까지 허용
    last_pos_expire: int = 60         # 역주행 마지막 위치 기록 만료 프레임

    # ==================== 카메라 전환 감지 관련 ====================
    relearn_frames: int = 600         # 재학습에 사용할 프레임 수
    cooldown_frames: int = 150        # 재학습 후 전환 감지 비활성 프레임 수
    switch_confirm_needed: int = 4    # 전환으로 확정하기 위해 필요한 연속 감지 횟수

    # ==================== 안정 대기 (전환 후 재학습 시작 조건) ====================
    stability_required_sec:  float = 4.0   # 전환 감지 후 이 시간 동안 화면이 안정돼야 재학습 시작
                                           # 4초 = 카메라 팬/줌 완료 후 고정 확인용 최소 시간
    stability_diff_threshold: float = 8.0  # 이 이하면 안정 판정 (인접 프레임 grayscale diff 기준)
                                           # 8.0: 조명 변화/차량 움직임 허용, 카메라 이동 차단
    relearn_abort_diff:       float = 15.0 # 재학습 중 이 이상이면 불안정 → 재학습 중단 후 대기 복귀
                                           # stability_diff_threshold보다 높게 — 일시적 차량 오판 방지

    # ==================== 경로 관련 ====================
    flow_map_path: Path = None        # flow_map 저장/로드 파일 경로
    result_dir: Path = None           # 결과 영상 저장 폴더 경로
    data_dir: Path = None             # 입력 데이터(영상) 폴더 경로
    
    # ==================== 로깅/실행 모드 ====================
    detect_only: bool = True            # True면 flow_map 필수(학습 안 함)
    log_dir: Path | None = None         # 로그 저장 폴더
    log_interval_frames: int = 5        # 트랙/프레임 로그를 N프레임마다 기록

    # ==================== 정체 탐지 히스테리시스 ====================
    congestion_hysteresis_sec: float = 3.0   # 정체 레벨 전환 유지 시간 (초) — 3.0→7.0: 순간 변동으로 인한 레벨 오락가락 방지

    # ==================== 초기 확정 구간 (학습 완료 직후) ====================
    initial_confirm_sec:     float = 5.0     # 학습 완료 후 이 시간 동안 초기 히스테리시스 적용
                                             # cell_dwell_ema·cell_persistence 안정화에 최소 3초 필요 → 5초로 여유
    initial_hysteresis_sec:  float = 2.0     # 초기 확정 구간 중 사용할 짧은 히스테리시스
                                             # 정규(7초)보다 짧게 — 학습 직후 현재 도로 상태를 빠르게 확정

    # ==================== CongestionPredictor 파라미터 ====================
    free_flow_speed: float = 100.0           # 자유 흐름 속도 기준 (km/h) — 회복 예측용
    prediction_history_window: int = 30      # 속도 히스토리 창 (프레임 수)

    # ==================== Phase 1 정체 탐지 파라미터 ====================
    count_ref: float = 8.0                  # 방향당 기준 차량 수 (count_ratio 계산용)
    stop_mag_threshold:      float = 3.0    # 절대 픽셀 정지 임계값 (affected_vehicles 판정용)
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
    smooth_jam_threshold:    float = 0.25   # jam_score 이 값 미만 → SMOOTH
    slow_jam_threshold:      float = 0.60   # jam_score 이 값 미만 → SLOW, 이상 → JAM
    density_max_vehicles:   float = 40.0   # density 정규화 기준 차량 수 — 이 값 이상이면 density=1.0 (포화)

    # ==================== jam_score EMA 스무딩 ====================
    # 비대칭 EMA: 악화(올라갈 때)는 빠르게, 호전(내려갈 때)은 느리게
    # 실제 교통 특성 반영 — 정체는 순식간에 쌓이지만 해소는 수분 이상 걸림
    jam_ema_alpha_up:   float = 0.40  # 악화 방향 EMA 속도 — 빠른 상승 유지하면서 순간 노이즈 흡수
                                      # 0.70은 EMA가 raw값과 거의 동일해져 비대칭 설계 의미 없어짐
                                      # 0.40: 1프레임에 40% 반영, 3프레임이면 raw의 ~78% 수렴
    jam_ema_alpha_down: float = 0.04  # 호전 방향 EMA 속도 (새 값 4% 반영)  — 약 25프레임에 걸쳐 반응

    # ==================== 학습 연장 ====================
    max_learning_extension:  float = 1.5    # learning_frames × 이 값 = 최대 학습 프레임 수

    # ==================== 화면 표시 설정 ====================
    display_width: int = 1280              # 화면 출력 창 너비 (픽셀). 0이면 원본 해상도 그대로
    display_height: int = 720              # 화면 출력 창 높이 (픽셀). 0이면 원본 해상도 그대로
    night_enhance: bool = True             # CLAHE 야간 저조도 보정 (평균 밝기 < 80 시 자동 적용)

    # ==================== 프레임 스킵(신호 끊김) 감지 ====================
    frame_skip_jump_px:    float = 80.0  # 차량 1대 순간이동 판정 픽셀 (1프레임 최대 이동)
                                         # 6fps 기준 정상 고속차 120km/h≈30px/f → 80px 초과는 이상
    frame_skip_ratio:      float = 0.5   # 전체 추적 차량 중 이 비율 이상이 jump면 스킵 프레임 확정
                                         # 0.5 = 절반 이상 동시 jump → 신호 끊김으로 판단
    post_skip_grace_frames: int  = 30    # 프레임 스킵 감지 후 TA 업데이트 추가 차단 프레임
                                         # 재연결 직후 속도 이력 미구성 차량이 jam_score를 낮추는 것을 방지
                                         # 6fps 기준 30프레임 = 5초 (IDManager 속도 이력 재구성 시간)

    # ==================== 정체 탐지 slow_ratio 파라미터 ====================
    slow_upper_nm: float = 0.70            # 서행 판정 상한 nm (nm < 이 값 → 서행) — 0.50→0.70: EMA smoothing 도입 후 원거리 정상차량 nm 0.5~0.7 범위 오판 방지
    norm_speed_ref_override: float = 0.0  # norm_speed_ref 고정값 (0이면 자기보정 baseline 사용)
    nm_cy_correction_k: float = 0.0        # nm cy 보정 계수 — 0=비활성 (bbox_h 정규화로 충분, cy 이중보정 시 상행/하행 nm 비대칭 오탐)

    # ==================== 방향별 자기보정 nm baseline ====================
    nm_baseline_ema_up:   float = 0.05    # baseline 상승 EMA (원활 복귀 시 빠르게 반응)
    nm_baseline_ema_down: float = 0.005   # baseline 하락 EMA (정체 지속 시 천천히 하락 — 약 10분 메모리)
    nm_baseline_warmup:   int   = 300     # baseline 유효 최소 누적 프레임 (10초@30fps)

    # ==================== flow map 기반 체류 탐지 ====================
    dwell_threshold_sec: float = 1.5       # 차량이 같은 셀에 이 초 이상 머물면 체류로 판정
                                           # (fps 무관하게 동일한 체감 — 30fps→45f, 10fps→15f 자동 변환)
                                           # 0.5→1.5: 실시간 스트림 6~7fps 환경에서 3프레임(0.5초)은 너무 짧아
                                           # 정상 차량도 체류 판정 → 1.5초(~10프레임)로 강화
    cell_dwell_ema_up:   float = 0.03      # 셀 점유 시 EMA 상승 속도 (30프레임 연속 → ema≈0.59)
                                           # 0.05→0.03: 6fps 서행 차량 2초 체류(12f) 시 ema 0.46→0.31 억제
    cell_dwell_ema_down: float = 0.02      # 셀 이탈 시 EMA 하락 속도 (50프레임 후 ema≈0.36)

    # ==================== 방향별 차선 분리 파라미터 ====================
    lane_cos_threshold: float = 0.0        # 방향 분류 코사인 임계값 (≥ 이면 A방향, < 이면 B방향)