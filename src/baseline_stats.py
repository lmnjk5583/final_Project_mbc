# 파일 경로: C:\final_pj\src\baseline_stats.py
# 역할: PassageRecord(차량 통과 기록)·BaselineStats(학습 기준선 통계) 데이터 클래스 정의
# 의존성: dataclasses (표준 라이브러리)

from dataclasses import dataclass                  # 데이터 클래스 데코레이터 임포트

# ── 상수 ─────────────────────────────────────────────────────────────
MIN_PASSAGE_DIST: float = 100.0                    # 유효 통과 판정 최소 거리(px) — config에서 덮어씀


# ======================================================================
# PassageRecord — 개별 차량의 화면 진입→퇴장 기록
# ======================================================================

@dataclass                                          # 데이터 클래스 선언
class PassageRecord:
    """한 차량이 화면에 진입해서 퇴장할 때까지의 기록을 담는다."""

    track_id: int                                   # 차량 추적 ID
    entry_frame: int                                # 화면 진입 프레임 번호
    entry_fx: float                                 # 진입 시 footpoint x 좌표
    entry_fy: float                                 # 진입 시 footpoint y 좌표
    exit_frame: int                                 # 화면 퇴장 프레임 번호
    exit_fx: float                                  # 퇴장 시 footpoint x 좌표
    exit_fy: float                                  # 퇴장 시 footpoint y 좌표
    dwell_frames: int                               # 체류 프레임 수 (exit_frame - entry_frame)
    entry_exit_dist: float                          # 진입~퇴장 직선 거리 (px)
    is_complete: bool                               # True=정상 퇴장, False=추적 소실(stale)

    @property                                       # 프로퍼티: 유효 통과인지 판별
    def is_valid(self) -> bool:
        """유효한 통과 기록인지 판별한다.
        조건: (1) 정상 퇴장 (2) 체류 프레임 > 0 (3) 이동 거리 >= MIN_PASSAGE_DIST
        """
        return (self.is_complete                    # 정상 퇴장이고
                and self.dwell_frames > 0           # 체류 프레임이 0보다 크고
                and self.entry_exit_dist >= MIN_PASSAGE_DIST)  # 최소 이동 거리 이상


# ======================================================================
# BaselineStats — 학습 구간에서 산출된 기준선 통계
# ======================================================================

@dataclass                                          # 데이터 클래스 선언
class BaselineStats:
    """학습(learning) 구간 종료 시 산출되는 기준선 통계.
    정체 판정의 모든 비교 기준이 이 객체에 담긴다.
    """

    free_flow_dwell: float                          # 자유 흐름 체류 시간: n>=10이면 percentile(dwells,10), n<10이면 min(dwells)
    typical_dwell: float                            # 일반 체류 시간: median(dwells)
    norm_speed_ref: float                           # 학습 구간 전체 normalized_mag의 중앙값
    count_ref: float                                # 학습 구간 전체 frame당 평균 활성 차량 수
    bbox_slope: float                               # bbox_h ~ cy 선형회귀 기울기 (원근 보정용)
    bbox_intercept: float                           # bbox_h ~ cy 선형회귀 절편 (원근 보정용)
    lcs: float                                      # Learning Congestion Score (0.0~1.0) — 학습 품질
    quality_warning: bool                           # True이면 typical_dwell > free_flow_dwell × 3 (학습 중 정체 의심)
    passage_count: int                              # 학습 구간 유효 passage 수
    is_fallback: bool                               # True=passage 부족 → fallback 모드 (규칙 기반 판정만 사용)
