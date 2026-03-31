# 파일 경로: C:\final_pj\src\passage_tracker.py
# 역할: 차량 진입/퇴장 기록(PassageRecord) 관리, dwell 집계,
#        BaselineStats 산출, LCS(Learning Congestion Score) 계산
# 의존성: math(표준), numpy(서드파티), baseline_stats(로컬)

import math                                            # hypot — 진입~퇴장 거리 계산용
import numpy as np                                     # median, percentile, polyfit 등 수치 연산

from baseline_stats import PassageRecord               # 개별 차량 통과 기록 데이터 클래스
from baseline_stats import BaselineStats               # 학습 기준선 통계 데이터 클래스


# ======================================================================
# PassageTracker — 차량 통과 기록 관리 + 기준선 통계 산출
# ======================================================================

class PassageTracker:
    """차량의 화면 진입·퇴장을 기록하고, 학습 구간 종료 시
    BaselineStats를 산출하는 모듈.

    Parameters
    ----------
    cfg : DetectorConfig
        min_passage_dist, min_passages_required, stop_mag_threshold 등 사용.
    state : DetectorState
        first_seen_frame, entry_positions 참조.
    """

    # ── 온라인 학습 EMA 속도 (baseline 점진 갱신용) ───────────────────
    _ONLINE_ALPHA: float = 0.01                        # 느린 EMA — 기준선은 천천히 갱신

    def __init__(self, cfg, state):
        """PassageTracker 초기화.

        Args:
            cfg: DetectorConfig — Phase 1 파라미터 포함.
            state: DetectorState — first_seen_frame, entry_positions 참조.
        """
        self.cfg = cfg                                 # 설정 객체 저장
        self.state = state                             # 런타임 상태 객체 저장

        # ── 통과 기록 ────────────────────────────────────────────────
        self._passages: list = []                      # 완료된 PassageRecord 목록 (유효·무효 모두 포함)
        self._entry_data: dict = {}                    # {track_id: (fx, fy, frame_num)} 진입 정보 캐시

        # ── 프레임별 누적 통계 (LCS·BaselineStats 산출용) ────────────
        self._frame_counts: list = []                  # [프레임별 활성 차량 수] — count_ref 산출
        self._exit_counts: list = []                   # [프레임별 퇴장 차량 수] — signal_C 산출
        self._all_norm_mags: list = []                 # 전 프레임 normalized_mag 평탄화 — norm_speed_ref 산출
        self._all_cy_vals: list = []                   # 전 프레임 cy 값 평탄화 — bbox 회귀용
        self._all_bbox_h_vals: list = []               # 전 프레임 bbox_h 값 평탄화 — bbox 회귀용

        # ── 산출된 기준선 (finalize_baseline 후 설정) ────────────────
        self._baseline: BaselineStats | None = None    # 학습 완료 후 기준선 객체

    # ==================================================================
    # 진입·퇴장 기록
    # ==================================================================

    def on_entry(self, track_id: int, fx: float, fy: float,
                 frame_num: int):
        """새 차량이 화면에 처음 등장할 때 호출.

        Args:
            track_id: ByteTrack 추적 ID.
            fx: footpoint x — (x1+x2)/2.
            fy: footpoint y — y2 (바운딩박스 하단).
            frame_num: 등장 프레임 번호.
        """
        self._entry_data[track_id] = (fx, fy, frame_num)  # 내부 캐시에 진입 정보 저장
        self.state.entry_positions[track_id] = (fx, fy)    # state에도 기록 (다른 모듈 참조용)

    def on_exit(self, track_id: int, fx: float, fy: float,
                frame_num: int, is_complete: bool = True):
        """차량이 화면에서 사라질 때 호출 — PassageRecord를 생성한다.

        Args:
            track_id: ByteTrack 추적 ID.
            fx: 퇴장 시 footpoint x.
            fy: 퇴장 시 footpoint y.
            frame_num: 퇴장 프레임 번호.
            is_complete: True=정상 퇴장, False=stale(추적 소실).
        """
        # ── 진입 정보 조회 ───────────────────────────────────────────
        entry = self._entry_data.pop(track_id, None)   # 캐시에서 꺼냄 (제거)
        if entry is not None:                          # 캐시에 있으면 사용
            entry_fx, entry_fy, entry_frame = entry    # 진입 footpoint + 프레임
        else:                                          # 캐시에 없으면 state에서 복원
            entry_frame = self.state.first_seen_frame.get(  # 첫 등장 프레임 조회
                track_id, frame_num                    # 없으면 현재 프레임 (방어)
            )
            entry_pos = self.state.entry_positions.get(     # 진입 위치 조회
                track_id, (fx, fy)                     # 없으면 현재 위치 (방어)
            )
            entry_fx, entry_fy = entry_pos             # 진입 footpoint 복원

        # ── PassageRecord 생성 ───────────────────────────────────────
        dwell_frames = frame_num - entry_frame         # 체류 프레임 수
        entry_exit_dist = math.hypot(                  # 진입~퇴장 직선 거리 (픽셀)
            fx - entry_fx, fy - entry_fy
        )

        record = PassageRecord(                        # 통과 기록 데이터 클래스 생성
            track_id=track_id,                         # 추적 ID
            entry_frame=entry_frame,                   # 진입 프레임
            entry_fx=entry_fx,                         # 진입 footpoint x
            entry_fy=entry_fy,                         # 진입 footpoint y
            exit_frame=frame_num,                      # 퇴장 프레임
            exit_fx=fx,                                # 퇴장 footpoint x
            exit_fy=fy,                                # 퇴장 footpoint y
            dwell_frames=dwell_frames,                 # 체류 프레임 수
            entry_exit_dist=entry_exit_dist,           # 진입~퇴장 거리
            is_complete=is_complete,                    # 정상 퇴장 여부
        )
        self._passages.append(record)                  # 기록 목록에 추가

    # ==================================================================
    # 프레임별 통계 수집
    # ==================================================================

    def record_frame_stats(self, active_count: int, exit_count: int,
                           norm_mags: list, cy_vals: list,
                           bbox_h_vals: list):
        """매 프레임 호출 — 활성 차량 수·퇴장 수·속도·위치 정보를 누적한다.

        Args:
            active_count: 이번 프레임 활성 차량 수.
            exit_count: 이번 프레임 퇴장 차량 수.
            norm_mags: [normalized_mag] — 각 활성 차량의 원근 보정 속도.
            cy_vals: [cy] — 각 활성 차량의 중심 y 좌표 (bbox 회귀용).
            bbox_h_vals: [bbox_h] — 각 활성 차량의 바운딩박스 높이.
        """
        self._frame_counts.append(active_count)        # 활성 차량 수 기록
        self._exit_counts.append(exit_count)           # 퇴장 차량 수 기록
        self._all_norm_mags.extend(norm_mags)          # norm_mag 평탄화 누적
        self._all_cy_vals.extend(cy_vals)              # cy 값 평탄화 누적
        self._all_bbox_h_vals.extend(bbox_h_vals)      # bbox_h 값 평탄화 누적

    # ==================================================================
    # 조회 메서드
    # ==================================================================

    def get_completed_count(self) -> int:
        """유효 완성 passage 수를 반환한다.

        유효 조건: is_complete=True, dwell_frames>0, entry_exit_dist>=MIN_PASSAGE_DIST.
        """
        return sum(1 for p in self._passages if p.is_valid)  # is_valid True인 것만 카운트

    def get_current_dwells(self, active_ids: set,
                           frame_num: int) -> list:
        """현재 활성 차량들의 체류 프레임 수 리스트를 반환한다.

        Args:
            active_ids: 현재 프레임에 보이는 차량 ID 집합.
            frame_num: 현재 프레임 번호.

        Returns:
            [dwell_frames, ...] — 각 활성 차량의 현재까지 체류 프레임 수.
        """
        dwells = []                                    # 결과 리스트 초기화
        for tid in active_ids:                         # 활성 ID 순회
            entry_frame = self.state.first_seen_frame.get(  # 첫 등장 프레임 조회
                tid, frame_num                         # 없으면 현재 프레임 (dwell=0)
            )
            dwells.append(frame_num - entry_frame)     # 체류 프레임 = 현재 - 첫등장
        return dwells                                  # 리스트 반환

    def get_exit_count_recent(self, n_frames: int) -> int:
        """최근 n_frames 프레임 동안 퇴장한 차량 수를 반환한다.

        Args:
            n_frames: 슬라이딩 윈도우 크기 (프레임 수).

        Returns:
            최근 n_frames 동안의 퇴장 차량 합계.
        """
        if not self._exit_counts:                      # 기록 없으면 0
            return 0                                   # 퇴장 없음
        return sum(self._exit_counts[-n_frames:])      # 최근 n개 합산

    # ==================================================================
    # LCS (Learning Congestion Score) 계산
    # ==================================================================

    def compute_lcs(self) -> float:
        """현재까지 수집된 데이터로 LCS를 계산한다.

        LCS = 0.40×signal_A + 0.35×signal_B + 0.25×signal_C

        signal_A: dwell 분포 편중도 (median/min 비율)
        signal_B: 속도 저하 정도 (1 - mean/max)
        signal_C: 차량 누적 정도 (1 - exit/entry)

        Returns:
            0.0~1.0 사이 점수. 데이터 부족 시 0.5.
        """
        # ── 유효 passage의 dwell 목록 ────────────────────────────────
        valid_dwells = [p.dwell_frames                 # 유효 passage의 체류 프레임만 추출
                        for p in self._passages
                        if p.is_valid]

        if len(valid_dwells) < 2:                      # 유효 passage 2개 미만이면
            return 0.5                                 # 판단 불가 → 중간값 반환

        # ── signal_A: dwell 분포 편중 ────────────────────────────────
        min_dwell = max(min(valid_dwells), 1)          # 최소 dwell (0 방지 → 1)
        median_dwell = float(np.median(valid_dwells))  # 중앙값 dwell
        signal_A = min(                                # 편중도 계산 (0~1 클램프)
            (median_dwell / min_dwell - 1) / 5.0,      # (중앙/최소 - 1) / 5
            1.0                                        # 상한 1.0
        )

        # ── signal_B: 속도 저하 정도 ────────────────────────────────
        if self._all_norm_mags:                        # norm_mag 데이터가 있으면
            max_speed = max(                           # 최대 속도 (0 방지)
                max(self._all_norm_mags), 1e-6
            )
            mean_speed = float(np.mean(                # 평균 속도
                self._all_norm_mags
            ))
            signal_B = 1.0 - mean_speed / max_speed    # 속도 저하 비율
        else:                                          # 데이터 없으면
            signal_B = 0.5                             # 중간값

        # ── signal_C: 차량 누적 (유출 부족) ──────────────────────────
        if self._frame_counts:                         # 프레임 데이터가 있으면
            total_entry = sum(self._frame_counts) + 1e-6  # 전체 활성 합계 (0 방지)
            total_exit = sum(self._exit_counts)        # 전체 퇴장 합계
            signal_C = max(                            # 누적 비율 (하한 0)
                0.0,
                1.0 - total_exit / total_entry         # 1 - (퇴장/활성)
            )
        else:                                          # 데이터 없으면
            signal_C = 0.5                             # 중간값

        # ── 가중 합산 ────────────────────────────────────────────────
        lcs = (0.40 * signal_A                         # dwell 편중 기여 (40%)
               + 0.35 * signal_B                       # 속도 저하 기여 (35%)
               + 0.25 * signal_C)                      # 차량 누적 기여 (25%)
        return lcs                                     # LCS 반환 (0.0~1.0)

    # ==================================================================
    # BaselineStats 산출 (학습 종료 시 1회 호출)
    # ==================================================================

    def finalize_baseline(self) -> BaselineStats:
        """학습 구간 종료 시 호출 — 수집된 데이터로 BaselineStats를 산출한다.

        Returns:
            BaselineStats 인스턴스. passage 부족 시 is_fallback=True.
        """
        # ── 유효 passage의 dwell 목록 ────────────────────────────────
        valid_dwells = [p.dwell_frames                 # 유효 passage dwell만 추출
                        for p in self._passages
                        if p.is_valid]
        n = len(valid_dwells)                          # 유효 passage 수

        # ── fallback 판정 ────────────────────────────────────────────
        is_fallback = (n < self.cfg.min_passages_required)  # 최소 passage 미달이면 fallback

        # ── free_flow_dwell / typical_dwell ──────────────────────────
        if n == 0:                                     # 유효 passage 없음
            free_flow_dwell = 1.0                      # 기본값 1.0 (fallback)
            typical_dwell = 1.0                        # 기본값 1.0 (fallback)
        elif n < 10:                                   # 10개 미만 → min 사용
            free_flow_dwell = float(min(valid_dwells)) # 가장 짧은 dwell = 자유 흐름
            typical_dwell = float(np.median(valid_dwells))  # 중앙값 = 일반 체류
        else:                                          # 10개 이상 → percentile 사용
            free_flow_dwell = float(                   # 하위 10% = 자유 흐름
                np.percentile(valid_dwells, 10)
            )
            typical_dwell = float(np.median(valid_dwells))  # 중앙값 = 일반 체류

        # ── norm_speed_ref ───────────────────────────────────────────
        if self._all_norm_mags:                        # 속도 데이터가 있으면
            norm_speed_ref = float(                    # 전체 norm_mag의 중앙값
                np.median(self._all_norm_mags)
            )
        else:                                          # 없으면 기본값
            norm_speed_ref = 0.1                       # 방어용 기본값

        # ── count_ref ────────────────────────────────────────────────
        if self._frame_counts:                         # 프레임 카운트 데이터가 있으면
            count_ref = float(                         # 프레임당 평균 활성 차량 수
                np.mean(self._frame_counts)
            )
        else:                                          # 없으면 기본값
            count_ref = 1.0                            # 방어용 기본값

        # ── bbox 선형 회귀 (원근 보정) ───────────────────────────────
        if len(self._all_cy_vals) >= 2:                # 데이터 2개 이상이면 회귀 가능
            coeffs = np.polyfit(                       # 1차 다항식 피팅: bbox_h ≈ slope × cy + intercept
                self._all_cy_vals,                     # x축: cy 좌표
                self._all_bbox_h_vals,                 # y축: bbox 높이
                1                                      # 1차 (직선)
            )
            bbox_slope = float(coeffs[0])              # 기울기
            bbox_intercept = float(coeffs[1])          # 절편
        else:                                          # 데이터 부족이면 기본값
            bbox_slope = 0.0                           # 기울기 0 (보정 없음)
            bbox_intercept = 50.0                      # 기본 bbox_h 50px

        # ── LCS 계산 ────────────────────────────────────────────────
        lcs = self.compute_lcs()                       # 현재 데이터 기반 LCS

        # ── 품질 경고 ────────────────────────────────────────────────
        quality_warning = (                            # 학습 중 정체 의심 판정
            n > 0                                      # 유효 passage 있고
            and typical_dwell > free_flow_dwell * 3    # 일반 체류가 자유흐름의 3배 초과
        )

        # ── BaselineStats 생성 ───────────────────────────────────────
        baseline = BaselineStats(                      # 기준선 통계 데이터 클래스
            free_flow_dwell=free_flow_dwell,           # 자유 흐름 체류 시간
            typical_dwell=typical_dwell,               # 일반 체류 시간
            norm_speed_ref=norm_speed_ref,             # 정상 속도 기준
            count_ref=count_ref,                       # 정상 차량 수 기준
            bbox_slope=bbox_slope,                     # bbox 회귀 기울기
            bbox_intercept=bbox_intercept,             # bbox 회귀 절편
            lcs=lcs,                                   # 학습 품질 점수
            quality_warning=quality_warning,           # 품질 경고 플래그
            passage_count=n,                           # 유효 passage 수
            is_fallback=is_fallback,                   # fallback 모드 여부
        )
        self._baseline = baseline                      # 내부에 기준선 저장 (온라인 갱신용)
        return baseline                                # 기준선 반환

    # ==================================================================
    # 온라인 학습 (SMOOTH 구간에서만 호출)
    # ==================================================================

    def update_baseline(self, fx: float, fy: float,
                        norm_mag: float):
        """SMOOTH 상태일 때 기준선을 점진적으로 갱신한다.

        detector.py에서 enable_online_flow_update 조건 + SMOOTH 판정 후 호출.

        Args:
            fx: footpoint x (현재 사용하지 않으나 확장용).
            fy: footpoint y (현재 사용하지 않으나 확장용).
            norm_mag: 해당 차량의 normalized_mag.
        """
        if self._baseline is None:                     # 기준선 미생성 상태면
            return                                     # 갱신 불가 → 스킵
        alpha = self._ONLINE_ALPHA                     # EMA 속도 0.01 (느린 갱신)
        self._baseline.norm_speed_ref = (              # norm_speed_ref EMA 갱신
            (1 - alpha) * self._baseline.norm_speed_ref  # 기존값 99%
            + alpha * norm_mag                         # 새 값 1%
        )

    # ==================================================================
    # 초기화
    # ==================================================================

    def reset(self):
        """카메라 전환 시 모든 내부 상태를 초기화한다."""
        self._passages.clear()                         # 통과 기록 전체 삭제
        self._entry_data.clear()                       # 진입 캐시 삭제
        self._frame_counts.clear()                     # 프레임 활성수 삭제
        self._exit_counts.clear()                      # 프레임 퇴장수 삭제
        self._all_norm_mags.clear()                    # 속도 데이터 삭제
        self._all_cy_vals.clear()                      # cy 데이터 삭제
        self._all_bbox_h_vals.clear()                  # bbox_h 데이터 삭제
        self._baseline = None                          # 기준선 삭제
        self.state.entry_positions.clear()             # state 진입 위치 삭제
