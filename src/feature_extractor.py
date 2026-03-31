# 파일 경로: C:\final_pj\src\feature_extractor.py
# 역할: 매 프레임 tracks+speeds를 받아 7차원 feature 벡터를 산출한다.
#        모든 값은 학습 구간 baseline 대비 비율 → 화각 변경 후에도 의미 동일.
# 의존성: numpy(서드파티), baseline_stats(로컬)

import numpy as np                                     # clip, mean 등 수치 연산

from baseline_stats import BaselineStats               # 학습 기준선 통계 데이터 클래스


# ======================================================================
# FeatureExtractor — 7차원 feature 벡터 계산기
# ======================================================================

class FeatureExtractor:
    """매 프레임 tracks·speeds를 받아 정체 판정용 feature 벡터를 산출한다.

    feature 벡터 7차원:
    ┌─────┬───────────────────┬────────────────────────────────────────────┐
    │ idx │ 이름              │ 계산식                                     │
    ├─────┼───────────────────┼────────────────────────────────────────────┤
    │  0  │ norm_speed_ratio  │ mean(norm_mags) / norm_speed_ref, clip 0~1│
    │  1  │ count_ratio       │ active_count / count_ref, clip 0~3        │
    │  2  │ stop_ratio        │ stopped / active_count                    │
    │  3  │ exit_rate_ratio   │ exit_last_30f / (count_ref×0.5), clip 0~3 │
    │  4  │ dwell_ratio       │ free_flow_dwell / mean(dwells), clip 0~1  │
    │  5  │ density_score     │ occupied_cells / total_cells              │
    │  6  │ rule_jam_score    │ 0.0 (congestion_judge가 채워넣음)          │
    └─────┴───────────────────┴────────────────────────────────────────────┘

    Parameters
    ----------
    cfg : DetectorConfig
        grid_size, stop_mag_threshold 등 사용.
    state : DetectorState
        first_seen_frame, frame_w, frame_h 참조.
    passage_tracker : PassageTracker
        get_current_dwells(), get_exit_count_recent() 호출.
    baseline_stats : BaselineStats or None
        학습 완료 후 set_baseline()으로 설정.
    """

    def __init__(self, cfg, state, passage_tracker,
                 baseline_stats=None):
        """FeatureExtractor 초기화.

        Args:
            cfg: DetectorConfig — grid_size, stop_mag_threshold 등.
            state: DetectorState — first_seen_frame, frame_w, frame_h.
            passage_tracker: PassageTracker — dwell·exit 조회용.
            baseline_stats: BaselineStats or None — 학습 완료 전 None.
        """
        self.cfg = cfg                                 # 설정 객체 저장
        self.state = state                             # 런타임 상태 저장
        self.passage_tracker = passage_tracker         # PassageTracker 참조
        self.baseline: BaselineStats | None = baseline_stats  # 기준선 (초기 None)

    # ── 기준선 설정 ──────────────────────────────────────────────────
    def set_baseline(self, baseline_stats: BaselineStats):
        """학습 완료 후 BaselineStats를 설정한다.

        Args:
            baseline_stats: finalize_baseline()이 반환한 기준선 객체.
        """
        self.baseline = baseline_stats                 # 기준선 갱신

    # ── feature 벡터 계산 ────────────────────────────────────────────
    def compute(self, tracks: list, speeds: dict,
                flow_map, frame_num: int) -> dict | None:
        """매 프레임 호출 — 7차원 feature 벡터를 계산한다.

        Args:
            tracks: [{id, x1, y1, x2, y2, cx, cy, ...}, ...].
            speeds: {track_id: mag(픽셀 이동량)}.
            flow_map: FlowMap 객체 (현재 Phase 1에서 미사용, Phase 2 확장용).
            frame_num: 현재 프레임 번호.

        Returns:
            dict(7차원 feature) — baseline 미설정 시 None.
        """
        if self.baseline is None:                      # 학습 완료 전이면
            return None                                # feature 계산 불가 → None

        # ── 기준값 로컬 참조 (가독성) ────────────────────────────────
        bl = self.baseline                             # 기준선 단축 참조
        active_count = len(tracks)                     # 현재 활성 차량 수

        # ── 차량별 normalized_mag 계산 ───────────────────────────────
        norm_mags = []                                 # 차량별 원근 보정 속도 리스트
        stopped_count = 0                              # 정지 차량 카운터
        norm_stop_thr = getattr(                       # norm_stop_threshold 없으면 구버전 호환
            self.cfg, "norm_stop_threshold", 0.05
        )
        min_bbox_h = getattr(                          # min_bbox_h 없으면 구버전 호환 (30px)
            self.cfg, "min_bbox_h", 30.0
        )
        speed_known_count = 0                          # 궤적 확인된 차량 수 (신규 제외)
        for t in tracks:                               # 각 차량 순회
            raw_bbox_h = t["y2"] - t["y1"]            # 바운딩박스 높이 (픽셀)
            bbox_h = max(raw_bbox_h, min_bbox_h)       # 최솟값 클램프 — 원거리 소형 박스 과대평가 방지
            tid = t["id"]                              # 트랙 ID
            if tid not in speeds:                      # speeds에 없으면 신규 → 완전 제외
                continue
            mag = speeds[tid]                          # 속도 조회
            speed_known_count += 1                     # 궤적 확인 차량 수 증가
            if mag <= 0:                               # speeds=0: 실제 정지 확정
                stopped_count += 1                     # 정지 카운트 (nm 계산 없이)
                continue
            nm = mag / bbox_h                          # normalized_mag = mag / bbox_h (원근 보정)
            norm_mags.append(nm)                       # 속도 목록에 추가
            if nm < norm_stop_thr:                     # nm < threshold이면 저속 정지로 판단
                stopped_count += 1                     # 정지 카운트

        # ── 활성 차량의 현재 dwell 조회 ──────────────────────────────
        active_ids = {t["id"] for t in tracks}         # 활성 차량 ID 집합
        current_dwells = self.passage_tracker.get_current_dwells(  # 체류 프레임 리스트
            active_ids, frame_num
        )

        # ── 최근 퇴장 차량 수 (exit_rate_window 기본 30프레임) ───────
        exit_last_30 = self.passage_tracker.get_exit_count_recent(  # 최근 30프레임 퇴장 수
            self.cfg.exit_rate_window
        )

        # ── 밀도 점수 (셀 점유율) ────────────────────────────────────
        grid_size = self.cfg.grid_size                 # 그리드 크기 (15)
        cell_h = self.state.frame_h / grid_size        # 셀 높이 (픽셀)
        cell_w = self.state.frame_w / grid_size        # 셀 너비 (픽셀)
        occupied_cells = set()                         # 차량이 있는 셀 좌표 집합
        for t in tracks:                               # 각 차량 순회
            fx = t.get("fx", (t["x1"] + t["x2"]) / 2) # footpoint x (없으면 cx 대용)
            fy = t.get("fy", t["y2"])                  # footpoint y (없으면 y2)
            r = int(np.clip(fy / cell_h, 0, grid_size - 1))  # 셀 행 (범위 클램프)
            c = int(np.clip(fx / cell_w, 0, grid_size - 1))  # 셀 열 (범위 클램프)
            occupied_cells.add((r, c))                 # 셀 좌표 추가
        total_cells = grid_size * grid_size            # 전체 셀 수 (225)
        density_score = len(occupied_cells) / total_cells  # 점유율 (0~1)

        # ── 7차원 feature 벡터 계산 ──────────────────────────────────
        # [0] norm_speed_ratio: 현재 속도 / 기준 속도 (1.0이면 정상)
        # 상위 50% 중앙값 사용: 정체 차량 소수가 nm을 낮춰도 정상 주행 차량의 속도를 반영
        sorted_nms = sorted(norm_mags)                 # nm 오름차순 정렬
        upper_half = sorted_nms[len(sorted_nms) // 2:]  # 상위 50% 슬라이싱
        rep_norm_mag = float(np.median(upper_half)) if norm_mags else 0.0  # 상위 50% 중앙값
        norm_speed_ratio = float(np.clip(              # 속도 비율 clip(0, 1)
            rep_norm_mag / max(bl.norm_speed_ref, 0.01),  # 기준 대비 비율
            0.0, 1.0                                   # 상한 1.0 (기준 이상은 정상)
        ))

        # [1] count_ratio: 현재 차량 수 / 기준 차량 수 (1.0이면 정상)
        count_ratio = float(np.clip(                   # 차량 수 비율 clip(0, 3)
            active_count / max(bl.count_ref, 1),       # 기준 대비 비율
            0.0, 3.0                                   # 상한 3.0 (3배 초과 클램프)
        ))

        # [2] stop_ratio: 궤적 확인된 차량 중 정지 비율
        # 분모: speed_known_count (신규 제외, 실제 정지·이동 모두 포함)
        stop_ratio = stopped_count / max(speed_known_count, 1)

        # [3] exit_rate_ratio: 퇴장률 / 기준 퇴장률 (높을수록 원활)
        baseline_exit_rate = max(bl.count_ref * 0.5, 0.01)  # 기준 퇴장률 = count_ref × 0.5
        exit_rate_ratio = float(np.clip(               # 퇴장률 비율 clip(0, 3)
            exit_last_30 / baseline_exit_rate,         # 실제 퇴장 / 기준
            0.0, 3.0                                   # 상한 3.0
        ))

        # [4] dwell_ratio: 자유흐름 체류 / 현재 평균 체류 (1.0이면 정상)
        avg_dwell = float(np.mean(current_dwells)) if current_dwells else 1.0  # 평균 체류
        dwell_ratio = float(np.clip(                   # 체류 비율 clip(0, 1)
            bl.free_flow_dwell / max(avg_dwell, 1),    # 자유흐름 / 현재 (1.0이면 동일)
            0.0, 1.0                                   # 상한 1.0
        ))

        # [5] density_score: 셀 점유율 (위에서 계산)
        # [6] rule_jam_score: CongestionJudge가 채워넣을 예정 (초기 0.0)

        # ── feature 딕셔너리 조립 ────────────────────────────────────
        return {                                       # 7차원 feature 벡터
            "norm_speed_ratio": norm_speed_ratio,      # [0] 속도 비율
            "count_ratio":      count_ratio,           # [1] 차량 수 비율
            "stop_ratio":       stop_ratio,            # [2] 정지 비율
            "exit_rate_ratio":  exit_rate_ratio,       # [3] 퇴장률 비율
            "dwell_ratio":      dwell_ratio,           # [4] 체류 비율
            "density_score":    density_score,         # [5] 밀도 점수
            "rule_jam_score":   0.0,                   # [6] jam_score (CJ 채움)
        }
