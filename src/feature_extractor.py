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

    feature 벡터 8차원:
    ┌─────┬───────────────────┬──────────────────────────────────────────────┐
    │ idx │ 이름              │ 계산식                                       │
    ├─────┼───────────────────┼──────────────────────────────────────────────┤
    │  0  │ norm_speed_ratio  │ median(upper_50%_nm) / norm_speed_ref       │
    │  1  │ count_ratio       │ active_count / count_ref, clip 0~3          │
    │  2  │ stop_ratio        │ (nm<0.06 차량) / speed_known_count          │
    │ 2.5 │ slow_ratio        │ (0.06≤nm<0.50 차량) / speed_known_count     │
    │  3  │ exit_rate_ratio   │ exit_last_30f / (count_ref×0.5), clip 0~3   │
    │  4  │ dwell_ratio       │ free_flow_dwell / mean(dwells), clip 0~1    │
    │  5  │ density_score     │ occupied_cells / density_max_vehicles        │
    │  6  │ rule_jam_score    │ 0.0 (congestion_judge가 채워넣음)            │
    └─────┴───────────────────┴──────────────────────────────────────────────┘

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
        stopped_count = 0                              # 정지 차량 카운터 (nm < 0.06)
        slow_count = 0                                 # 서행 차량 카운터 (0.06 ≤ nm < slow_upper_nm)
        norm_stop_thr = getattr(                       # norm_stop_threshold 없으면 구버전 호환
            self.cfg, "norm_stop_threshold", 0.05
        )
        slow_upper_nm = getattr(                       # 서행 상한 nm — 역주행 게이트(0.15)와 별개
            self.cfg, "slow_upper_nm", 0.50            # 기본 0.50: nm≥0.50 → 정상 주행
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
                slow_count += 1                        # 정지는 서행의 부분집합 — 서행에도 포함
                continue
            nm = mag / bbox_h                          # normalized_mag = mag / bbox_h (원근 보정)
            norm_mags.append(nm)                       # 속도 목록에 추가
            if nm < norm_stop_thr:                     # nm < 0.06 → 저속 정지
                stopped_count += 1                     # 정지 카운트
                slow_count += 1                        # 정지는 서행의 부분집합 — 서행에도 포함
            elif nm < slow_upper_nm:                   # 0.06 ≤ nm < 0.50 → 서행 구간
                slow_count += 1                        # 서행 카운트

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
        total_cells = grid_size * grid_size            # 전체 셀 수 (400 = 20×20)
        raw_density = len(occupied_cells) / total_cells    # 원시 점유율 (고속도로 20대 = 0.05 수준)

        # ── density 정규화: 현실적 최대 차량 수 기준으로 0~1 확장 ───────
        # 20×20 그리드에서 차량 20대 → raw_density = 0.05 → 기여 거의 0
        # density_max_vehicles 대 이상이면 포화(1.0)로 처리
        density_max = getattr(self.cfg, "density_max_vehicles", 20.0)  # 포화 기준 차량 수
        density_scale = density_max / total_cells          # 포화 기준 점유율 (예: 20/400 = 0.05)
        density_score = float(np.clip(                     # 정규화: raw / 포화기준, 최대 1.0
            raw_density / density_scale, 0.0, 1.0
        ))

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
        # ── 소표본 신뢰도 보정: 차량 수가 3대 미만이면 stop_ratio 최대 기여 제한 ──
        # 차량 2대 중 1대 정지 → stop_ratio=0.50(원본) vs 0.33(보정)
        # 차량 1대 정지     → stop_ratio=1.00(원본) vs 0.33(보정)
        # 차량 3대+        → 보정 없음 (신뢰도 충분)
        _MIN_RELIABLE = 3                                  # 신뢰 가능 최소 차량 수
        _raw_stop = stopped_count / max(speed_known_count, 1)  # 원시 stop_ratio
        if speed_known_count < _MIN_RELIABLE:              # 차량 수 부족 → 신뢰도 가중치 적용
            _reliability = speed_known_count / _MIN_RELIABLE  # 0 ~ 1 신뢰도 (차량수/3)
            stop_ratio = _raw_stop * _reliability          # 최대 기여 (1/3, 2/3, 1) 제한
        else:                                              # 차량 수 충분 → 그대로 사용
            stop_ratio = _raw_stop                         # 신뢰도 보정 불필요

        # [2.5] slow_ratio: 궤적 확인된 차량 중 서행 비율 (0.06 ≤ nm < 0.15)
        # 서행 차량은 정지도 아니고 정상도 아닌 중간 영역 — jam_score fallback 핵심 지표
        # stop_ratio와 동일한 소표본 신뢰도 보정 적용
        _raw_slow = slow_count / max(speed_known_count, 1) # 원시 slow_ratio
        if speed_known_count < _MIN_RELIABLE:              # 차량 수 부족 → 신뢰도 가중치 적용
            slow_ratio = _raw_slow * _reliability          # _reliability는 stop_ratio에서 이미 계산됨
        else:                                              # 차량 수 충분 → 그대로 사용
            slow_ratio = _raw_slow                         # 신뢰도 보정 불필요

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
        return {                                       # 8차원 feature 벡터
            "norm_speed_ratio": norm_speed_ratio,      # [0] 속도 비율
            "count_ratio":      count_ratio,           # [1] 차량 수 비율
            "stop_ratio":       stop_ratio,            # [2] 정지 비율 (nm < 0.06)
            "slow_ratio":       slow_ratio,            # [2.5] 서행 비율 (0.06 ≤ nm < 0.15)
            "exit_rate_ratio":  exit_rate_ratio,       # [3] 퇴장률 비율
            "dwell_ratio":      dwell_ratio,           # [4] 체류 비율
            "density_score":    density_score,         # [5] 밀도 점수
            "rule_jam_score":   0.0,                   # [6] jam_score (CJ 채움)
        }
