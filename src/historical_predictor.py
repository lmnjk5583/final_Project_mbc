# 파일 경로: 최종 프로젝트/src/historical_predictor.py
# 역할: 시각별(hour × 5분 슬롯) 과거 jam_score를 CSV에 누적하고,
#        5분 후 정체 수준을 예측한다.
#
# 슬롯 구조:
#   하루 = 24h × 12슬롯/h = 288 슬롯 (slot_id = hour*12 + minute//5)
#   CSV 컬럼: hour, minute_start, count, jam_sum
#     - hour        : 0~23
#     - minute_start: 0,5,10,...,55  (5분 창의 시작 분)
#     - count       : 이 슬롯에서 기록된 5분 창의 수
#     - jam_sum     : 각 5분 창 중앙값의 합계
#
# 기록 흐름 (매 프레임 호출 → 5분 창 단위로 자동 집계):
#   record(jam_score) 호출 → 내부 버퍼 누적
#   슬롯 경계(매 5분) 도달 → 버퍼 중앙값 계산 → CSV 갱신 → 버퍼 초기화
#
# 예측:
#   predict(dt) → (dt + 5분) 슬롯의 평균값 → 레벨 + 신뢰도 반환
#   데이터 없으면 None → 패널에 "Training..." 표시

import csv
import os
from datetime import datetime, timedelta


class HistoricalPredictor:
    """시각 슬롯별 jam_score 이력 기반 5분 후 정체 수준 예측기.

    Parameters
    ----------
    csv_path : str | Path
        슬롯 데이터 저장 CSV 경로. 없으면 첫 flush 시 자동 생성.
    smooth_threshold : float
        jam_score 이 값 미만 → SMOOTH.
    slow_threshold : float
        jam_score 이 값 미만 → SLOW, 이상 → JAM.
    min_conf_samples : int
        신뢰도 100%에 필요한 최소 5분 창 수. 기본값 14 (약 1시간 10분).
    """

    _COLUMNS = ("hour", "minute_start", "count", "jam_sum")

    def __init__(
        self,
        csv_path,
        smooth_threshold: float = 0.25,
        slow_threshold: float   = 0.60,
        min_conf_samples: int   = 14,
        min_window_sec: float   = 150.0,
    ):
        self._csv_path      = str(csv_path)
        self._smooth_thr    = smooth_threshold
        self._slow_thr      = slow_threshold
        self._min_conf      = min_conf_samples
        self._min_window_sec = min_window_sec  # 이 초 미만 버퍼는 flush 스킵 (기본 2.5분)

        # ── 슬롯 데이터: slot_id → [count, jam_sum] ──────────────────
        # slot_id = hour * 12 + minute // 5  (0 ~ 287)
        self._slots: dict[int, list] = {}

        # ── 현재 5분 창 버퍼 ──────────────────────────────────────────
        self._buf_slot: int          = -1    # 현재 누적 중인 슬롯 ID (-1 = 미초기화)
        self._buf_values: list       = []    # 이 슬롯에서 수집된 jam_score 리스트
        self._buf_start_dt: datetime | None = None  # 버퍼 첫 record 시각
        self._buf_last_dt:  datetime | None = None  # 버퍼 마지막 record 시각
        self._dirty: bool            = False

        self._load()

    # ==================== 슬롯 ID 계산 ====================

    @staticmethod
    def _to_slot_id(dt: datetime) -> int:
        """datetime → slot_id (0~287)."""
        return dt.hour * 12 + dt.minute // 5

    # ==================== 로드 / 저장 ====================

    def _load(self) -> None:
        """CSV가 있으면 슬롯 데이터를 메모리에 로드한다."""
        if not os.path.exists(self._csv_path):
            return
        try:
            with open(self._csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h   = int(row["hour"])
                    m   = int(row["minute_start"])
                    sid = h * 12 + m // 5
                    self._slots[sid] = [int(row["count"]), float(row["jam_sum"])]
            total = sum(v[0] for v in self._slots.values())
            print(f"📊 HistoricalPredictor 로드: {len(self._slots)}슬롯 / {total}창 ({self._csv_path})")
        except Exception as e:
            print(f"⚠️  HistoricalPredictor 로드 실패: {e}")

    def save(self) -> None:
        """슬롯 데이터를 CSV에 저장(전체 재작성)한다."""
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._csv_path)), exist_ok=True)
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._COLUMNS)
                writer.writeheader()
                for sid in sorted(self._slots):
                    h   = sid // 12
                    m   = (sid % 12) * 5
                    cnt, jsum = self._slots[sid]
                    writer.writerow({
                        "hour":         h,
                        "minute_start": m,
                        "count":        cnt,
                        "jam_sum":      round(jsum, 6),
                    })
            self._dirty = False
        except Exception as e:
            print(f"⚠️  HistoricalPredictor 저장 실패: {e}")

    # ==================== 기록 ====================

    def record(self, jam_score: float, dt: datetime | None = None) -> None:
        """현재 프레임의 jam_score를 내부 버퍼에 추가한다.

        슬롯 경계(5분 경계)를 넘어가면 이전 버퍼의 중앙값을 CSV에 기록하고
        새 버퍼를 시작한다. 매 프레임 호출하면 된다.
        _is_frame_skip=False인 프레임에서만 호출해야 한다 (호출측 책임).
        """
        if dt is None:
            dt = datetime.now()

        cur_slot = self._to_slot_id(dt)

        # ── 슬롯 경계 → 이전 버퍼 flush ──────────────────────────────
        if cur_slot != self._buf_slot:
            if self._buf_values and self._buf_slot >= 0:
                self._flush_buffer()
            self._buf_slot    = cur_slot
            self._buf_values  = []
            self._buf_start_dt = dt
            self._buf_last_dt  = None

        self._buf_last_dt = dt
        self._buf_values.append(float(jam_score))

    def _flush_buffer(self) -> None:
        """현재 버퍼의 중앙값을 슬롯에 누적하고 CSV에 저장한다.

        버퍼의 실제 시간 커버리지가 min_window_sec 미만이면 저장을 스킵한다.
        - 즉시 종료: 30초짜리 버퍼가 5분 창과 동등한 count=1로 저장되는 오염 방지
        - 카메라 전환: 학습/대기 구간 제외 후 2분치 데이터만 쌓인 창 오염 방지
        """
        if not self._buf_values:
            return

        # ── 최소 시간 커버리지 검사 ────────────────────────────────────
        if (self._buf_start_dt is not None
                and self._buf_last_dt is not None
                and self._min_window_sec > 0):
            elapsed = (self._buf_last_dt - self._buf_start_dt).total_seconds()
            if elapsed < self._min_window_sec:
                return   # 데이터 부족 → 이 창은 무시

        # ── 중앙값 계산 ────────────────────────────────────────────────
        sorted_v = sorted(self._buf_values)
        n        = len(sorted_v)
        if n % 2 == 1:
            median = sorted_v[n // 2]
        else:
            median = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0

        # ── 슬롯 누적 ──────────────────────────────────────────────────
        sid = self._buf_slot
        if sid not in self._slots:
            self._slots[sid] = [0, 0.0]
        self._slots[sid][0] += 1
        self._slots[sid][1] += median
        self._dirty = True

        # ── 즉시 flush: 5분마다 1회 → I/O 부담 없음 ──────────────────
        self.save()

    def flush_current(self) -> None:
        """프로그램 종료 시 마지막 미완성 창을 강제로 flush한다."""
        if self._buf_values and self._buf_slot >= 0:
            self._flush_buffer()
            self._buf_values = []

    # ==================== 예측 ====================

    # ── 슬롯 평균 jam 조회 (내부 헬퍼) ──────────────────────────────
    def _slot_avg(self, sid: int) -> float:
        """슬롯 평균 jam_score 반환. 데이터 없으면 -1."""
        s = self._slots.get(sid)
        if s is None or s[0] == 0:
            return -1.0
        return s[1] / s[0]

    def _slot_conf(self, sid: int) -> float:
        """슬롯 신뢰도 반환 (0~1). 데이터 없으면 0."""
        s = self._slots.get(sid)
        if s is None or s[0] == 0:
            return 0.0
        return min(s[0] / max(self._min_conf, 1), 1.0)

    # ── 빈 슬롯 보간 (최대 탐색 범위: ±_INTERP_MAX_GAP 슬롯) ────────
    _INTERP_MAX_GAP: int = 12   # 60분 = 12슬롯 이내만 보간 (이 이상이면 신뢰 불가)

    def _interpolate(self, target_sid: int) -> tuple[float, float] | None:
        """target_sid가 비었을 때 양측 이웃 슬롯으로 선형 보간한다.

        Returns
        -------
        (avg_jam, confidence) | None
            보간 성공이면 (값, 신뢰도), 범위 내 데이터 없으면 None.
        """
        # ── 이전(prev) 슬롯 탐색 ──────────────────────────────────────
        prev_sid, prev_dist = None, 0
        for d in range(1, self._INTERP_MAX_GAP + 1):
            sid = (target_sid - d) % 288
            if self._slot_avg(sid) >= 0:
                prev_sid, prev_dist = sid, d
                break

        # ── 이후(next) 슬롯 탐색 ──────────────────────────────────────
        next_sid, next_dist = None, 0
        for d in range(1, self._INTERP_MAX_GAP + 1):
            sid = (target_sid + d) % 288
            if self._slot_avg(sid) >= 0:
                next_sid, next_dist = sid, d
                break

        if prev_sid is None and next_sid is None:
            return None   # 전체 데이터 없음

        # ── 한쪽만 있으면 그대로 사용 (최근접 이웃) ──────────────────
        if prev_sid is None:
            avg_jam = self._slot_avg(next_sid)
            base_conf = self._slot_conf(next_sid)
            gap = next_dist
        elif next_sid is None:
            avg_jam = self._slot_avg(prev_sid)
            base_conf = self._slot_conf(prev_sid)
            gap = prev_dist
        else:
            # ── 양측 선형 보간 ────────────────────────────────────────
            total = prev_dist + next_dist          # 두 슬롯 간 전체 거리
            t     = prev_dist / total              # 0=prev 위치, 1=next 위치
            avg_jam   = self._slot_avg(prev_sid) * (1.0 - t) + self._slot_avg(next_sid) * t
            base_conf = min(self._slot_conf(prev_sid), self._slot_conf(next_sid))
            gap = total                            # 간격이 클수록 신뢰도 페널티

        # ── 간격 페널티: 간격 1슬롯→거의 그대로, 12슬롯→절반 ─────────
        gap_factor = max(0.0, 1.0 - gap / (self._INTERP_MAX_GAP * 2))
        conf = base_conf * gap_factor

        return float(avg_jam), float(conf)

    def predict(self, dt: datetime | None = None) -> list | None:
        """현재 시각 기준 1시간·2시간·3시간 후 슬롯의 정체 수준을 예측한다.

        각 horizon에 대해 해당 슬롯 데이터가 없으면 양측 이웃 슬롯으로 선형 보간한다.
        모든 horizon에서 보간 범위(±60분) 내에 데이터가 없으면 None ("Training..." 표시).

        Returns
        -------
        list[dict] | None
            [{"horizon_sec": int, "horizon_min": int,
              "predicted_level": str, "confidence": float, "jam_score": float,
              "interpolated": bool}, ...]  — 3개 원소 (60/120/180분)
        """
        if dt is None:
            dt = datetime.now()

        results = []
        for horizon_min in (60, 120, 180):
            future_dt  = dt + timedelta(minutes=horizon_min)
            target_sid = self._to_slot_id(future_dt)

            slot = self._slots.get(target_sid)
            if slot is not None and slot[0] > 0:
                avg_jam = slot[1] / slot[0]
                conf    = min(slot[0] / max(self._min_conf, 1), 1.0)
                interp  = False
            else:
                result = self._interpolate(target_sid)
                if result is None:
                    continue   # 이 horizon은 데이터 없음 — 건너뜀
                avg_jam, conf = result
                interp = True

            level = self._jam_to_level(avg_jam)
            results.append({
                "horizon_sec":     horizon_min * 60,
                "horizon_min":     horizon_min,
                "predicted_level": level,
                "confidence":      round(conf, 4),
                "jam_score":       round(avg_jam, 4),
                "interpolated":    interp,
            })

        return results if results else None

    # ==================== 내부 유틸 ====================

    def _jam_to_level(self, jam_score: float) -> str:
        if jam_score < self._smooth_thr:
            return "SMOOTH"
        if jam_score < self._slow_thr:
            return "SLOW"
        return "JAM"

    # ==================== 방향 반전 스왑 ====================

    def swap_slots_with(self, other: "HistoricalPredictor") -> None:
        """두 HistoricalPredictor의 슬롯 데이터를 교환하고 양쪽 CSV를 저장한다.

        카메라가 180° 회전해 a/b 방향이 바뀌었을 때 호출한다.
        메모리 내 _slots dict를 교환한 뒤 양쪽 모두 즉시 저장(강제 flush).
        """
        self._slots, other._slots = other._slots, self._slots
        # 양쪽 모두 강제 저장 (_dirty 우회)
        self._dirty = True
        other._dirty = True
        self.save()
        other.save()

    # ==================== 진단 ====================

    def get_slot_count(self) -> int:
        """현재 메모리에 로드된 슬롯 수 (최대 288)."""
        return len(self._slots)

    def get_total_windows(self) -> int:
        """누적된 5분 창 수 합계."""
        return sum(v[0] for v in self._slots.values())
