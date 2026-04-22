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
    ):
        self._csv_path      = str(csv_path)
        self._smooth_thr    = smooth_threshold
        self._slow_thr      = slow_threshold
        self._min_conf      = min_conf_samples

        # ── 슬롯 데이터: slot_id → [count, jam_sum] ──────────────────
        # slot_id = hour * 12 + minute // 5  (0 ~ 287)
        self._slots: dict[int, list] = {}

        # ── 현재 5분 창 버퍼 ──────────────────────────────────────────
        self._buf_slot: int       = -1   # 현재 누적 중인 슬롯 ID (-1 = 미초기화)
        self._buf_values: list    = []   # 이 슬롯에서 수집된 jam_score 리스트
        self._dirty: bool         = False

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
            self._buf_slot   = cur_slot
            self._buf_values = []

        self._buf_values.append(float(jam_score))

    def _flush_buffer(self) -> None:
        """현재 버퍼의 중앙값을 슬롯에 누적하고 CSV에 저장한다."""
        if not self._buf_values:
            return

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

    def predict(self, dt: datetime | None = None) -> list | None:
        """현재 시각 기준 5분 후 슬롯의 정체 수준을 예측한다.

        Returns
        -------
        list[dict] | None
            데이터 있으면 단일 dict 리스트:
            {"horizon_sec": 300, "horizon_min": 5,
             "predicted_level": str, "confidence": float, "jam_score": float}
            해당 슬롯 데이터가 없으면 None ("Training..." 표시).
        """
        if dt is None:
            dt = datetime.now()

        future_dt  = dt + timedelta(minutes=5)
        target_sid = self._to_slot_id(future_dt)

        slot = self._slots.get(target_sid)
        if slot is None or slot[0] == 0:
            return None

        avg_jam = slot[1] / slot[0]
        conf    = min(slot[0] / max(self._min_conf, 1), 1.0)
        level   = self._jam_to_level(avg_jam)

        return [{
            "horizon_sec":     300,
            "horizon_min":     5,
            "predicted_level": level,
            "confidence":      round(conf, 4),
            "jam_score":       round(avg_jam, 4),
        }]

    # ==================== 내부 유틸 ====================

    def _jam_to_level(self, jam_score: float) -> str:
        if jam_score < self._smooth_thr:
            return "SMOOTH"
        if jam_score < self._slow_thr:
            return "SLOW"
        return "JAM"

    # ==================== 진단 ====================

    def get_slot_count(self) -> int:
        """현재 메모리에 로드된 슬롯 수 (최대 288)."""
        return len(self._slots)

    def get_total_windows(self) -> int:
        """누적된 5분 창 수 합계."""
        return sum(v[0] for v in self._slots.values())
