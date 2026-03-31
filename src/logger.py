# 탐지 결과를 정형 데이터(CSV)로 저장하는 로거

import csv
from pathlib import Path
from datetime import datetime


class CSVLogger:
    """
    덮어쓰기 방지:
    - log_dir 아래에 매 실행마다 run_YYYYmmdd_HHMMSS 폴더를 생성
    - 그 폴더 안에 frame_log.csv / track_log.csv / events_log.csv 저장

    생성되는 파일:
      1) frame_log.csv  : 프레임 단위 요약(프레임당 1행)
      2) track_log.csv  : 트랙 단위 상세(프레임 × 모든 트랙 1행)
      3) events_log.csv : 역주행 확정 이벤트 단위(차량 1대 확정당 1행)
    """

    def __init__(self, log_dir: Path):
        # 사용자가 넣어준 log_dir는 "베이스 폴더"로 사용
        base_dir = Path(log_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # 실행마다 고유한 run 폴더 생성 (동일 초에 두 번 실행될 수도 있어 충돌 방지 루프)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"run_{run_id}"
        idx = 1
        while run_dir.exists():
            run_dir = base_dir / f"run_{run_id}_{idx}"
            idx += 1
        run_dir.mkdir(parents=True, exist_ok=False)

        # 실제 저장되는 폴더(런 폴더)
        self.log_dir = run_dir
        print(f"📝 로그 저장 폴더: {self.log_dir}")

        # Excel에서 한글 깨짐 방지를 위해 utf-8-sig(BOM) 사용
        self.frame_fp = open(self.log_dir / "frame_log.csv", "w", newline="", encoding="utf-8-sig")
        self.track_fp = open(self.log_dir / "track_log.csv", "w", newline="", encoding="utf-8-sig")
        self.event_fp = open(self.log_dir / "events_log.csv", "w", newline="", encoding="utf-8-sig")

        self.frame_writer = csv.writer(self.frame_fp)
        self.track_writer = csv.writer(self.track_fp)
        self.event_writer = csv.writer(self.event_fp)

        # ---------------- 프레임 로그 헤더 (한글) ----------------
        self.frame_writer.writerow([
            "프레임번호", "영상시간(초)", "추적중_차량수",
            "역주행확정_차량수", "Flow샘플총합",
            "장면전환감지", "모드",
            "jam_score", "정체레벨"                                 # Phase 1/2 정체 탐지 컬럼 추가
        ])

        # ---------------- 트랙 로그 헤더 (한글) ----------------
        self.track_writer.writerow([
            "프레임번호", "영상시간(초)", "트랙ID", "표시라벨(W)",
            "x1", "y1", "x2", "y2", "중심x", "중심y",
            "속도(mag)", "방향ndx", "방향ndy", "속도임계값",
            "흐름vx", "흐름vy", "현재cos",
            "판정상태", "찬성(agree)", "반대(disagree)", "스킵(skip)", "총평가수(total)",
            "반대비율", "역주행의심카운트", "역주행확정(0/1)"
        ])

        # ---------------- 이벤트 로그 헤더 (한글) ----------------
        self.event_writer.writerow([
            "라벨(W)", "트랙ID",
            "등장프레임", "의심시작프레임", "확정프레임",
            "등장→확정(초)", "의심→확정(초)"
        ])

    def log_frame(self, frame_num, time_sec, active_tracks, wrong_confirmed_count,
                  flow_samples_total, camera_switch_triggered, mode,
                  jam_score: float = 0.0, congestion_level: str = "SMOOTH"):
        """프레임당 1행 기록.

        Args:
            frame_num: 프레임 번호.
            time_sec: 영상 타임라인 시간(초).
            active_tracks: 현재 추적 중인 차량 수.
            wrong_confirmed_count: 역주행 확정 차량 수.
            flow_samples_total: flow_map 전체 샘플 수.
            camera_switch_triggered: 장면 전환 감지 여부.
            mode: "LEARNING" / "DETECTING" 등.
            jam_score: 현재 jam_score (0.0~1.0). 기본 0.0.
            congestion_level: 현재 정체 레벨. 기본 "SMOOTH".
        """
        self.frame_writer.writerow([
            frame_num, round(time_sec, 3), active_tracks,
            wrong_confirmed_count, int(flow_samples_total),
            int(bool(camera_switch_triggered)), mode,
            round(jam_score, 4), congestion_level               # 정체 탐지 컬럼
        ])

    def log_track(self, row: list):
        """트랙당 1행 기록 (row 순서는 헤더 순서와 동일해야 함)"""
        self.track_writer.writerow(row)

    def log_events_from_stats(self, detection_stats: dict):
        """종료 시 detection_stats를 이벤트 로그로 저장"""
        for label, s in sorted(detection_stats.items()):
            self.event_writer.writerow([
                label, s.get("track_id"),
                s.get("first_frame"), s.get("suspect_frame"), s.get("detect_frame"),
                s.get("seconds_from_appear"), s.get("seconds_from_suspect")
            ])

    def close(self):
        """파일 닫기"""
        self.frame_fp.close()
        self.track_fp.close()
        self.event_fp.close()