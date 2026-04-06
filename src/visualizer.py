# 모든 시각화를 담당
# 궤적, 방향 화살표, 속도, 역주행 경고, 흐름장, 투표 디버그, 내적값, 패널, 키보드 토글

import cv2
import numpy as np


class Visualizer:
    def __init__(self, cfg, state, flow_map):
        self.cfg = cfg
        self.st = state
        self.flow = flow_map

        # ==================== 시각화 옵션 ====================
        self.show_trails = True          # 궤적(꼬리) 선 표시 ON/OFF
        self.show_direction = True       # 차량 이동 방향 화살표 표시 ON/OFF
        self.show_flow = False           # 학습된 흐름장(배경 화살표) 표시 ON/OFF
        self.show_speed = False           # 차량 속도 텍스트 표시 ON/OFF
        self.show_info_panel = False      # 왼쪽 상단 정보 패널 표시 ON/OFF
        self.show_vote_debug = False     # 역주행 투표 디버그(점/텍스트) 표시 ON/OFF
        self.show_dot_product = False     # 내적(코사인 유사도) 값 표시 ON/OFF
        self.show_detection_stats = True # 탐지 소요시간 패널 ON/OFF
        self.show_congestion_panel = True  # 하단 Down/Up 정체 패널 ON/OFF (단축키 C)
        self.show_bbox = False           # 차량 바운딩박스 표시 ON/OFF (단축키 B, 기본 OFF)

    # ==================== 키보드 입력 처리 ====================
    def handle_keys(self, key):
        """실행 도중 키보드 입력으로 시각화 옵션 토글"""
        if key == ord("t"):
            self.show_trails = not self.show_trails
            print(f"궤적: {'ON' if self.show_trails else 'OFF'}")
        elif key == ord("d"):
            self.show_direction = not self.show_direction
            print(f"방향: {'ON' if self.show_direction else 'OFF'}")
        elif key == ord("f"):
            self.show_flow = not self.show_flow
            print(f"흐름장: {'ON' if self.show_flow else 'OFF'}")
        elif key == ord("s"):
            self.show_speed = not self.show_speed
            print(f"속도: {'ON' if self.show_speed else 'OFF'}")
        elif key == ord("i"):
            self.show_info_panel = not self.show_info_panel
            print(f"패널: {'ON' if self.show_info_panel else 'OFF'}")
        elif key == ord("v"):
            self.show_vote_debug = not self.show_vote_debug
            print(f"투표 디버그: {'ON' if self.show_vote_debug else 'OFF'}")
        elif key == ord("p"):
            self.show_dot_product = not self.show_dot_product
            print(f"내적값: {'ON' if self.show_dot_product else 'OFF'}")
        elif key == ord("c"):                                      # C키: 정체 패널 토글
            self.show_congestion_panel = not self.show_congestion_panel
            print(f"정체 패널: {'ON' if self.show_congestion_panel else 'OFF'}")
        elif key == ord("b"):                                      # B키: 바운딩박스 토글
            self.show_bbox = not self.show_bbox
            print(f"바운딩박스: {'ON' if self.show_bbox else 'OFF'}")

    # ==================== 궤적 그리기 ====================
    def draw_trajectory(self, frame, track_id, is_wrong=False):
        """한 차량의 최근 궤적(선)을 프레임 위에 그리기"""
        traj = self.st.trajectories[track_id]  # 해당 차량의 궤적 리스트
        if len(traj) < 2:
            return  # 점이 2개 미만이면 그릴 수 없음

        for i in range(1, len(traj)):
            alpha = i / len(traj)                              # 오래된 점일수록 작은 값 (0~1)
            thickness = max(1, int(alpha * 3))                 # 최근 점일수록 선 두껍게

            if is_wrong:  # 역주행이면 붉은 계열 그라데이션
                grad_color = (0, int(100 * (1 - alpha)), int(200 + 55 * alpha))
            else:         # 정상은 초록 계열 그라데이션
                grad_color = (0, int(255 * alpha), 0)

            pt1 = (int(traj[i - 1][0]), int(traj[i - 1][1]))  # 이전 위치
            pt2 = (int(traj[i][0]), int(traj[i][1]))           # 현재 위치
            cv2.line(frame, pt1, pt2, grad_color, thickness, cv2.LINE_AA)

        # 마지막 점에 현재 위치 표시 (점)
        cx, cy = int(traj[-1][0]), int(traj[-1][1])
        dot_color = (0, 0, 255) if is_wrong else (0, 255, 0)  # 역주행: 빨강, 정상: 초록
        cv2.circle(frame, (cx, cy), 4, dot_color, -1, cv2.LINE_AA)

    # ==================== 방향 화살표 ====================
    def draw_direction_arrow(self, frame, cx, cy, vdx, vdy, speed, is_wrong=False):
        """차량의 이동 방향을 화살표로 표시"""
        arrow_length = min(60, max(25, speed * 3))    # 속도에 비례한 화살표 길이 (최소 25, 최대 60)
        end_x = int(cx + vdx * arrow_length)          # 화살표 끝점 x
        end_y = int(cy + vdy * arrow_length)          # 화살표 끝점 y

        color = (0, 0, 255) if is_wrong else (0, 255, 0)  # 역주행: 빨강, 정상: 초록
        thickness = 3 if is_wrong else 2                   # 역주행은 더 두껍게

        cv2.arrowedLine(frame, (int(cx), int(cy)), (end_x, end_y),
                        color, thickness, cv2.LINE_AA, tipLength=0.35)

        if is_wrong and self.st.frame_num % 6 < 3:  # 역주행인 경우 화살표 끝에 X 마크 깜빡임
            cv2.drawMarker(frame, (end_x, end_y), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 15, 2)

    # ==================== 속도 라벨 ====================
    def draw_speed_label(self, frame, x1, y2, speed, cy, is_wrong=False):
        """바운딩 박스 아래쪽에 '현재 속도 / 임계 속도'를 텍스트로 표시"""
        ratio = cy / self.st.frame_h
        threshold = self.cfg.base_speed_threshold * (0.3 + 0.7 * ratio)  # 위치 기반 속도 임계값
        speed_text = f"{speed:.1f}/{threshold:.1f}"                      # ex) "15.3/10.0"
        color = (0, 0, 255) if is_wrong else (200, 200, 200)             # 역주행: 빨강, 정상: 회색

        (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (int(x1), int(y2) + 2),           # 텍스트 배경 박스
                      (int(x1) + tw + 4, int(y2) + th + 6), (0, 0, 0), -1)
        cv2.putText(frame, speed_text, (int(x1) + 2, int(y2) + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # ==================== 역주행 경고 ====================
    def draw_wrong_way_alert(self, frame, track_id, x1, y1, x2, y2):
        """역주행 경고 표시 (박스 + 'WRONG WAY!! [W1]' 텍스트)"""
        if self.st.frame_num % 4 < 2:  # 깜빡이는 빨간 테두리 (2프레임마다 on/off)
            cv2.rectangle(frame, (int(x1) - 3, int(y1) - 3),
                          (int(x2) + 3, int(y2) + 3), (0, 0, 255), 3)

        display_label = self.st.display_id_map.get(track_id)  # W1, W2 등의 라벨
        if display_label:
            label = f"WRONG WAY!! [{display_label}]"
        else:
            label = f"WRONG WAY!! ID:{track_id}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (int(x1), int(y1) - th - 12),    # 텍스트 배경 박스
                      (int(x1) + tw + 8, int(y1) - 2), (0, 0, 200), -1)
        cv2.putText(frame, label, (int(x1) + 4, int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ==================== 정상 차량 박스 ====================
    def draw_normal_box(self, frame, track_id, x1, y1, x2, y2):
        """정상 차량은 초록 박스/ID만 표시"""
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # ==================== 투표 디버그 ====================
    def draw_vote_debug(self, frame, track_id, debug_info, x2, y1):
        """디버그용: 각 샘플 포인트의 투표 결과(agree/disagree/skip)를 시각화"""
        if not debug_info or not debug_info.get("points"):
            # CONFIRMED 상태일 때 간단히 LOCKED 표시
            if debug_info and debug_info.get("status") == "CONFIRMED":
                dl = self.st.display_id_map.get(track_id, f"ID:{track_id}")
                cv2.putText(frame, f"{dl} LOCKED",
                            (int(x2) + 5, int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            return

        # 각 투표 포인트에 색깔 점으로 표시
        for px, py, cos_val, vote_type in debug_info["points"]:
            px, py = int(px), int(py)
            if vote_type == "agree":         # 정상 방향
                cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
            elif vote_type == "disagree":    # 역방향
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
            else:                            # skip
                cv2.circle(frame, (px, py), 6, (128, 128, 128), -1)

        total = debug_info["total"]                            # 총 투표 수
        if total > 0:
            ratio = debug_info["disagree"] / total             # 역방향 비율
            vote_text = f"{debug_info['agree']}ok {debug_info['disagree']}bad ({ratio:.0%})"

            (tw, th), _ = cv2.getTextSize(vote_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            tx, ty = int(x2) + 5, int(y1)
            cv2.rectangle(frame, (tx, ty - th - 4),            # 텍스트 배경 박스
                          (tx + tw + 4, ty + 2), (0, 0, 0), -1)

            color = (0, 0, 255) if ratio >= self.cfg.vote_threshold else (0, 200, 0)
            cv2.putText(frame, vote_text, (tx + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # ==================== 흐름장(배경 화살표) ====================
    def draw_flow(self, frame):
        """flow_map의 raw 셀 데이터를 셀 중심 위치에 화살표로 표시.

        기존: get_interpolated() 사용 → conflict 시 None 반환 → 화살표 누락
        수정: self.flow.flow[r,c] raw 벡터를 직접 읽어 셀 중심에 그림.
              judge.py는 여전히 get_interpolated()를 사용하므로 탐지 정확도 무관.
        """
        gs = self.flow.grid_size                               # 그리드 크기
        cw = self.flow.cell_w                                  # 셀 너비
        ch = self.flow.cell_h                                  # 셀 높이
        import numpy as np                                     # norm 계산용
        for r in range(gs):                                    # 행 순회
            for c in range(gs):                                # 열 순회
                v = self.flow.flow[r, c]                       # raw 셀 벡터 직접 읽기
                mag = np.linalg.norm(v)                        # 벡터 크기
                if mag < 0.1:                                  # 데이터 없는 셀 건너뜀
                    continue
                uv = v / (mag + 1e-6)                          # 단위 벡터
                cx = int((c + 0.5) * cw)                       # 셀 중심 x
                cy = int((r + 0.5) * ch)                       # 셀 중심 y
                ex = int(cx + uv[0] * 18)                      # 화살표 끝 x
                ey = int(cy + uv[1] * 18)                      # 화살표 끝 y
                cnt = self.flow.count[r, c]                    # 학습 샘플 수
                # 학습 데이터 있는 셀: 밝은 cyan / smoothing만 된 셀: 어두운 cyan
                color = (255, 255, 0) if cnt > 0 else (120, 120, 0)
                cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                                color, 1, tipLength=0.4)        # 화살표 그리기

    # ==================== 내적(코사인 유사도) 표시 ====================
    def draw_dot_product(self, frame, x2, y1, cos_values, is_wrong=False):
        """디버그용: 한 차량의 내적(코사인 유사도) 통계를 텍스트로 표시"""
        if not cos_values:
            return

        avg_cos = np.mean(cos_values)   # 평균 코사인 값
        min_cos = min(cos_values)       # 최소값
        max_cos = max(cos_values)       # 최대값

        # 평균이 양수면 초록 계열, 음수면 빨강 계열
        if avg_cos > 0:
            color = (0, int(min(255, avg_cos * 255)), 0)
        else:
            color = (0, 0, int(min(255, abs(avg_cos) * 255)))

        text1 = f"dot: {avg_cos:.2f}"            # 평균값
        text2 = f"[{min_cos:.2f}~{max_cos:.2f}]" # 범위

        tx = int(x2) + 5   # 텍스트 x 위치
        ty = int(y1) + 15  # 텍스트 y 위치

        (tw1, th1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        (tw2, th2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        max_tw = max(tw1, tw2)  # 두 줄 중 더 긴 너비

        cv2.rectangle(frame, (tx - 2, ty - th1 - 4),     # 배경 박스
                      (tx + max_tw + 4, ty + th2 + 8), (0, 0, 0), -1)

        cv2.putText(frame, text1, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(frame, text2, (tx, ty + th2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        thr_text = f"thr: {self.cfg.cos_threshold}"  # 기준 임계값 표시
        cv2.putText(frame, thr_text, (tx, ty + th2 * 2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 255), 1, cv2.LINE_AA)

    # ==================== 탐지 소요시간 통계 패널(우측 상단) ====================
    def draw_detection_stats(self, frame):
        """우측 상단에 역주행 탐지 소요시간 통계 패널 표시"""
        st = self.st

        # 아직 역주행으로 확정된 차량(W1, W2...)이 없다면 표시할 게 없으므로 반환
        if not st.detection_stats:
            return

        # detection_stats: { 'W1': {...}, 'W2': {...}, ... } → 라벨 이름으로 정렬
        stats_list = sorted(st.detection_stats.items())
        n = len(stats_list)

        # 패널 크기 계산
        line_h = 20           # 한 줄당 세로 높이
        header_h = 55         # 상단 제목/헤더 영역 높이
        panel_h = header_h + line_h * (n + 2) + 10  # +2는 '차량별 라인들' + '평균 라인'
        panel_w = 340
        px = st.frame_w - panel_w - 10  # 패널 시작 x (오른쪽 여백 10px)
        py = 10                          # 패널 시작 y (위쪽 여백 10px)

        # 반투명 배경 그리기
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 100, 255), 1)

        # 제목 영역
        cv2.putText(frame, "Detection Time Stats", (px + 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"(FPS: {st.video_fps:.1f})", (px + 220, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # 컬럼 헤더: ID / Appear / Suspect / Total
        y_off = py + 42
        for lbl, x_pos in [("ID", 10), ("Appear", 55), ("Suspect", 135), ("Total", 225)]:
            cv2.putText(frame, lbl, (px + x_pos, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # 헤더 아래 구분선
        y_off += 5
        cv2.line(frame, (px + 5, y_off), (px + panel_w - 5, y_off), (80, 80, 80), 1)

        # 각 역주행 차량별 통계 (W1, W2...)
        y_off += line_h
        all_appear_sec = []   # 각 라벨의 '등장→확정' 시간 (초)
        all_suspect_sec = []  # 각 라벨의 '의심→확정' 시간 (초)

        for label, stats in stats_list:
            # 예: "14f (0.47s)" 형식 문자열
            appear_text = f"{stats['frames_from_appear']}f ({stats['seconds_from_appear']:.1f}s)"
            suspect_text = f"{stats['frames_from_suspect']}f ({stats['seconds_from_suspect']:.1f}s)"
            total_sec = stats["seconds_from_appear"]  # 등장→확정 초

            all_appear_sec.append(stats["seconds_from_appear"])    # 등장→확정 초 리스트에 추가
            all_suspect_sec.append(stats["seconds_from_suspect"])  # 의심→확정 초 리스트에 추가

            # 한 줄 출력: ID / Appear / Suspect / Total
            cv2.putText(frame, label, (px + 10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, appear_text, (px + 55, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, suspect_text, (px + 135, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 100), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{total_sec:.1f}s", (px + 250, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            y_off += line_h

        # 평균 통계 (하단 요약 영역)
        if len(all_appear_sec) > 0:
            y_off += 3
            cv2.line(frame, (px + 5, y_off), (px + panel_w - 5, y_off), (80, 80, 80), 1)
            y_off += line_h

            # 전체 역주행 차량들의 평균 등장→확정 / 의심→확정 시간(초)
            avg_appear = np.mean(all_appear_sec)
            avg_suspect = np.mean(all_suspect_sec)

            cv2.putText(frame, "AVG", (px + 10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{avg_appear:.1f}s", (px + 55, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{avg_suspect:.1f}s", (px + 135, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{avg_appear:.1f}s", (px + 250, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    # ==================== 정보 패널(좌측 상단) ====================
    def draw_info_panel(self, frame, active_count):
        """좌측 상단에 시스템 상태/통계 정보 패널 표시"""
        st = self.st
        cfg = self.cfg

        panel_h = 250  # 패널 높이
        panel_w = 300  # 패널 너비

        # 반투명 오버레이 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)     # 70% overlay + 30% 원본
        cv2.rectangle(frame, (5, 5), (panel_w, panel_h), (100, 100, 100), 1)  # 테두리

        y_off = 25   # 첫 줄 y 위치
        gap = 22     # 줄 간 간격

        # 상태 텍스트 및 색상 (학습 / 재학습 / 감지 중)
        if st.relearning:
            status, status_color = "RE-LEARNING", (0, 165, 255)   # 주황색 계열
        elif st.is_learning:
            status, status_color = "LEARNING", (0, 200, 255)      # 하늘색 계열
        else:
            status, status_color = "DETECTING", (0, 255, 0)       # 초록색

        cv2.putText(frame, f"Status: {status}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        # 현재 프레임 번호
        y_off += gap
        cv2.putText(frame, f"Frame: {st.frame_num}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 학습/재학습 진행률 바
        if st.is_learning or st.relearning:
            y_off += gap
            if st.relearning:
                elapsed = st.frame_num - st.relearn_start_frame
                progress = min(100, elapsed / cfg.relearn_frames * 100)  # 재학습 진행률
            else:
                progress = min(100, st.frame_num / cfg.learning_frames * 100)  # 초기 학습 진행률

            bar_w = 150                            # 진행률 바의 너비
            filled = int(bar_w * progress / 100)   # 채워진 부분 너비

            # 진행률 바 바탕
            cv2.rectangle(frame, (15, y_off - 5), (15 + bar_w, y_off + 10), (50, 50, 50), -1)
            # 채워진 부분
            cv2.rectangle(frame, (15, y_off - 5), (15 + filled, y_off + 10), status_color, -1)
            cv2.putText(frame, f"{progress:.0f}%", (170, y_off + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        # 현재 추적 중인 차량 수
        y_off += gap
        cv2.putText(frame, f"Tracking: {active_count} vehicles", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 역주행 차량 수 (W1, W2... 라벨 기준)
        y_off += gap
        wrong_count = len(set(st.display_id_map.values()))
        wrong_color = (0, 0, 255) if wrong_count > 0 else (100, 100, 100)
        cv2.putText(frame, f"Wrong-way vehicles: {wrong_count}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, wrong_color, 1, cv2.LINE_AA)

        # 역주행 라벨 목록 (예: [W1, W2])
        if wrong_count > 0:
            y_off += gap
            labels = sorted(set(st.display_id_map.values()))
            cv2.putText(frame, f"  [{', '.join(labels)}]", (15, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # flow_map에 쌓인 전체 샘플 수(학습에 사용된 벡터 수)
        y_off += gap
        cv2.putText(frame, f"Samples: {self.flow.count.sum()}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 속도 임계값 범위 (화면 위쪽~아래쪽)
        y_off += gap
        lo = cfg.base_speed_threshold * 0.3
        hi = cfg.base_speed_threshold * 1.0
        cv2.putText(frame, f"Speed thr: {lo:.1f}~{hi:.1f}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # ID 재매칭 횟수 근사: 라벨 개수 대비 사용된 ID 개수로 추정
        y_off += gap
        id_changes = max(0, len(st.display_id_map) - wrong_count)
        cv2.putText(frame, f"ID re-matches: {id_changes}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 카메라 상태: 재학습 중인지, 정상 모니터링 중인지
        y_off += gap
        cam = "Switched" if st.relearning else "Monitoring"
        cam_color = (0, 165, 255) if st.relearning else (0, 255, 0)
        cv2.putText(frame, f"Camera: {cam}", (15, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1, cv2.LINE_AA)

    # ==================== 학습 중 상태 패널 ====================
    def draw_learning_status(self, frame, frame_num: int):
        """학습·재학습 중 좌하단에 '데이터 수집 중' 패널을 표시한다."""
        fh, fw = frame.shape[:2]                                   # 프레임 크기
        panel_w, panel_h = 280, 52                                 # 패널 너비·높이
        px, py = 8, fh - panel_h - 8                               # 패널 좌상단 위치
        overlay = frame.copy()                                     # 반투명 배경용 복사
        cv2.rectangle(overlay, (px, py),
                      (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)     # 75% 불투명 합성
        cv2.rectangle(frame, (px, py),
                      (px + panel_w, py + panel_h), (0, 165, 255), 2)  # 주황 테두리
        cv2.putText(frame, "Analyzing traffic...",                # 메인 텍스트
                    (px + 10, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Collecting data  ({frame_num} frames)",  # 프레임 수 표시
                    (px + 10, py + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # ==================== 조치 권고 텍스트 생성 ====================
    @staticmethod
    def _get_action_text(level: str, duration_sec: float) -> str:
        """정체 레벨·지속 시간에 따라 조치 권고 문자열을 반환한다.

        plan.md §3.1 기반:
          SLOW       → VSL 하향 권고 + VMS 서행 안내
          CONGESTED  → VMS 우회 안내 + 램프 미터링 권고
          CONGESTED 5분+ → 순찰대 출동 요청 추가

        Args:
            level: "SMOOTH", "SLOW", "CONGESTED".
            duration_sec: SLOW/CONGESTED 지속 시간(초).

        Returns:
            조치 권고 문자열. SMOOTH이면 빈 문자열.
        """
        if level == "SMOOTH":                                      # 원활 — 조치 불필요
            return ""
        if level == "SLOW":                                        # 서행
            return "VSL reduce  |  VMS: slow down"                # Papageorgiou 2006, MDPI 2024
        if level == "CONGESTED":                                   # 정체
            if duration_sec >= 300:                                # 5분(300초) 이상 지속
                return "VMS detour  |  Ramp meter  |  Patrol dispatch"  # FHWA CHART
            return "VMS detour  |  Ramp metering"                 # ALINEA 1991, MDPI 2024
        return ""                                                  # 알 수 없는 레벨

    # ==================== 정체 상태 패널(하단 좌우 분리, Down/Up 각각) ====================
    def draw_congestion_status(self, frame,
                               level_a: str, jam_score_a: float, duration_sec_a: float,
                               level_b: str, jam_score_b: float, duration_sec_b: float,
                               label_a: str = "UP", label_b: str = "DOWN"):
        """화면 하단 좌측(Down)·우측(Up)에 방향별 정체 상태 미니 패널을 그린다.

        label_a/label_b 값에 따라 자동으로 DOWN↔UP 데이터를 좌우에 배치한다.
        두 패널 사이 위쪽에 조치 권고 텍스트를 표시한다.

        Args:
            frame: BGR 이미지 프레임.
            level_a: A방향 정체 레벨 ("SMOOTH"/"SLOW"/"CONGESTED").
            jam_score_a: A방향 jam_score (0.0~1.0).
            duration_sec_a: A방향 SLOW/CONGESTED 지속 시간(초).
            level_b: B방향 정체 레벨.
            jam_score_b: B방향 jam_score.
            duration_sec_b: B방향 지속 시간(초).
            label_a: A방향 표시 레이블 ("UP" 또는 "DOWN").
            label_b: B방향 표시 레이블 ("DOWN" 또는 "UP").
        """
        fh, fw = frame.shape[:2]                                   # 프레임 높이·너비

        # ── 레벨별 색상 (BGR) ─────────────────────────────────────
        _lv_colors = {                                             # 레벨→BGR 색상 매핑
            "SMOOTH":    (0, 200, 0),                              # 초록 — 원활
            "SLOW":      (0, 200, 255),                            # 노랑 — 서행
            "CONGESTED": (0, 0, 255),                              # 빨강 — 정체
        }

        # ── label 기준으로 Down/Up 데이터 매핑 ────────────────────
        # label_a가 "DOWN"이면 A=Down(좌), B=Up(우), 아니면 반대
        if label_a == "DOWN":                                      # A방향이 하행이면
            down_lv, down_jam, down_dur = level_a, jam_score_a, duration_sec_a
            up_lv,   up_jam,   up_dur   = level_b, jam_score_b, duration_sec_b
        else:                                                      # A방향이 상행이면
            up_lv,   up_jam,   up_dur   = level_a, jam_score_a, duration_sec_a
            down_lv, down_jam, down_dur = level_b, jam_score_b, duration_sec_b
        color_down = _lv_colors.get(down_lv, (200, 200, 200))     # Down 색상
        color_up   = _lv_colors.get(up_lv,   (200, 200, 200))     # Up 색상

        # ── 최악 방향 기준 조치 권고 ──────────────────────────────
        _lv_order = {"SMOOTH": 0, "SLOW": 1, "CONGESTED": 2}      # 레벨 심각도 순서
        if _lv_order.get(down_lv, 0) >= _lv_order.get(up_lv, 0): # Down이 더 나쁘면
            worst_level, worst_dur = down_lv, down_dur             # 최악 = Down
        else:                                                      # Up이 더 나쁘면
            worst_level, worst_dur = up_lv, up_dur                 # 최악 = Up
        action = self._get_action_text(worst_level, worst_dur)     # 권고 문자열

        # ── 미니 패널 크기 ────────────────────────────────────────
        panel_w = 190                                              # 각 패널 너비 (기존 380 → 반으로)
        panel_h = 58                                               # 기본 패널 높이 (지속시간 없을 때)
        dur_extra = 14                                             # 지속시간 표시 시 추가 높이
        margin = 8                                                 # 화면 가장자리 여백
        bar_w = panel_w - 52                                       # jam bar 너비 (수치 공간 확보)

        def _draw_mini_panel(px, py, title, level, jam, dur_sec, color):
            """미니 패널 1개 (제목+레벨, jam bar, 지속시간) 그리기."""
            p_h = panel_h + (dur_extra if dur_sec > 0 else 0)     # 지속시간 유무에 따라 높이 조정
            # 반투명 배경
            overlay = frame.copy()                                 # 원본 복사 (합성용)
            cv2.rectangle(overlay, (px, py),
                          (px + panel_w, py + p_h), (20, 20, 20), -1)  # 어두운 배경
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)  # 75% 불투명 합성
            cv2.rectangle(frame, (px, py),
                          (px + panel_w, py + p_h), color, 2)     # 레벨 색상 테두리
            # 제목 + 레벨 텍스트 ("Down  CONGESTED")
            cv2.putText(frame, f"{title}  {level}",
                        (px + 8, py + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
            # jam_score 바
            bar_x = px + 8                                         # 바 시작 x
            bar_y = py + 27                                        # 바 시작 y
            bar_h = 10                                             # 바 높이
            filled = int(bar_w * min(1.0, max(0.0, jam)))         # 채워진 너비
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)  # 바 배경
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + filled, bar_y + bar_h), color, -1)  # 채워진 부분
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)  # 바 테두리
            cv2.putText(frame, f"{jam:.2f}",                       # jam 수치 텍스트
                        (bar_x + bar_w + 4, bar_y + bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
            # 지속 시간 (SLOW/CONGESTED 유지 시간)
            if dur_sec > 0:                                        # 지속시간 있을 때만
                mins = int(dur_sec // 60)                          # 분 계산
                secs = int(dur_sec % 60)                           # 초 계산
                dur_text = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                cv2.putText(frame, dur_text,                       # 지속시간 텍스트
                            (px + 8, py + p_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, cv2.LINE_AA)

        # ── Down 패널 (하단 좌측) ──────────────────────────────────
        down_h = panel_h + (dur_extra if down_dur > 0 else 0)     # Down 패널 실제 높이
        up_h   = panel_h + (dur_extra if up_dur   > 0 else 0)     # Up 패널 실제 높이
        max_h  = max(down_h, up_h)                                 # 두 패널 중 더 큰 높이 (y 기준 통일)
        py_panel = fh - max_h - margin                             # 공통 상단 y (하단 기준)
        _draw_mini_panel(margin, py_panel,                         # 좌측 여백에 배치
                         "Down", down_lv, down_jam, down_dur, color_down)

        # ── Up 패널 (하단 우측) ────────────────────────────────────
        _draw_mini_panel(fw - panel_w - margin, py_panel,          # 우측 여백에 배치
                         "Up", up_lv, up_jam, up_dur, color_up)

        # ── 조치 권고 (두 패널 사이 위쪽, 중앙 정렬) ─────────────
        if action:                                                 # 조치 있을 때만
            font_scale = 0.52                                      # 텍스트 크기 (기존 0.38 → 가시성 향상)
            font_thick = 2                                         # 텍스트 두께 (기존 1 → 가시성 향상)
            (tw, th), baseline = cv2.getTextSize(                  # 텍스트 크기 측정 (중앙 정렬용)
                action, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            text_x = (fw - tw) // 2                               # 화면 가로 중앙
            text_y = py_panel - 10                                 # 패널 바로 위에 표시
            # 반투명 배경 박스 (텍스트 가독성 확보)
            pad = 6                                                # 배경 박스 패딩
            overlay = frame.copy()                                 # 합성용 복사
            cv2.rectangle(overlay,
                          (text_x - pad, text_y - th - pad),      # 박스 좌상단
                          (text_x + tw + pad, text_y + baseline + pad),  # 박스 우하단
                          (20, 20, 20), -1)                        # 어두운 배경
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)  # 40% 불투명 합성
            cv2.putText(frame, action,                             # 조치 권고 텍스트
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 200, 255), font_thick, cv2.LINE_AA)