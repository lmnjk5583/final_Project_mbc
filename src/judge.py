# 파일 경로: C:\final_pj\src\judge.py
# 역할: 흐름장과 코사인 유사도 기반 역주행 판별
#        원근 기반 속도 게이트 + 다중 포인트 투표 + 시간적 히스테리시스(카운팅)
#        ③ 다중 스케일 윈도우: 단기(velocity_window) + 장기(×2) 모두 역방향이어야 의심
#        ① 전체 궤적 방향 검증: 확정 시 traj[0]→traj[-1] 전체 방향도 역방향이어야 확정

import numpy as np


class WrongWayJudge:
    def __init__(self, cfg, flow_map, state):
        self.cfg = cfg                  # 설정 객체 저장
        self.flow = flow_map            # FlowMap 참조
        self.st = state                 # DetectorState 참조

    def get_speed_threshold(self, cy):
        """화면 상의 y 위치에 따라 속도 임계값을 달리 적용 (원근 보정)"""
        ratio = cy / self.st.frame_h        # 화면 상단(0)~하단(1) 비율
        scale = 0.3 + 0.7 * ratio           # 위는 0.3배, 아래는 1.0배 근처
        return self.cfg.base_speed_threshold * scale  # 위치에 따른 속도 임계값

    # ── 내부 유틸: 단일 방향 벡터가 flow와 역방향인지 확인 ──────────────
    def _is_against_flow(self, ndx, ndy, px, py):
        """(ndx, ndy) 방향이 (px, py) 위치의 flow 벡터와 역방향인지 반환.

        Args:
            ndx, ndy: 단위 방향 벡터.
            px, py: flow 벡터를 조회할 위치.

        Returns:
            True=역방향, False=정방향 또는 판단 불가(flow 없음).
        """
        flow_v = self.flow.get_interpolated(px, py)  # 해당 위치 flow 벡터 조회
        if flow_v is None:                            # flow 없으면 판단 불가
            return False
        cos = float(ndx * flow_v[0] + ndy * flow_v[1])  # 코사인 유사도
        return cos < self.cfg.cos_threshold           # 임계값 미만이면 역방향

    def check(self, track_id, traj, ndx, ndy, speed, cy):
        """한 차량에 대해 flow_map과 방향 비교, 투표 방식으로 역주행 여부 판정.

        Args:
            track_id: ByteTrack 추적 ID.
            traj: 해당 차량의 (cx, cy) 궤적 리스트.
            ndx, ndy: velocity_window 기반 단기 단위 방향 벡터.
            speed: 단기 이동량(픽셀).
            cy: 현재 y 좌표 (원근 보정용).

        Returns:
            (is_wrong: bool, disagree_ratio: float, debug_info: dict)
        """
        cfg = self.cfg                                # 설정 단축 참조
        st = self.st                                  # 상태 단축 참조

        # ── 이미 확정된 차량은 즉시 True ────────────────────────────────
        if track_id in st.wrong_way_ids:
            return True, 1.0, {"status": "CONFIRMED", "cos_values": []}

        # ── 원근 기반 속도 게이트 ────────────────────────────────────────
        adaptive_threshold = self.get_speed_threshold(cy)   # 위치별 속도 임계값
        if speed < adaptive_threshold:                       # 너무 느리면 판정 X
            # 속도 부족 = 방향 불명확 → last_correct_frame 갱신 안 함
            # (실제 역주행 차량도 초기에 느릴 수 있으므로 설정 시 역주행 탐지 차단 위험)
            return False, 0, {"status": "slow", "cos_values": []}

        # ── 방향 급변 필터 (CCTV 글자/오클루전 오탐 차단) ──────────────
        # CCTV 텍스트 오버레이로 차량이 가려지면 YOLO bbox 위치가 틀어지고
        # velocity_window 기반 endpoint-to-endpoint 방향 벡터가 급격히 반전된다.
        # 직전 프레임의 방향 벡터와 현재 방향이 cos < -0.5 (120° 이상 차이)이면
        # "방향 급변" → 오클루전 노이즈로 판단해 이번 프레임 판정을 건너뜀.
        prev_vel = st.last_velocity.get(track_id)           # 직전 방향 벡터
        if prev_vel is not None:                            # 기록된 이전 방향이 있으면
            cos_dir = float(                                # 현재 vs 이전 방향 코사인
                ndx * prev_vel[0] + ndy * prev_vel[1]
            )
            if cos_dir < -0.5:                              # 120° 이상 방향 급변
                st.last_velocity[track_id] = (ndx, ndy)    # 방향 갱신 후 건너뜀
                st.last_correct_frame[track_id] = st.frame_num  # 방향 급변 = 직전까지 정상
                return False, 0, {"status": "dir_jump_filtered", "cos_values": []}
        st.last_velocity[track_id] = (ndx, ndy)            # 현재 방향 기록 (다음 프레임용)

        # ── 단기 투표: 궤적 포인트 샘플링 → disagree_ratio 계산 ─────────
        n_points = min(len(traj), 8)                   # 최대 8포인트 샘플링
        step = max(1, len(traj) // n_points)           # 궤적 샘플링 간격

        agree = 0                                      # 정방향 투표 수
        disagree = 0                                   # 역방향 투표 수
        skip = 0                                       # flow 없어서 스킵한 수
        debug_points = []                              # 디버그용 포인트 목록
        cos_values = []                                # 내적값 목록 (표시용)

        for idx in range(0, len(traj), step):          # 샘플 포인트 순회
            px, py = traj[idx]                         # 궤적 포인트 좌표
            flow_v = self.flow.get_interpolated(px, py)  # 해당 위치 flow 벡터

            if flow_v is None:                         # 유효한 flow 없으면
                skip += 1
                debug_points.append((px, py, 0, "skip"))
                continue

            cos_sim = ndx * flow_v[0] + ndy * flow_v[1]  # 코사인 유사도
            cos_values.append(cos_sim)                 # 내적값 기록

            if cos_sim < self.cfg.cos_threshold:      # 역방향 판정 (jitter 보정 적용)
                disagree += 1
                debug_points.append((px, py, cos_sim, "disagree"))
            else:                                      # 정방향 판정
                agree += 1
                debug_points.append((px, py, cos_sim, "agree"))

        total_checked = agree + disagree               # 실제 평가된 포인트 수

        # debug_info 기본 구조 (long/global 검사 결과는 아래서 추가)
        debug_info = {
            "agree": agree,
            "disagree": disagree,
            "skip": skip,
            "total": total_checked,
            "points": debug_points,
            "threshold": adaptive_threshold,
            "status": "voting",
            "cos_values": cos_values,
            "long_cos": None,     # ③ 장기 윈도우 cos 값 (디버그용)
            "global_cos": None,   # ① 전체 궤적 cos 값 (디버그용)
        }

        if total_checked < 3:                          # 샘플 부족(flow 없는 구역 등) 시 판정 보류
            # flow 없음 = 방향 불명확 → last_correct_frame 갱신 안 함
            # (실제 역주행 차량이 eroded 구역 통과 시 오판 방지)
            return False, 0, debug_info

        disagree_ratio = disagree / total_checked      # 역방향 비율

        # ── 단기 투표 통과 여부 확인 ─────────────────────────────────────
        if disagree_ratio < cfg.vote_threshold:        # 역방향 비율 낮으면
            st.wrong_way_count[track_id] = max(        # 의심 카운트 감소
                0, st.wrong_way_count[track_id] - 2
            )
            st.last_correct_frame[track_id] = st.frame_num  # 정상 주행 프레임 갱신
            return False, disagree_ratio, debug_info

        # ── ③ 장기 윈도우 검사 (다중 스케일) ───────────────────────────
        # 단기(velocity_window=15f) 투표를 통과했어도,
        # 장기(velocity_window×2=30f) 방향도 역방향이어야 의심으로 인정.
        # 차선 변경·일시 곡선 같은 단기 노이즈는 장기 방향이 정상이므로 필터링.
        long_window = cfg.velocity_window * 2          # 장기 윈도우 크기 (기본 30프레임)
        long_suspect = True                            # 기본값: 패스 (궤적 부족 시 생략)

        if len(traj) >= long_window:                   # 장기 궤적이 쌓인 경우만 검사
            lvdx = traj[-1][0] - traj[-long_window][0]  # 장기 x 이동량
            lvdy = traj[-1][1] - traj[-long_window][1]  # 장기 y 이동량
            lmag = np.sqrt(lvdx**2 + lvdy**2)           # 장기 이동 크기 (픽셀)
            avg_lmove = lmag / long_window               # 장기 프레임당 평균 이동

            if (lmag > cfg.min_move_distance            # 누적 이동 충분
                    and avg_lmove > cfg.min_move_per_frame):  # 프레임당 이동 충분
                lndx = lvdx / lmag                      # 장기 단위 x 방향
                lndy = lvdy / lmag                      # 장기 단위 y 방향
                cx_last, cy_last = traj[-1]             # 현재 위치 (flow 조회용)
                flow_v_long = self.flow.get_interpolated(cx_last, cy_last)

                if flow_v_long is not None:             # flow 유효하면
                    long_cos = float(                   # 장기 방향 코사인
                        lndx * flow_v_long[0] + lndy * flow_v_long[1]
                    )
                    debug_info["long_cos"] = round(long_cos, 4)  # 디버그 기록
                    long_suspect = (long_cos < self.cfg.cos_threshold)  # 역방향이어야 True (jitter 보정 적용)
                # flow 없으면 long_suspect = True (장기 검사 면제)
            # 이동 부족(느린 차량)이면 long_suspect = True (장기 검사 면제)

        if not long_suspect:                           # 장기 방향이 정상 → 단기 노이즈로 간주
            st.wrong_way_count[track_id] = max(
                0, st.wrong_way_count[track_id] - 2
            )
            st.last_correct_frame[track_id] = st.frame_num  # 장기 정상 = 정상으로 기록
            debug_info["status"] = "long_window_ok"   # 장기 윈도우 통과로 필터링
            return False, disagree_ratio, debug_info

        # ── 정상→역방향 급전환 가드 ─────────────────────────────────────
        # 방금 전까지 정상 주행하던 차량이 갑자기 역방향으로 바뀌는 경우,
        # CCTV 글자 가림 등 일시 노이즈일 가능성이 높다.
        # 마지막 정상 주행 판정 프레임으로부터 direction_change_guard_frames 이내이면
        # wrong_way_count 증가를 막고 건너뜀.
        last_ok = st.last_correct_frame.get(track_id, 0)        # 마지막 정상 프레임
        frames_since_correct = st.frame_num - last_ok            # 정상 판정 후 경과 프레임
        if frames_since_correct <= cfg.direction_change_guard_frames:
            st.wrong_way_count[track_id] = 0              # 가드 발동 시 카운트 완전 리셋
            # (가드 만료 후 잠깐 누적된 카운트도 초기화 → 장시간 텍스트 겹침 방지)
            debug_info["status"] = "change_guard_filtered"       # 가드로 필터링됨
            return False, disagree_ratio, debug_info

        # ── 의심 카운트 증가 (단기·장기 모두 역방향 통과) ───────────────
        if track_id not in st.first_suspect_frame:     # 첫 의심 시작 기록
            st.first_suspect_frame[track_id] = st.frame_num
            print(f"   ⚠️ ID:{track_id} 역주행 의심 시작 "
                  f"(frame={st.frame_num}, "
                  f"첫등장={st.first_seen_frame.get(track_id, '?')})")

        st.wrong_way_count[track_id] += 1             # 의심 카운트 +1

        # ── 의심 횟수 임계값 도달 → 확정 전 다단계 검증 ─────────────────
        if st.wrong_way_count[track_id] >= cfg.wrong_count_threshold:

            # ── ★ 방향 급변 필터 (사용자 요청) ─────────────────────────
            # 정상 주행 중인 차량이 갑자기 역방향으로 바뀌면 CCTV 글자/오클루전 가능.
            # last_correct_frame이 기록되어 있고(한 번이라도 정상으로 판정된 적 있고)
            # 의심 시작(first_suspect_frame)이 마지막 정상 프레임으로부터
            # direction_change_guard_frames 이내이면 → 급변으로 판정 → 확정 거부.
            # 실제 역주행 차량은 처음부터 역방향이므로 last_correct_frame=0 → 이 검사 통과.
            lcf = st.last_correct_frame.get(track_id, 0)       # 마지막 정상 프레임
            fsf = st.first_suspect_frame.get(track_id, 0)      # 첫 의심 프레임
            if lcf > 0 and (fsf - lcf) <= cfg.direction_change_guard_frames:
                st.wrong_way_count[track_id] = 0               # 카운트 완전 리셋
                debug_info["status"] = "sudden_change_rejected" # 급변 거부
                return False, disagree_ratio, debug_info

            # ── ① 전체 궤적 방향 최종 검증 ──────────────────────────────
            # 저장된 전체 궤적(traj[0]→traj[-1])의 시작→끝 방향도 역방향이어야 최종 확정.
            # 의심 카운트가 쌓이는 도중 전체 이동은 사실 정방향인 경우(합류·차선변경)를 걸러냄.
            global_ok = True                           # 기본값: 패스 (궤적 짧으면 생략)

            if len(traj) >= cfg.velocity_window:       # 단기 윈도우 이상 궤적 있을 때
                gvdx = traj[-1][0] - traj[0][0]       # 전체 궤적 시작→끝 x 이동
                gvdy = traj[-1][1] - traj[0][1]       # 전체 궤적 시작→끝 y 이동
                gmag = np.sqrt(gvdx**2 + gvdy**2)     # 전체 이동 크기

                if gmag > cfg.min_move_distance:       # 전체 이동이 충분할 때만
                    gndx = gvdx / gmag                 # 전체 단위 x 방향
                    gndy = gvdy / gmag                 # 전체 단위 y 방향

                    # flow를 여러 위치에서 시도 (현재→시작→중간 순서)
                    # 현재 위치가 eroded/빈 셀이면 다른 위치에서 확인
                    flow_v_global = None                # flow 벡터 초기값
                    for try_idx in [-1, 0, len(traj) // 2]:  # 현재, 시작, 중간
                        try_x, try_y = traj[try_idx]   # 시도할 좌표
                        fv = self.flow.get_interpolated(try_x, try_y)
                        if fv is not None:              # 유효한 flow 발견
                            flow_v_global = fv          # 사용
                            break

                    if flow_v_global is not None:      # flow 유효하면
                        global_cos = float(            # 전체 궤적 방향 코사인
                            gndx * flow_v_global[0] + gndy * flow_v_global[1]
                        )
                        debug_info["global_cos"] = round(global_cos, 4)  # 디버그 기록
                        global_ok = (global_cos < cfg.cos_threshold)  # 역방향이어야 확정 (전체 궤적은 원본 threshold)

            if global_ok:                              # 전체 궤적도 역방향 → 최종 확정
                st.wrong_way_ids.add(track_id)         # 역주행 차량으로 등록
                debug_info["status"] = "CONFIRMED"
                return True, disagree_ratio, debug_info
            else:                                      # 전체 궤적이 정상 방향 → 확정 취소
                st.wrong_way_count[track_id] = max(
                    0, st.wrong_way_count[track_id] - 2
                )
                st.last_correct_frame[track_id] = st.frame_num  # 전체 궤적 정상 = 정상으로 기록
                debug_info["status"] = "global_traj_ok"  # 전체 궤적 검증으로 필터링
                return False, disagree_ratio, debug_info

        return False, disagree_ratio, debug_info       # 아직 의심 횟수 부족
