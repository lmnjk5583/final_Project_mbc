# 역주행 라벨(W1, W2) 관리 + ID 재매칭(occlusion 후 재등장) + 오래된 트랙 정리


import numpy as np


class IDManager:
    def __init__(self, cfg, flow_map, state):
        self.cfg = cfg
        self.flow = flow_map
        self.st = state

    # ==================== 라벨 조회 ====================
    def get_display_label(self, track_id):
        """표시용 역주행 라벨(W1, W2 등) 조회 (ID가 바뀌어도 고정)"""
        return self.st.display_id_map.get(track_id)

    # ==================== 라벨 부여/승계 ====================
    def assign_label(self, track_id, matched_from=None):
        """역주행 확정 시 표시 라벨 부여 또는 기존 라벨 승계"""
        st = self.st

        if track_id in st.display_id_map:
            return  # 이미 라벨 있음

        # 기존 역주행 차량으로부터 이어받는 경우
        if matched_from and matched_from in st.display_id_map:
            old_label = st.display_id_map[matched_from]   # 기존 라벨
            st.display_id_map[track_id] = old_label       # 새 ID에 같은 라벨 부여
            print(f"🔄 라벨 이어받기: ID:{matched_from}({old_label}) → ID:{track_id}({old_label})")
        else:
            # 새로 발견된 역주행 차량이면 W1, W2 ... 순서대로 부여
            label = f"W{st.next_wrong_way_label}"
            st.display_id_map[track_id] = label           # 라벨 등록
            st.next_wrong_way_label += 1                   # 다음 번호 증가

            first_frame = st.first_seen_frame.get(track_id, st.frame_num)
            suspect_frame = st.first_suspect_frame.get(track_id, st.frame_num)

            frames_from_appear = st.frame_num - first_frame      # 등장→확정 프레임 수
            frames_from_suspect = st.frame_num - suspect_frame   # 의심→확정 프레임 수

            seconds_from_appear = frames_from_appear / st.video_fps    # 등장→확정 초
            seconds_from_suspect = frames_from_suspect / st.video_fps  # 의심→확정 초

            st.detection_stats[label] = {
                "track_id": track_id,
                "first_frame": first_frame,
                "suspect_frame": suspect_frame,
                "detect_frame": st.frame_num,
                "frames_from_appear": frames_from_appear,
                "frames_from_suspect": frames_from_suspect,
                "seconds_from_appear": round(seconds_from_appear, 2),
                "seconds_from_suspect": round(seconds_from_suspect, 2),
            }

            print(f"🚨 역주행 확정: ID:{track_id} → {label}")
            print(f"   등장→확정: {frames_from_appear}프레임 ({seconds_from_appear:.2f}초)")
            print(f"   의심→확정: {frames_from_suspect}프레임 ({seconds_from_suspect:.2f}초)")

    # ==================== ID 재매칭 ====================
    def check_reappear(self, track_id, cx, cy):
        """역주행 차량이 사라졌다가 새 ID로 다시 나타난 경우, 위치/방향으로 재매칭"""
        st = self.st
        cfg = self.cfg

        # 이미 이 ID 자체가 역주행 확정이면 더 볼 필요 없음
        if track_id in st.wrong_way_ids:
            return True

        # 현재 ID의 궤적 가져오기
        traj = st.trajectories.get(track_id, [])
        if len(traj) < 5:
            # 궤적이 너무 짧으면 이동 방향을 신뢰하기 어려움
            return False

        # 현재 ID의 '전체 이동 방향' 계산 (처음 위치 → 현재 위치)
        vdx = traj[-1][0] - traj[0][0]        # [-1]은 현재 위치(리스트의 마지막 요소)
        vdy = traj[-1][1] - traj[0][1]
        v_mag = np.sqrt(vdx ** 2 + vdy ** 2)
        if v_mag < 3:
            # 거의 안 움직이면 그냥 패스 (정체/대기 상태일 수 있음)
            return False

        ndx, ndy = vdx / v_mag, vdy / v_mag   # 단위 방향 벡터

        # 과거에 사라졌던 '역주행 차량들'의 마지막 위치들과 비교
        for old_id, (ox, oy, old_frame) in list(st.wrong_way_last_pos.items()):
            if old_id == track_id:
                continue  # 자기 자신과는 비교할 필요 없음

            # (1) 사라진 지 N프레임 이내인 경우만 후보로 봄
            if (st.frame_num - old_frame) >= cfg.reappear_frame_limit:
                continue

            # (2) 과거 위치(ox, oy)와 현재 위치(cx, cy)가 충분히 가까운지 확인
            dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            if dist >= cfg.id_match_distance:
                continue  # 일정 거리 이상이면 다른 차

            # (3) 여전히 흐름장과 반대 방향인지 확인 (정상 차량과 구분)
            flow_v = self.flow.get_interpolated(cx, cy)
            if flow_v is None:
                continue  # flow 정보가 없으면 방향 검증 불가 → 재매칭 포기
            cos_sim = ndx * flow_v[0] + ndy * flow_v[1]
            if cos_sim > cfg.cos_threshold:
                # 흐름과 비슷한 방향이면 '정상'일 가능성이 높으므로 매칭하지 않음
                continue

            # (4) 새 ID 자체에 의심 이력(wrong_way_count >= 1)이 있어야 재매칭 허용
            # 자체 의심 없이 위치만 가까운 경우 연쇄 오탐 방지
            if st.wrong_way_count.get(track_id, 0) < 1:
                continue

            # 위 조건들을 모두 통과 → 과거 역주행 차량(old_id)이 새 ID(track_id)로 다시 잡힌 것으로 판단
            # wrong_way_count를 threshold-1로 설정: judge.py가 다음 프레임에서 최종 확정
            # (즉시 wrong_way_ids에 추가하지 않아 연쇄 확정 방지)
            st.wrong_way_count[track_id] = cfg.wrong_count_threshold - 1
            self.assign_label(track_id, matched_from=old_id)        # 기존 라벨(W1 등) 승계 예약
            del st.wrong_way_last_pos[old_id]                       # 이전 기록 삭제
            return True

        return False

    # ==================== 오래된 트랙 정리 ====================
    def cleanup(self, active_ids):
        """오랫동안 보이지 않는 차량 트랙/카운트/위치 기록 정리"""
        st = self.st
        cfg = self.cfg

        for tid in list(st.trajectories.keys()):
            if tid not in active_ids:                   # 이번 프레임에 감지되지 않은 ID라면
                st._stale_counter[tid] += 1             # 안 보인 프레임 수 +1

                # 역주행 차량이 '처음' 사라지는 시점에 마지막 위치를 한 번만 기록
                if tid in st.wrong_way_ids and st._stale_counter[tid] == 1:
                    traj = st.trajectories[tid]
                    if traj:
                        # (x, y, 사라진 프레임 번호)
                        st.wrong_way_last_pos[tid] = (
                            traj[-1][0], traj[-1][1], st.frame_num
                        )

                # N프레임 이상 계속 안 보이면 그 ID 관련 정보는 메모리에서 완전히 제거
                if st._stale_counter[tid] > cfg.stale_threshold:
                    del st.trajectories[tid]                  # 궤적 삭제
                    st.wrong_way_count.pop(tid, None)         # 역주행 카운트 삭제
                    st._stale_counter.pop(tid, None)          # 스테일 카운터 삭제
                    st.wrong_way_ids.discard(tid)             # 역주행 확정 집합에서도 제거
                    st.last_velocity.pop(tid, None)           # 방향 벡터 삭제
                    st.last_correct_frame.pop(tid, None)      # 정상 판정 프레임 삭제
                    st.stable_velocity.pop(tid, None)         # 안정 방향 기준점 삭제
                    st.direction_change_frame.pop(tid, None)  # 급변 감지 시점 삭제
                    st.direction_was_stable.pop(tid, None)    # 안정 상태 플래그 삭제
                    # (ByteTrack이 동일 ID를 새 차량에 재할당해도 역주행으로 오탐 방지)
            else:
                # 이번 프레임에 다시 나타난 ID면 카운터 삭제
                st._stale_counter.pop(tid, None)
                # 역주행 확정 ID가 같은 ID 그대로 재등장하면 last_pos 초기화
                # (last_pos가 남아있으면 다른 ID가 이 위치와 오매칭될 수 있음)
                if tid in st.wrong_way_ids:
                    st.wrong_way_last_pos.pop(tid, None)

        # 역주행 마지막 위치 기록 중, 너무 오래된 것은 삭제
        old = [k for k, (_, _, f) in st.wrong_way_last_pos.items()
               if st.frame_num - f > cfg.last_pos_expire]
        for k in old:
            del st.wrong_way_last_pos[k]