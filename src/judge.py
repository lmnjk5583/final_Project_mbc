# 파일 경로: C:\final_pj\src\judge.py
# 역할: 흐름장과 코사인 유사도 기반 역주행 판별
#        원근 기반 속도 게이트 + 다중 포인트 투표 + 시간적 히스테리시스(카운팅)
#        ③ 다중 스케일 윈도우: 단기(velocity_window) + 장기(×2) 모두 역방향이어야 의심
#        ① 전체 궤적 방향 검증: 확정 시 traj[0]→traj[-1] 전체 방향도 역방향이어야 확정
#        개선 3: smoothed_mask 셀에서 cos_threshold 완화 (보간 셀 오탐 방지)
#        [수정] 역주행 미확정 문제 해결: 과도한 가드 완화 + 확정 경로 단순화

import numpy as np


class WrongWayJudge:
    def __init__(self, cfg, flow_map, state):
        self.cfg = cfg
        self.flow = flow_map
        self.st = state

    def get_speed_threshold(self, cy):
        """(레거시) 화면 y 위치 기반 raw 속도 임계값 — 로그 표시 전용."""
        ratio = cy / self.st.frame_h
        scale = 0.3 + 0.7 * ratio
        return self.cfg.base_speed_threshold * scale

    def _get_cos_threshold(self, px, py, level="short"):
        """위치 기반 cos_threshold 반환."""
        if level == "global":
            return self.cfg.cos_threshold

        r, c = self.flow.get_cell_rc(px, py)
        if self.flow.is_smoothed(r, c):
            if level == "long":
                return -0.60
            return -0.50
        return self.cfg.cos_threshold

    def _is_against_flow(self, ndx, ndy, px, py):
        """(ndx, ndy) 방향이 (px, py) 위치의 flow 벡터와 역방향인지 반환."""
        flow_v = self.flow.get_interpolated(px, py)
        if flow_v is None:
            return False
        cos = float(ndx * flow_v[0] + ndy * flow_v[1])
        threshold = self._get_cos_threshold(px, py, level="short")
        return cos < threshold

    # ── 핵심 헬퍼: 전체 궤적 방향 검증 ────────────────────────────────
    def _verify_global_trajectory(self, traj, track_dir, debug_info):
        """traj[0]→traj[-1] 전체 방향이 flow와 역방향인지 검증.

        Returns:
            (ok: bool, gndx: float, gndy: float)
            ok=True → 역방향 확정 가능
            ok=False → 정방향 또는 flow 없음
        """
        cfg = self.cfg

        if len(traj) < cfg.velocity_window:
            return False, 0.0, 0.0

        gvdx = traj[-1][0] - traj[0][0]
        gvdy = traj[-1][1] - traj[0][1]
        gmag = np.sqrt(gvdx ** 2 + gvdy ** 2)

        # 이동 부족 → 단기 투표에 위임 (정지 차량 예외)
        if gmag <= cfg.min_move_distance:
            return True, 0.0, 0.0   # 이동 없으면 global 검증 면제

        gndx = gvdx / gmag
        gndy = gvdy / gmag

        # ── traj_ref_cos: 최근 윈도우 방향으로 계산 (기록용만, 조기 차단 안 함) ──
        # flow map 비교가 훨씬 신뢰성 높음.
        # traj_vs_ref_normal 조기 차단은 flow 루프 전에 실행 시 실제 역주행 증거를
        # 확인하기도 전에 차단해버리는 문제 발생 → flow 없을 때 최후 fallback으로만 사용.
        if self.flow._ref_dx is not None:
            v_win = min(len(traj), cfg.velocity_window)
            rvdx = traj[-1][0] - traj[-v_win][0]
            rvdy = traj[-1][1] - traj[-v_win][1]
            rmag = np.sqrt(rvdx ** 2 + rvdy ** 2)
            if rmag > cfg.min_move_distance:
                rndx, rndy = rvdx / rmag, rvdy / rmag
            else:
                rndx, rndy = gndx, gndy
            traj_ref_cos = float(rndx * self.flow._ref_dx + rndy * self.flow._ref_dy)
            debug_info["traj_ref_cos"] = round(traj_ref_cos, 4)

        # ── flow map으로 역방향 확인 ──────────────────────────────────────
        # 고정 3개 위치 → 중앙선처럼 빈 구역 통과 시 모두 None 가능
        # → 전체 궤적에서 스텝 단위로 최대 8개 위치 시도 (커버리지 확장)
        #
        # get_interpolated(direction=track_dir)는 내부적으로 오염-인식 fallback 수행:
        #   ① 채널 데이터 있음 → 채널 벡터 반환
        #   ② 채널 없음 + 글로벌이 같은 방향 → 글로벌 반환
        #   ③ 채널 없음 + 글로벌이 반대 방향 → None (오염 방지)
        # track_dir='a' 전용 추가 처리: ③에서 None 반환 시
        #   = 'a' 채널 없음 + 글로벌 flow가 'b' 방향 = 이 위치의 정상 흐름은 'b'
        #   = 'a' 차량이 'b' 영역을 지나고 있음 → 역주행의 직접 증거
        #   → 글로벌 'b' 벡터를 역주행 판단에 그대로 사용
        # track_dir='b': direction=None 재조회 금지
        #   → 학습 중 빈 채널에서 정상 'b' 차량도 cos≈-1.0으로 오탐 유발
        _n_traj = len(traj)
        _step = max(1, _n_traj // 8)
        _checked_cells = set()

        for _k in range(0, _n_traj, _step):
            try_x, try_y = traj[-(_k + 1)]
            _ck = (int(try_x / max(self.flow.cell_w, 1.0)),
                   int(try_y / max(self.flow.cell_h, 1.0)))
            if _ck in _checked_cells:
                continue
            _checked_cells.add(_ck)

            fv = self.flow.get_interpolated(try_x, try_y, direction=track_dir)
            if fv is None and track_dir == 'a' and self.flow._ref_dx is not None:
                # ③ 감지: 'a' 채널 없음 + 글로벌이 'b' 방향인지 확인
                _gv = self.flow.get_interpolated(try_x, try_y, direction=None)
                if _gv is not None:
                    _cos_g = float(_gv[0] * self.flow._ref_dx
                                   + _gv[1] * self.flow._ref_dy)
                    if _cos_g < 0:   # 글로벌 flow가 'b' 방향 → 역주행 증거
                        fv = _gv
            if fv is None:
                continue

            global_cos = float(gndx * fv[0] + gndy * fv[1])
            debug_info["global_cos"] = round(global_cos, 4)

            threshold = self._get_cos_threshold(try_x, try_y, level="global")
            if global_cos < threshold:
                return True, gndx, gndy  # 역방향 확인
            else:
                return False, gndx, gndy  # 정방향 확인

        # ── 모든 위치 flow 없음 — ref_dx 최후 fallback ───────────────────
        if self.flow._ref_dx is not None:
            trc = debug_info.get("traj_ref_cos", None)
            if trc is not None:
                if track_dir != 'b':
                    # trc > 0.85: ref_dx와 거의 정방향(18° 이내) → 정상으로 판단
                    # 0.3 기준은 너무 좁음: 차선 방향이 ref_dx와 다를 때 역주행 차량도 차단
                    if trc > 0.85:
                        debug_info["status"] = "traj_vs_ref_normal"
                        return False, gndx, gndy
                    elif trc <= 0.0:
                        # ref와 반대 방향 → 역주행 확정
                        return True, gndx, gndy
                    # 0.0 < trc <= 0.85: 애매 → 확정 불가 (flow 증거 없으면 보수적)
                elif track_dir == 'b':
                    # 'b' 채널에 데이터가 전혀 없으면 one-way 도로 가능성
                    b_has_data = bool(np.any(self.flow.count_b > 0))
                    if not b_has_data and trc <= -0.7:
                        debug_info["status"] = "oneway_ref_confirm"
                        return True, gndx, gndy

        return False, gndx, gndy   # 확정 불가

    def check(self, track_id, traj, ndx, ndy, speed, cy, bbox_h: float = 30.0,
              track_dir=None):
        """한 차량에 대해 flow_map과 방향 비교, 투표 방식으로 역주행 여부 판정."""
        cfg = self.cfg
        st = self.st

        # ── 이미 확정된 차량은 즉시 True ────────────────────────────────
        if track_id in st.wrong_way_ids:
            return True, 1.0, {"status": "CONFIRMED", "cos_values": []}

        # ── 최소 추적 나이 체크 ──────────────────────────────────────────
        _min_age = getattr(cfg, "min_wrongway_track_age", 30)
        _track_age = st.frame_num - st.first_seen_frame.get(track_id, st.frame_num)

        if _track_age < _min_age:
            _bh_c = max(bbox_h, cfg.min_bbox_h)
            _nm = speed / _bh_c
            if _nm >= cfg.norm_speed_gate_threshold and traj:
                _fv = self.flow.get_interpolated(
                    traj[-1][0], traj[-1][1], direction=track_dir)
                if _fv is not None:
                    _cos = float(ndx * _fv[0] + ndy * _fv[1])
                    if _cos >= cfg.cos_threshold:
                        st.last_correct_frame[track_id] = st.frame_num
            return False, 0, {"status": "too_young", "cos_values": []}

        # ── nm 기반 속도 게이트 ──────────────────────────────────────────
        _bh_clamped = max(bbox_h, cfg.min_bbox_h)
        nm_speed = speed / _bh_clamped

        if nm_speed < cfg.norm_speed_gate_threshold:
            # 서행 중 정방향이면 lcf 갱신 (가속 후 fast-track 오탐 방지)
            if (ndx != 0.0 or ndy != 0.0) and traj:
                _fv_slow = self.flow.get_interpolated(
                    traj[-1][0], traj[-1][1], direction=track_dir)
                if _fv_slow is not None:
                    _slow_cos = float(ndx * _fv_slow[0] + ndy * _fv_slow[1])
                    if _slow_cos >= cfg.cos_threshold:
                        st.last_correct_frame[track_id] = st.frame_num
            return False, 0, {"status": "slow", "cos_values": []}

        # ── 방향 급변 필터 ───────────────────────────────────────────────
        prev_vel = st.last_velocity.get(track_id)
        if prev_vel is not None:
            cos_dir = float(ndx * prev_vel[0] + ndy * prev_vel[1])
            if cos_dir < -0.5:
                st.last_velocity[track_id] = (ndx, ndy)
                st.last_correct_frame[track_id] = st.frame_num
                st.direction_change_frame[track_id] = st.frame_num
                st.wrong_way_count[track_id] = 0
                return False, 0, {"status": "dir_jump_filtered", "cos_values": []}
        st.last_velocity[track_id] = (ndx, ndy)

        # ── 안정 방향 대비 급변 감지 (edge detection) ────────────────────
        _stable = st.stable_velocity.get(track_id)
        if _stable is not None:
            _cos_vs_stable = float(ndx * _stable[0] + ndy * _stable[1])
            _was_stable = st.direction_was_stable.get(track_id, True)
            _is_stable_now = (_cos_vs_stable >= cfg.direction_change_cos_threshold)
            if _was_stable and not _is_stable_now:
                st.direction_change_frame[track_id] = st.frame_num
            st.direction_was_stable[track_id] = _is_stable_now

        # ── 방향 급변 가드 early-exit ────────────────────────────────────
        _last_chg = st.direction_change_frame.get(track_id, 0)
        if (_last_chg > 0
                and (st.frame_num - _last_chg) <= cfg.direction_change_guard_frames):
            st.wrong_way_count[track_id] = 0
            return False, 0, {"status": "direction_change_guard", "cos_values": []}

        # ── 단기 투표: 궤적 포인트 샘플링 ──────────────────────────────
        n_points = min(len(traj), 8)
        step = max(1, len(traj) // n_points)

        agree = 0
        disagree = 0
        skip = 0
        debug_points = []
        cos_values = []

        for idx in range(0, len(traj), step):
            px, py = traj[idx]
            flow_v = self.flow.get_interpolated(px, py, direction=track_dir)

            if flow_v is None:
                # ── 1차 fallback: 글로벌 flow map(방향 필터 없음)으로 재시도 ──
                # track_dir 채널에 데이터가 없을 때 글로벌 맵 활용.
                # 효과:
                #   one-way 역주행 차량: 자기 차선의 글로벌 flow(정방향) 대비 반대 → disagree ✓
                #   two-way 정상 'b' 차량: 'b' 차선의 글로벌 flow(b방향) 대비 일치 → agree ✓
                # ref_dx 직접 비교는 one-way/two-way 구분 불가 → 사용하지 않음
                global_fv = self.flow.get_interpolated(px, py, direction=None)
                if global_fv is not None:
                    cos_global = float(ndx * global_fv[0] + ndy * global_fv[1])
                    cos_values.append(cos_global)
                    pt_threshold = self._get_cos_threshold(px, py, level="short")
                    if cos_global < pt_threshold:
                        disagree += 1
                        debug_points.append((px, py, cos_global, "disagree_global"))
                    else:
                        agree += 1
                        debug_points.append((px, py, cos_global, "agree_global"))
                    continue
                # ── 2차 fallback: 글로벌 flow도 없음 → skip ──────────────────
                # ref_dx 기반 추정은 one-way/two-way 구분 불가로 제거
                skip += 1
                debug_points.append((px, py, 0, "skip"))
                continue

            cos_sim = ndx * flow_v[0] + ndy * flow_v[1]
            cos_values.append(cos_sim)

            pt_threshold = self._get_cos_threshold(px, py, level="short")
            if cos_sim < pt_threshold:
                disagree += 1
                debug_points.append((px, py, cos_sim, "disagree"))
            else:
                agree += 1
                debug_points.append((px, py, cos_sim, "agree"))

        total_checked = agree + disagree

        debug_info = {
            "agree": agree,
            "disagree": disagree,
            "skip": skip,
            "total": total_checked,
            "points": debug_points,
            "threshold": nm_speed,
            "status": "voting",
            "cos_values": cos_values,
            "long_cos": None,
            "global_cos": None,
        }

        if total_checked < 3:
            return False, 0, debug_info

        disagree_ratio = disagree / total_checked

        # ── 단기 투표 통과 여부 ──────────────────────────────────────────
        if disagree_ratio < cfg.vote_threshold:
            st.wrong_way_count[track_id] = max(
                0, st.wrong_way_count[track_id] - 2)
            st.last_correct_frame[track_id] = st.frame_num
            st.stable_velocity[track_id] = (ndx, ndy)
            st.direction_was_stable[track_id] = True
            return False, disagree_ratio, debug_info

        # ── ③ 장기 윈도우 검사 ──────────────────────────────────────────
        long_window = cfg.velocity_window * 2
        long_suspect = True   # 궤적 부족 시 면제

        if len(traj) >= long_window:
            _lw = long_window
            _lsi = len(traj) - _lw
            _lpfx = [traj[_lsi + i + 1][0] - traj[_lsi + i][0]
                     for i in range(_lw - 1)]
            _lpfy = [traj[_lsi + i + 1][1] - traj[_lsi + i][1]
                     for i in range(_lw - 1)]
            lvdx = float(np.median(_lpfx)) * (_lw - 1)
            lvdy = float(np.median(_lpfy)) * (_lw - 1)
            lmag = np.sqrt(lvdx ** 2 + lvdy ** 2)
            avg_lmove = lmag / _lw

            if (lmag > cfg.min_move_distance
                    and avg_lmove > cfg.min_move_per_frame):
                lndx = lvdx / lmag
                lndy = lvdy / lmag
                cx_last, cy_last = traj[-1]
                flow_v_long = self.flow.get_interpolated(
                    cx_last, cy_last, direction=track_dir)

                if flow_v_long is not None:
                    long_cos = float(
                        lndx * flow_v_long[0] + lndy * flow_v_long[1])
                    debug_info["long_cos"] = round(long_cos, 4)
                    long_threshold = self._get_cos_threshold(
                        cx_last, cy_last, level="long")
                    long_suspect = (long_cos < long_threshold)
                # flow 없으면 long_suspect=True (면제)

        if not long_suspect:
            st.wrong_way_count[track_id] = max(
                0, st.wrong_way_count[track_id] - 2)
            st.last_correct_frame[track_id] = st.frame_num
            st.stable_velocity[track_id] = (ndx, ndy)
            st.direction_was_stable[track_id] = True
            debug_info["status"] = "long_window_ok"
            return False, disagree_ratio, debug_info

        # ════════════════════════════════════════════════════════════════
        # 여기까지 왔으면: 단기 + 장기 모두 역방향 → 의심 확정 경쟁
        # ════════════════════════════════════════════════════════════════

        _cur_age = st.frame_num - st.first_seen_frame.get(track_id, st.frame_num)
        _ft_min_age = getattr(cfg, "fast_confirm_min_age", 45)
        _age_gate_end_ft = (st.first_seen_frame.get(track_id, 0)
                            + _min_age + cfg.velocity_window)
        _lcf_ft = st.last_correct_frame.get(track_id, 0)

        # ── 재연결 가드 ──────────────────────────────────────────────────
        _post_rec = getattr(st, "post_reconnect_frame", 0)
        _reconnect_guard = (_post_rec > 0
                            and (st.frame_num - _post_rec)
                            <= cfg.direction_change_guard_frames)

        # ── fast-track 조건 ──────────────────────────────────────────────
        # [수정] post_slow_guard 제거 — lcf 갱신 수정으로 이미 보호됨
        #         fast-track 최소 나이를 velocity_window*2로 완화 (45f → 30f)
        #         _lcf 조건: age_gate_end 이후 정방향 없었으면 충분
        _ft_age_ok = _cur_age >= max(_ft_min_age,
                                     cfg.velocity_window * 2)
        _ft_lcf_ok = _lcf_ft <= _age_gate_end_ft

        if (disagree_ratio >= cfg.fast_confirm_ratio
                and nm_speed >= cfg.fast_confirm_speed
                and _ft_lcf_ok
                and _ft_age_ok
                and not _reconnect_guard):

            debug_info_ft = dict(debug_info)
            global_ok_ft, gndx_ft, gndy_ft = self._verify_global_trajectory(
                traj, track_dir, debug_info_ft)

            if global_ok_ft:
                st.wrong_way_ids.add(track_id)
                debug_info_ft["status"] = "FAST_CONFIRMED"
                print(f"   🚨 ID:{track_id} 역주행 즉시 확정 "
                      f"(fast-track, frame={st.frame_num}, "
                      f"disagree={disagree_ratio:.2f}, nm={nm_speed:.2f}, "
                      f"global_cos={debug_info_ft.get('global_cos')}, "
                      f"traj_ref_cos={debug_info_ft.get('traj_ref_cos')})")
                return True, disagree_ratio, debug_info_ft
            else:
                # fast-track 실패 → 10프레임 간격으로만 출력 (로그 과다 방지)
                if st.wrong_way_count[track_id] % 10 == 0:
                    print(f"   ⚡ ID:{track_id} fast-track 실패 "
                          f"(global_ok=False, status={debug_info_ft.get('status')}, "
                          f"global_cos={debug_info_ft.get('global_cos')}, "
                          f"traj_ref_cos={debug_info_ft.get('traj_ref_cos')})")
                debug_info.update(debug_info_ft)

        # ── 의심 카운트 증가 ─────────────────────────────────────────────
        if track_id not in st.first_suspect_frame:
            st.first_suspect_frame[track_id] = st.frame_num
            print(f"   ⚠️ ID:{track_id} 역주행 의심 시작 "
                  f"(frame={st.frame_num}, "
                  f"첫등장={st.first_seen_frame.get(track_id, '?')}, "
                  f"disagree={disagree_ratio:.2f}, nm={nm_speed:.2f})")

        st.wrong_way_count[track_id] += 1

        # ── 매 프레임 wrong_count 상태 출력 (미확정 추적용) ─────────────
        if st.wrong_way_count[track_id] % 10 == 0:
            print(f"   🔍 ID:{track_id} wrong_count={st.wrong_way_count[track_id]}"
                  f"/{cfg.wrong_count_threshold} "
                  f"(disagree={disagree_ratio:.2f}, nm={nm_speed:.2f}, "
                  f"long_cos={debug_info.get('long_cos')}, "
                  f"lcf={_lcf_ft}, age_gate_end={_age_gate_end_ft})")

        # ── wrong_count_threshold 도달 → 다단계 확정 검증 ───────────────
        if st.wrong_way_count[track_id] < cfg.wrong_count_threshold:
            return False, disagree_ratio, debug_info

        # ① normal-path 최소 나이 가드
        _normal_min_age = getattr(cfg, "fast_confirm_min_age", 45)
        if _cur_age < _normal_min_age:
            debug_info["status"] = "normal_path_too_young"
            return False, disagree_ratio, debug_info

        # ② 방향 급변 필터 (sudden_change_rejected)
        # 의도: 차량이 정방향으로 달리다(lcf 최근) 갑자기 역방향 의심(fsf ≈ lcf 이후)이면
        #       ID 오염·순간 추적 오류로 판단해 차단.
        # 조건: fsf > lcf (의심 시작이 마지막 정방향 이후여야 함)
        #       + (fsf - lcf) 간격이 guard_frames 이내 (급변 판정)
        # 버그 수정: 이전에는 fsf < lcf 일 때도 (fsf-lcf)가 음수 → 항상 <= guard → 영구차단
        #   → global_ok=False 후 lcf=현재프레임으로 갱신되면 fsf(=61) < lcf(=95) 가 되어
        #     이후 normal-path가 영구적으로 sudden_change_rejected 에 걸리는 루프 발생
        lcf = st.last_correct_frame.get(track_id, 0)
        fsf = st.first_suspect_frame.get(track_id, 0)
        _scr_boundary = (st.first_seen_frame.get(track_id, 0)
                         + _min_age + cfg.velocity_window)
        if (lcf > _scr_boundary
                and fsf > lcf                                # 반드시 정방향(lcf) 이후에 의심 시작
                and (fsf - lcf) <= cfg.direction_change_guard_frames):
            st.wrong_way_count[track_id] = 0
            debug_info["status"] = "sudden_change_rejected"
            return False, disagree_ratio, debug_info

        # ③ 전체 궤적 방향 최종 검증
        global_ok, gndx, gndy = self._verify_global_trajectory(
            traj, track_dir, debug_info)

        # ── 진단 출력 ────────────────────────────────────────────────────
        _ref_status = (f"ref_dx={self.flow._ref_dx:.3f}"
                       if self.flow._ref_dx is not None else "ref_dx=None")
        _cur_fv_dir = self.flow.get_interpolated(
            traj[-1][0], traj[-1][1], direction=track_dir)
        _cur_fv_str = (f"({_cur_fv_dir[0]:.3f},{_cur_fv_dir[1]:.3f})"
                       if _cur_fv_dir is not None else "None")
        # normal-path 반복 출력 억제 (10프레임 간격 또는 확정/실패 시)
        if global_ok or st.wrong_way_count[track_id] % 10 == 0:
            print(f"   🔎 [normal] ID:{track_id} "
                  f"track_dir={track_dir} {_ref_status} "
                  f"vel=({ndx:.3f},{ndy:.3f}) "
                  f"flow_at_pos={_cur_fv_str} "
                  f"global_cos={debug_info.get('global_cos', '?')} "
                  f"traj_ref_cos={debug_info.get('traj_ref_cos', '?')} "
                  f"global_ok={global_ok} "
                  f"disagree={disagree_ratio:.2f}")

        if global_ok:
            st.wrong_way_ids.add(track_id)
            debug_info["status"] = "CONFIRMED"
            print(f"   🚨 ID:{track_id} 역주행 확정! "
                  f"(normal-path, frame={st.frame_num})")
            return True, disagree_ratio, debug_info
        else:
            # flow 데이터가 전혀 없어서 검증 불가한 경우(global_cos 미설정) →
            # 잘못된 감소로 bounce loop 방지: wrong_count 유지
            if debug_info.get("global_cos") is not None:
                st.wrong_way_count[track_id] = max(
                    0, st.wrong_way_count[track_id] - 2)
            # lcf는 갱신하지 않음: global_traj 검증 실패는 투표 disagree=1.0 상태에서도
            # 발생할 수 있음 (flow map 공백·궤적 방향 이슈). 이 상황에서 lcf를 현재프레임으로
            # 갱신하면 fast-track(_ft_lcf_ok) 영구 차단 + sudden_change_rejected 루프 유발.
            debug_info["status"] = "global_traj_ok"
            return False, disagree_ratio, debug_info

        return False, disagree_ratio, debug_info