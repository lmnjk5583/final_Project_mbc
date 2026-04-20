# 파일 경로: C:\final_pj\src\flow_map.py
# 15x15 그리드 기반 정상 흐름장
# EMA 학습 + 이중 선형 보간(Bilinear Interpolation) + 방향 일관성 공간 평활화
# smoothed_mask: 보간으로 채워진 셀(실 데이터 없음) 추적 — judge.py에서 cos_threshold 완화에 사용

import numpy as np                                       # 수치 계산
from collections import deque                             # BFS 큐 (flood-fill용)
from pathlib import Path                                  # 경로 조작


class FlowMap:
    def __init__(self, grid_size: int, alpha: float, min_samples: int,
                 bbox_alpha_decay: float = 0.5,
                 bbox_gating_alpha_ratio: float = 0.3,
                 edge_margin: int = 1,
                 max_cross_flow_cells: float = 1.2):
        self.grid_size = grid_size                        # 흐름 맵을 나눌 격자 크기 (N x N)
        self.alpha = alpha                                # EMA 학습 속도 (새 데이터 반영 비율)
        self.min_samples = min_samples                    # 셀당 최소 학습 샘플 수 (이하이면 공간 보정)
        self._bbox_alpha_decay = bbox_alpha_decay         # bbox 거리당 alpha 감쇠율
        self._bbox_gating_ratio = bbox_gating_alpha_ratio # 이 비율 미만 셀 → 방향 게이팅·count 비적용
        self._edge_margin = edge_margin                   # 학습 제외 가장자리 셀 수
        self._max_cross_flow = max_cross_flow_cells       # 횡방향(차선 횡단) 최대 확산 셀 수

        # 각 셀의 정상 이동 방향 벡터 (ndx, ndy) — 학습 단계에서 단일 글로벌 맵으로 학습
        self.flow = np.zeros((grid_size, grid_size, 2), np.float32)
        # 각 셀의 학습 데이터 개수
        self.count = np.zeros((grid_size, grid_size), np.int32)

        # ── 양방향(Dual-Channel) flow map ─────────────────────────────────
        # 학습 완료 후 build_directional_channels()로 글로벌 맵을 A/B로 분리.
        # A = ref_direction과 cos >= 0 방향 차량의 셀
        # B = ref_direction과 cos < 0 방향 차량의 셀
        # 판정 시: 차량의 진행 방향 채널을 우선 조회 → 반대 차선 차량에 의한
        #           중앙선 오염이 구조적으로 불가능 (다른 채널만 오염 가능)
        # 해당 채널에 데이터 없으면 글로벌 맵으로 fallback (호환성 보장)
        self.flow_a  = np.zeros((grid_size, grid_size, 2), np.float32)
        self.count_a = np.zeros((grid_size, grid_size), np.int32)
        self.flow_b  = np.zeros((grid_size, grid_size, 2), np.float32)
        self.count_b = np.zeros((grid_size, grid_size), np.int32)

        self.frame_w = 0                                  # 영상 너비
        self.frame_h = 0                                  # 영상 높이
        self.cell_w = 1.0                                 # 셀 하나의 너비
        self.cell_h = 1.0                                 # 셀 하나의 높이

        # Phase 1 정체 탐지용 — 셀별 정상 normalized_speed EMA
        self.speed_ref = np.zeros((grid_size, grid_size), np.float32)  # 셀별 정상 norm_speed 기준값

        # apply_boundary_erosion() / apply_overlap_erosion()이 제거한 셀 — 재학습 금지
        self.eroded_mask = np.zeros((grid_size, grid_size), dtype=bool)  # True=영구 빈 셀

        # bbox 겹침 추적 — 반대 차선 차량의 bbox가 이 셀을 밟은 횟수
        # learn_step(bbox=...) 호출 시 방향 게이팅으로 EMA 갱신이 거부된 셀에 카운트
        # apply_overlap_erosion()에서 threshold 이상이면 중앙선 경계로 판정 → 제거
        self._bbox_contra_count = np.zeros((grid_size, grid_size), np.int32)

        # ── 개선 1: smoothed_mask — 보간으로 채워진 셀 추적 ──────────────
        # apply_spatial_smoothing()에서 count=0 셀이 이웃 평균으로 채워지면 True
        # learn_step()에서 실 데이터가 들어오면 False로 해제
        # judge.py에서 이 마스크가 True인 셀은 cos_threshold를 완화하여 오탐 방지
        self.smoothed_mask = np.zeros((grid_size, grid_size), dtype=bool)  # True=보간 채움, 실 데이터 없음

        self._learn_call_count = 0                        # learn_step 호출 횟수 (디버그용)

        # ── 양방향 채널 기준 방향 (build_directional_channels 호출 후 설정) ──
        # get_interpolated의 contamination-aware global fallback에 사용:
        #   채널 데이터 없는 셀에서 글로벌 맵 방향이 쿼리 방향과 반대면 None 반환
        #   → 오염된 글로벌 벡터가 판정에 개입하는 것을 구조적으로 차단
        self._ref_dx: float | None = None
        self._ref_dy: float | None = None

    # ==================== 초기화/리셋 ====================
    def init_grid(self, frame_w, frame_h):
        """영상 해상도에 맞게 flow_map 그리드 셀 크기 설정"""
        self.frame_w = frame_w                            # 영상 너비 저장
        self.frame_h = frame_h                            # 영상 높이 저장
        self.cell_w = frame_w / self.grid_size            # 셀 너비 = 영상 너비 / 그리드 수
        self.cell_h = frame_h / self.grid_size            # 셀 높이 = 영상 높이 / 그리드 수

    def reset(self):
        """flow_map과 count를 0으로 초기화"""
        self.flow[:] = 0                                  # 흐름 벡터 초기화
        self.count[:] = 0                                 # 샘플 수 초기화
        self.speed_ref[:] = 0                             # 셀별 정상 속도 기준값 초기화
        self.eroded_mask[:] = False                       # 경계 마스크 초기화 (재학습 시 초기화)
        self.smoothed_mask[:] = False                     # 보간 마스크 초기화 (재학습 시 초기화)
        self._bbox_contra_count[:] = 0                    # bbox 반대방향 방문 카운터 초기화
        self._learn_call_count = 0                        # 호출 카운터 초기화
        self.flow_a[:] = 0                                # A채널 벡터 초기화
        self.count_a[:] = 0                               # A채널 카운터 초기화
        self.flow_b[:] = 0                                # B채널 벡터 초기화
        self.count_b[:] = 0                               # B채널 카운터 초기화
        self._ref_dx = None                               # 기준 방향 리셋
        self._ref_dy = None

    # ==================== 좌표 변환 ====================
    def _cell_coords(self, x, y):
        """픽셀 좌표(x, y)를 그리드 좌표(r, c)로 변환"""
        r = (y / self.cell_h) - 0.5                       # 행 인덱스 실수 값 (0~grid_size-1 부근)
        c = (x / self.cell_w) - 0.5                       # 열 인덱스 실수 값
        return r, c

    # ==================== bbox 셀 목록 헬퍼 ====================
    def _get_bbox_cells(self, bx1, by1, bx2, by2):
        """bbox 영역에 포함되는 모든 그리드 셀 (r, c) 인덱스 목록 반환."""
        gs = self.grid_size
        r_min = int(np.clip(by1 / self.cell_h, 0, gs - 1))
        r_max = int(np.clip(by2 / self.cell_h, 0, gs - 1))
        c_min = int(np.clip(bx1 / self.cell_w, 0, gs - 1))
        c_max = int(np.clip(bx2 / self.cell_w, 0, gs - 1))
        return [(r, c) for r in range(r_min, r_max + 1)
                       for c in range(c_min, c_max + 1)]

    # ==================== EMA 기반 학습 ====================
    def learn_step(self, x1, y1, x2, y2, min_move, bbox=None,
                   traj_ndx=None, traj_ndy=None):
        """한 차량의 이동 벡터를 flow_map에 반영 (EMA 기반 학습).

        Args:
            x1, y1: 이전 위치 (velocity_window 프레임 전 중심점).
            x2, y2: 현재 footpoint (bbox 중심).
            min_move: 최소 이동 거리 (이하이면 무시).
            bbox: (bx1, by1, bx2, by2) — 현재 프레임 bbox 좌표.
                  None이면 기존 중심점 1셀 방식. 지정하면 bbox 전체 셀에 EMA 갱신.
            traj_ndx, traj_ndy: 궤적 전체 방향 벡터 (traj[0] → traj[-1]).
                  지정 시 중앙점(dist=0) 게이팅·갱신에 사용 — velocity_window 노이즈 차단.
                  None이면 velocity_window 방향(ndx, ndy) 사용.

        셀 유형별 동작:
          dist=0 (중앙점): 게이팅 적용 — trajectory 방향 기준 (velocity_window 아님).
                           반대 방향(cos<-0.4) → EMA 거부 + count-=2 (2배 빠른 잠금 해제).
                           게이팅 통과 시에만 EMA 갱신 + count 증가.
                           _bbox_contra_count: 중앙점 기반으로만 추적 (중앙선 경계 검출).
          dist=1 (bbox 인접): 게이팅 적용 (velocity_window 방향), contra 추적 없음.
          dist≥2 (bbox 원거리): 게이팅 없음, count 미증가 (항상 soft).
        """
        dx, dy = x2 - x1, y2 - y1                        # 이동 벡터
        mag = np.sqrt(dx ** 2 + dy ** 2)                  # 크기(속도)
        if mag < min_move:                                # 너무 작은 움직임은 무시
            return

        ndx, ndy = dx / mag, dy / mag                     # 단위 방향 벡터 (velocity_window 기반)
        gs = self.grid_size

        # ── 업데이트 대상 셀 목록 결정 ──────────────────────────────────
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            cells = self._get_bbox_cells(*bbox)            # bbox 전체 셀
            # bbox 중심 셀 — 거리 기반 alpha 감쇠의 기준점
            _cx = (bx1 + bx2) / 2
            _cy = (by1 + by2) / 2
            r_center = int(np.clip(_cy / self.cell_h, 0, gs - 1))
            c_center = int(np.clip(_cx / self.cell_w, 0, gs - 1))
            _log_r, _log_c = r_center, c_center
        else:
            # 기존 방식: 이동 경로 중간 점 1셀
            _r = int((y1 + y2) / 2 / self.cell_h)
            _c = int((x1 + x2) / 2 / self.cell_w)
            _log_r = int(np.clip(_r, 0, gs - 1))
            _log_c = int(np.clip(_c, 0, gs - 1))
            cells = [(_log_r, _log_c)]
            r_center, c_center = _log_r, _log_c

        # ── cross-flow 제한용 방향 벡터 (차선 횡단 방향 확산 차단) ─────────
        # bbox 확산 시 이동 방향에 수직인 방향(=차선 횡단)으로의 확산을
        # max_cross_flow_cells 이내로 제한. 이동 방향과 평행한 방향(=차선 내
        # 전후방)은 제한 없음. 중앙선 annotation 없이 구조적 오염 차단.
        _cf_dx = traj_ndx if traj_ndx is not None else ndx  # cross-flow 계산용 방향
        _cf_dy = traj_ndy if traj_ndy is not None else ndy

        # ── 셀별 EMA 갱신 ────────────────────────────────────────────────
        for r, c in cells:
            # erosion으로 제거된 셀은 영구 재학습 금지
            if self.eroded_mask[r, c]:
                continue

            # 가장자리 마진 내 셀은 학습 제외 (궤적 미확립 상태에서 오염 방지)
            if (r < self._edge_margin or r >= gs - self._edge_margin
                    or c < self._edge_margin or c >= gs - self._edge_margin):
                continue

            # ── 거리 기반 alpha 감쇠 ─────────────────────────────────────
            if bbox is not None:
                dist = max(abs(r - r_center), abs(c - c_center))  # Chebyshev 거리
                alpha_ratio = self._bbox_alpha_decay ** dist       # 감쇠 비율
                alpha_cell  = self.alpha * alpha_ratio             # 실제 적용 alpha
            else:
                dist        = 0                                    # 단일 셀 모드 → 항상 중앙점
                alpha_ratio = 1.0
                alpha_cell  = self.alpha

            # ── cross-flow 제한: 차선 횡단 방향 확산 차단 ─────────────────
            # dist>=1 셀에 대해 bbox 중심→해당 셀 오프셋의 횡방향 성분을 계산.
            # 차량 이동 방향에 수직인 방향 = 차선 횡단 방향.
            # 이 성분이 max_cross_flow_cells 초과 시 학습 스킵.
            # 이동 방향과 평행(전후방)은 제한 없음 → 차선 내 커버리지 유지.
            # 설계 근거:
            #   고속도로 중앙선은 이동 방향과 평행 → 횡방향 확산이 차선 넘김.
            #   기존 bbox_learn_w_ratio는 bbox 폭만 제한 (각도 무관).
            #   이 방식은 차량 실제 이동 방향 기반 → 곡선 구간에서도 적응.
            if bbox is not None and dist >= 1:
                _off_px_x = (c - c_center) * self.cell_w       # 셀 오프셋 (px)
                _off_px_y = (r - r_center) * self.cell_h
                # 이동 방향 수직 성분: perp = (-dy, dx)
                # cross_px = |offset · perp| = |off_x * (-dy) + off_y * dx|
                _cross_px = abs(-_cf_dy * _off_px_x + _cf_dx * _off_px_y)
                _avg_cell = (self.cell_w + self.cell_h) * 0.5  # 셀 크기 평균
                if _cross_px / _avg_cell > self._max_cross_flow:
                    continue                                    # 횡방향 초과 → 스킵

            existing = self.flow[r, c]
            emag = np.linalg.norm(existing)

            # ── 중앙점(dist=0): trajectory 기반 게이팅 적용 ────────────
            # 설계 근거:
            #   - 반대 차선 차량의 bbox 중심이 인접 차선 셀을 오염시키는 문제 차단
            #   - velocity_window는 전환 구간에서 노이즈가 많아 게이팅에 부적합
            #     → 궤적 전체 방향(traj[0]→traj[-1])은 실제 진행 방향을 안정적으로 반영
            #   - 반대 방향 거부 시 count -= 2 (기존 -1 대비 2배 빠른 잠금 해제)
            #     → 오염된 셀이 정상 차량으로 min_samples/2 번만 덮어씌워지면 재학습 가능
            #   - _bbox_contra_count: 중앙점 기반으로만 추적
            #     → 실제 차량 중심이 침범한 경우만 중앙선 경계로 판정 (bbox 확장 오탐 방지)
            if dist == 0:
                # ── traj 방향 없으면 중앙점 갱신 자체를 건너뜀 ────────────
                # traj=None: velocity_window 미만이거나 이동량 부족 (방향 신뢰 불가).
                # 이 상태에서 확립된 셀(count>=min_samples)을 갱신하면 오염 위험.
                # 새 셀(count<min_samples)은 velocity_window 방향으로 초기 학습 허용.
                if traj_ndx is None:
                    if self.count[r, c] >= self.min_samples:
                        continue                         # 확립 셀: traj 없으면 건너뜀
                    # 미확립 셀: velocity_window로 초기 학습 허용
                    upd_x, upd_y = ndx, ndy
                else:
                    upd_x, upd_y = traj_ndx, traj_ndy  # trajectory 방향 사용

                # 게이팅: 확립된 셀에서 방향이 반대이면 갱신 거부
                if self.count[r, c] >= self.min_samples and emag > 0.1:
                    cos_val = float(upd_x * existing[0] / emag + upd_y * existing[1] / emag)
                    if cos_val < -0.4:                    # 반대 방향 → 거부
                        if bbox is not None:
                            self._bbox_contra_count[r, c] += 1  # 중앙선 경계 후보 기록
                        self.count[r, c] = max(0, self.count[r, c] - 2)  # 2배 빠른 잠금 해제
                        continue                         # EMA 갱신 거부 ← 핵심
                # 게이팅 통과 → EMA 갱신
                self.flow[r, c, 0] = (1 - alpha_cell) * self.flow[r, c, 0] + alpha_cell * upd_x
                self.flow[r, c, 1] = (1 - alpha_cell) * self.flow[r, c, 1] + alpha_cell * upd_y
                self.count[r, c] += 1
                if self.smoothed_mask[r, c]:
                    self.smoothed_mask[r, c] = False
                continue                                  # 나머지 로직 스킵

            # ── dist >= 1: 방향 게이팅 유지 (bbox edge cells) ────────────
            # _bbox_contra_count는 여기서 추적하지 않음:
            #   bbox 확장으로 반대 차선 차량의 bbox가 경계를 넘어온 경우
            #   → dist=1 셀에 contra를 누적하면 정상 차선의 유효 셀까지 과침식
            #   → 실제 중앙선 침범은 dist=0 추적만으로 충분히 검출 가능
            #
            # [중앙선 침범 방지] dist=1 게이팅:
            #   - traj 방향 우선 사용 (velocity_window보다 안정적 — dist=0과 동일 이유)
            #   - 임계값 -0.4 → -0.2 강화 (완만한 역방향도 거부)
            if alpha_ratio >= self._bbox_gating_ratio:    # dist=1 (decay^1=0.5 ≥ 0.3)
                if self.count[r, c] >= self.min_samples and emag > 0.1:
                    # dist=0과 동일하게 traj 방향 우선, 없으면 velocity_window 사용
                    _g1x = traj_ndx if traj_ndx is not None else ndx
                    _g1y = traj_ndy if traj_ndy is not None else ndy
                    cos_val = float(_g1x * existing[0] / emag + _g1y * existing[1] / emag)
                    if cos_val < -0.2:                    # -0.4 → -0.2 강화 (완만한 역방향도 거부)
                        continue                         # contra·count 변경 없이 스킵
            else:
                # ── dist≥2: soft EMA지만 확립 셀은 반대방향 거부 ──────────
                # 기존: 게이팅 없음 → 반대 차선 bbox 원거리 셀이 soft EMA로 조금씩 오염
                # 수정: 확립 셀(count≥min_samples)에서 반대방향이면 soft EMA도 거부
                # dist=1(-0.2)보다 완화된 -0.3 사용 — 멀리서 넓게 학습하는 soft 셀 보호
                if self.count[r, c] >= self.min_samples and emag > 0.1:
                    _g2x = traj_ndx if traj_ndx is not None else ndx
                    _g2y = traj_ndy if traj_ndy is not None else ndy
                    cos_val = float(_g2x * existing[0] / emag + _g2y * existing[1] / emag)
                    if cos_val < -0.3:                    # 확립 셀 soft 오염 방지
                        continue

            # EMA 갱신 (중심에서 가까울수록 alpha 큼 → 강하게 학습)
            self.flow[r, c, 0] = (1 - alpha_cell) * self.flow[r, c, 0] + alpha_cell * ndx
            self.flow[r, c, 1] = (1 - alpha_cell) * self.flow[r, c, 1] + alpha_cell * ndy

            # count: dist=1(강한 기여)만 증가, dist≥2(soft)는 항상 미증가 (잠금 방지)
            if alpha_ratio >= self._bbox_gating_ratio:
                self.count[r, c] += 1

            if self.smoothed_mask[r, c]:                 # 보간 셀에 실 데이터 → 표시 해제
                self.smoothed_mask[r, c] = False

        # 디버그: 100회마다 학습 현황 출력
        self._learn_call_count += 1
        if self._learn_call_count % 100 == 0:
            active_cells = int(np.sum(self.count > 0))
            total_samples = int(self.count.sum())
            angle = np.degrees(np.arctan2(ndy, ndx))
            contra_total = int(self._bbox_contra_count.sum())
            print(f"   📈 learn_step #{self._learn_call_count}: "
                  f"cell[{_log_r},{_log_c}] cnt={self.count[_log_r,_log_c]}, "
                  f"angle={angle:+.0f}°, "
                  f"active_cells={active_cells}/{self.grid_size**2}, "
                  f"total={total_samples}, contra_hits={contra_total}")

    # ==================== 이중 선형 보간 ====================
    def _interpolate_arr(self, x, y, flow_arr):
        """flow_arr에서 (x, y) 위치의 이중 선형 보간 단위 벡터 반환.
        보간 결과가 너무 작으면 None.
        """
        r, c = self._cell_coords(x, y)
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        dr, dc = r - r0, c - c0
        gs = self.grid_size - 1
        r0, r1 = int(np.clip(r0, 0, gs)), int(np.clip(r0 + 1, 0, gs))
        c0, c1 = int(np.clip(c0, 0, gs)), int(np.clip(c0 + 1, 0, gs))
        top     = (1 - dc) * flow_arr[r0, c0] + dc * flow_arr[r0, c1]
        bottom  = (1 - dc) * flow_arr[r1, c0] + dc * flow_arr[r1, c1]
        final_v = (1 - dr) * top + dr * bottom
        mag = np.linalg.norm(final_v)
        return final_v / (mag + 1e-6) if mag > 0.1 else None

    def get_interpolated(self, x, y, direction=None):
        """이중 선형 보간으로 (x, y) 위치의 흐름 벡터 추정.

        Args:
            x, y: 픽셀 좌표.
            direction: 'a' | 'b' | None.
                'a'/'b' 지정 시 해당 방향 채널(flow_a / flow_b)을 우선 조회.

        채널 활성화(build_directional_channels 이후):
          ① 채널에 데이터 있으면 → 채널 벡터 사용 (반대 차선 오염 없음)
          ② 채널 데이터 없음 → 오염-인식 글로벌 fallback:
               글로벌 방향이 쿼리 방향과 일치하면 반환,
               반대 방향이면 None (오염 벡터 → vote loop skip)
          채널 미활성화(direction=None 또는 학습 완료 전) → 글로벌 그대로 사용
        """
        if direction is not None and self._ref_dx is not None:
            # ── 양방향 채널 활성화 상태 ─────────────────────────────────────
            chan_flow  = self.flow_a  if direction == 'a' else self.flow_b
            chan_count = self.count_a if direction == 'a' else self.count_b
            r, c = self._cell_coords(x, y)
            r0 = int(np.clip(np.floor(r), 0, self.grid_size - 1))
            c0 = int(np.clip(np.floor(c), 0, self.grid_size - 1))
            r1 = min(r0 + 1, self.grid_size - 1)
            c1 = min(c0 + 1, self.grid_size - 1)
            # 4개 인접 셀 중 최소 2개에 채널 데이터가 있을 때만 채널 보간 사용.
            # 1개만 있으면 경계 셀 1개의 방향이 그대로 반영 → 방향 오차가 크면 오탐 유발.
            # 2개 이상: 보간이 두 방향의 평균 → 이상치 영향 희석.
            _ch_cnt = ((1 if chan_count[r0, c0] > 0 else 0)
                       + (1 if chan_count[r0, c1] > 0 else 0)
                       + (1 if chan_count[r1, c0] > 0 else 0)
                       + (1 if chan_count[r1, c1] > 0 else 0))
            if _ch_cnt >= 2:
                result = self._interpolate_arr(x, y, chan_flow)
                if result is not None:
                    return result

            # ② 채널 데이터 없음 → 오염-인식 글로벌 fallback ─────────────
            # 글로벌 맵 방향이 쿼리 방향과 반대(오염)이면 None 반환.
            # 예: A차량 flow_a 조회 → 없음 → 글로벌=B방향(오염) → None 반환
            #     A차량 flow_a 조회 → 없음 → 글로벌=A방향(clean) → 반환
            _gv = self._interpolate_arr(x, y, self.flow)
            if _gv is not None:
                _cos_g = float(_gv[0] * self._ref_dx + _gv[1] * self._ref_dy)
                if (direction == 'a') == (_cos_g >= 0):
                    return _gv        # 방향 일치 → 신뢰 가능
            return None               # 방향 불일치 → 오염 가능성 → skip

        # 글로벌 맵 (채널 미활성화·direction=None)
        return self._interpolate_arr(x, y, self.flow)

    # ==================== 공간 평활화 (원본 방식 + 방향 일관성 보호) ====================
    def apply_spatial_smoothing(self, verbose=False):
        """데이터가 부족한 셀을 주변 이웃 평균으로 보정한다.

        Args:
            verbose: True이면 상세 진단 출력 (학습 완료 시에만 True로 호출).

        원본 코드 방식을 기반으로 하되, 중앙 분리대 경계 오염을 방지한다.

        동작 원칙:
          ① count >= min_samples 셀 → 이미 충분한 데이터, 건드리지 않음
             smoothed_mask = False 보장 (실 데이터 있는 셀)
          ② 0 < count < min_samples 셀 → 자신의 방향 기준 같은 방향 이웃만 평균
          ③ count=0 셀 → 이웃이 전부 같은 방향이면 채움 (이웃끼리 반대면 skip)
             이웃끼리 cos < -0.3이면 중앙 분리대 경계 → 채우지 않음
             채움 시 smoothed_mask = True (보간 전용 셀 표시)
        """
        if verbose:                                       # 상세 진단 시에만 전체 셀 출력
            print("\n📊 [진단] smoothing 전 — 전체 셀 상태")
            print(f"   {'[r,c]':>8} {'count':>6} {'angle°':>8} {'mag':>6} {'smooth':>7}")
            for r in range(self.grid_size):               # 모든 행 순회
                for c in range(self.grid_size):           # 모든 열 순회
                    v = self.flow[r, c]                   # 해당 셀 벡터
                    m = np.linalg.norm(v)                 # 벡터 크기
                    if m > 0.01:                          # 유효한 벡터만 출력
                        angle = np.degrees(np.arctan2(v[1], v[0]))  # 각도
                        cnt = self.count[r, c]            # 샘플 수
                        sm = "★" if self.smoothed_mask[r, c] else ""  # 보간 셀 표시
                        mk = " ⚠️" if cnt < self.min_samples else ""
                        print(f"   [{r:2d},{c:2d}] {cnt:6d} {angle:8.1f} {m:6.3f}{mk} {sm}")

        new_map = self.flow.copy()                        # 기존 맵 복사본 (수정 대상)
        filled_count = 0                                  # count=0에서 채워진 셀 수
        reinforced_count = 0                              # 0<count<min_samples 강화된 셀 수

        # ── BFS flood-fill: 격자 모서리부터 연결된 빈 셀 = "외부" 판별 ────
        # 외부 = 차량이 실제로 지나간 영역 밖 (도로 이외 구역, 하늘, 갓길 등)
        # 모서리에서 BFS로 연결된 count=0 셀은 외부로 표시 → 채우지 않음
        # eroded_mask 셀은 중앙 분리대 경계 → BFS 통행 차단(벽 역할)
        gs = self.grid_size                               # 그리드 크기 단축 참조
        exterior = np.zeros((gs, gs), dtype=bool)         # 외부 셀 마스크 (True=외부)
        bfs_queue = deque()                               # BFS 탐색 큐

        for r in range(gs):                               # 좌우 경계 행 순회
            for c in [0, gs - 1]:                         # 첫 열·마지막 열
                if (self.count[r, c] == 0                 # 빈 셀이고
                        and not self.eroded_mask[r, c]    # erosion 셀(벽)이 아니고
                        and not exterior[r, c]):           # 아직 미방문이면
                    exterior[r, c] = True                 # 외부 표시
                    bfs_queue.append((r, c))              # 큐에 추가

        for c in range(gs):                               # 상하 경계 열 순회
            for r in [0, gs - 1]:                         # 첫 행·마지막 행
                if (self.count[r, c] == 0
                        and not self.eroded_mask[r, c]
                        and not exterior[r, c]):
                    exterior[r, c] = True
                    bfs_queue.append((r, c))

        while bfs_queue:                                  # BFS 확산 (4방향)
            r, c = bfs_queue.popleft()                    # 현재 셀 꺼냄
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4방향
                nr, nc = r + dr, c + dc                   # 이웃 좌표
                if (0 <= nr < gs and 0 <= nc < gs         # 범위 내
                        and not exterior[nr, nc]          # 미방문
                        and self.count[nr, nc] == 0       # 빈 셀
                        and not self.eroded_mask[nr, nc]):  # erosion 벽 아님
                    exterior[nr, nc] = True               # 외부로 표시
                    bfs_queue.append((nr, nc))            # 큐에 추가

        interior_holes = int(np.sum(                      # 내부 홀 수 (디버그용)
            (~exterior) & (self.count == 0) & (~self.eroded_mask)
        ))

        for r in range(self.grid_size):                   # 모든 칸 순회
            for c in range(self.grid_size):

                own_cnt = self.count[r, c]                # 이 셀의 학습 샘플 수

                # ── ① count >= min_samples: 충분한 셀은 건드리지 않음 ──────
                if own_cnt >= self.min_samples:           # 충분한 셀 → 변경 불필요
                    self.smoothed_mask[r, c] = False      # 실 데이터 충분 → 보간 표시 확실히 해제
                    continue

                # ── 3×3 이웃 범위 계산 ──────────────────────────────────────
                r_s = max(0, r - 1)                       # 이웃 행 범위 시작
                r_e = min(self.grid_size, r + 2)          # 이웃 행 범위 끝
                c_s = max(0, c - 1)                       # 이웃 열 범위 시작
                c_e = min(self.grid_size, c + 2)          # 이웃 열 범위 끝

                if own_cnt == 0:
                    # ── ③ count=0: 외부 셀이면 채우지 않음 ──────────────────
                    # 외부에서 1칸 확장 허용 시 반대 차선 방향이 경계를 넘어 오염됨.
                    # (확장된 셀끼리 방향 일치 → boundary_erosion이 제거 불가)
                    # 순수 내부 홀(exterior=False)만 채움 — 외부 확장 없음.
                    if exterior[r, c]:                    # 외부 셀이면
                        continue                          # 채우지 않음

                    # ── 이웃끼리 방향 일관성 확인 후 채움 (내부 홀만) ────────
                    # 이웃 중 유효 벡터 수집
                    neighbor_vecs = []                    # 이웃 유효 벡터 목록
                    for nr in range(r_s, r_e):            # 이웃 행 순회
                        for nc in range(c_s, c_e):        # 이웃 열 순회
                            if nr == r and nc == c:       # 자기 자신 건너뜀
                                continue
                            nv = self.flow[nr, nc]        # 이웃 벡터
                            if np.linalg.norm(nv) > 0.1:  # 유효 벡터만
                                neighbor_vecs.append(nv)  # 목록에 추가

                    if len(neighbor_vecs) == 0:           # 유효 이웃 없으면 건너뜀
                        continue

                    # 이웃끼리 반대 방향 쌍이 있는지 검사 (중앙 분리대 경계 판별)
                    boundary = False                      # 경계 플래그 초기화
                    for i in range(len(neighbor_vecs)):    # 모든 이웃 쌍 검사
                        for j in range(i + 1, len(neighbor_vecs)):
                            ni = neighbor_vecs[i] / (np.linalg.norm(neighbor_vecs[i]) + 1e-6)
                            nj = neighbor_vecs[j] / (np.linalg.norm(neighbor_vecs[j]) + 1e-6)
                            if float(np.dot(ni, nj)) < -0.3:  # 반대 방향 쌍 발견
                                boundary = True           # 경계로 판정
                                break
                        if boundary:                      # 경계면 외부 루프도 탈출
                            break

                    if boundary:                          # 중앙 분리대 경계 → 채우지 않음
                        continue

                    # 이웃이 전부 같은 방향 → 평균으로 채움 (원본 방식)
                    avg_v = np.mean(neighbor_vecs, axis=0)  # 이웃 평균 벡터
                    if np.linalg.norm(avg_v) > 0.1:       # 유효한 평균이면
                        new_map[r, c] = avg_v             # 셀 채움
                        filled_count += 1                 # 채움 카운터 증가
                        # ── 개선 1: 보간으로 채워진 셀 표시 ──────────────────
                        # count=0 셀이 이웃 평균으로 채워짐 → 실 데이터 없는 보간 셀
                        # judge.py에서 이 셀은 cos_threshold를 완화해 오탐 방지
                        self.smoothed_mask[r, c] = True   # 보간 채움 표시

                else:
                    # ── ② 0 < count < min_samples: 자신의 방향 기준 강화 ───
                    own_v = self.flow[r, c]               # 셀 자신의 현재 벡터
                    own_mag = np.linalg.norm(own_v)       # 자신의 벡터 크기
                    if own_mag < 0.05:                    # 벡터가 너무 작으면
                        continue                          # 방향 불명확 → 건너뜀

                    own_norm = own_v / (own_mag + 1e-6)   # 기준 단위 벡터

                    # 자신과 같은 방향(cos >= 0)인 이웃만 수집
                    consistent_vecs = [own_v]             # 자신 포함
                    for nr in range(r_s, r_e):            # 이웃 행 순회
                        for nc in range(c_s, c_e):        # 이웃 열 순회
                            if nr == r and nc == c:       # 자기 자신 건너뜀
                                continue
                            nv = self.flow[nr, nc]        # 이웃 벡터
                            n_mag = np.linalg.norm(nv)    # 이웃 벡터 크기
                            if n_mag < 0.1:               # 벡터 없는 셀 건너뜀
                                continue
                            n_norm = nv / (n_mag + 1e-6)  # 이웃 단위 벡터
                            cos_val = float(np.dot(own_norm, n_norm))  # 코사인
                            if cos_val >= 0:              # 같은 방향(90° 이내)만
                                consistent_vecs.append(nv)  # 목록에 추가

                    avg_v = np.mean(consistent_vecs, axis=0)  # 일관 방향 평균
                    if np.linalg.norm(avg_v) > 0.1:       # 유효한 평균이면
                        new_map[r, c] = avg_v             # 셀 업데이트
                        reinforced_count += 1             # 강화 카운터 증가

        self.flow = new_map                               # 맵 갱신

        # 간단 요약 출력 (매 호출 시)
        active = int(np.sum(self.count > 0))              # 실 학습 데이터 있는 셀 수
        total_with_flow = int(np.sum(                     # 유효 벡터 있는 셀 수
            np.linalg.norm(self.flow, axis=2) > 0.1
        ))
        smoothed_total = int(np.sum(self.smoothed_mask))  # 보간 채움 셀 총 수
        exterior_total = int(np.sum(exterior))            # 외부 셀 수 (flood-fill 결과)
        print(f"   🔄 smoothing: {filled_count}셀 채움 (내부홀/{interior_holes}개), "
              f"{reinforced_count}셀 강화, "
              f"learned={active}/{self.grid_size**2}, "
              f"total_flow={total_with_flow}, "
              f"smoothed={smoothed_total}, "
              f"exterior={exterior_total}(채움 제외)")

        if verbose:                                       # 상세 진단 시에만 후 상태 출력
            print(f"\n📊 [진단] smoothing 후 — 전체 셀 상태")
            print(f"   {'[r,c]':>8} {'count':>6} {'angle°':>8} {'mag':>6} {'smooth':>7}")
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    v = self.flow[r, c]
                    m = np.linalg.norm(v)
                    if m > 0.01:
                        angle = np.degrees(np.arctan2(v[1], v[0]))
                        cnt = self.count[r, c]
                        sm = "★" if self.smoothed_mask[r, c] else ""  # 보간 셀 표시
                        mk = " ⚠️" if cnt < self.min_samples else ""
                        print(f"   [{r:2d},{c:2d}] {cnt:6d} {angle:8.1f} {m:6.3f}{mk} {sm}")

    # ==================== Phase 1 정체 탐지용 — 셀별 속도 학습 ====================
    def learn_baseline(self, fx: float, fy: float, norm_speed: float):
        """SMOOTH 온라인 구간에서 셀별 정상 normalized_speed를 EMA로 갱신한다.

        Args:
            fx: footpoint x 좌표.
            fy: footpoint y 좌표.
            norm_speed: 해당 차량의 normalized_mag (mag / bbox_h).
        """
        r = int(np.clip(fy / self.cell_h, 0, self.grid_size - 1))  # 셀 행 계산 (범위 클램프)
        c = int(np.clip(fx / self.cell_w, 0, self.grid_size - 1))  # 셀 열 계산 (범위 클램프)
        if self.speed_ref[r, c] == 0:                     # 첫 번째 데이터이면
            self.speed_ref[r, c] = norm_speed             # 그대로 설정
        else:                                              # 이미 값이 있으면 EMA 갱신
            self.speed_ref[r, c] = (                      # EMA: 기존값 (1-alpha) + 새값 alpha
                (1 - self.alpha) * self.speed_ref[r, c]
                + self.alpha * norm_speed
            )

    # ==================== 개선 1: smoothed_mask 조회 메서드 ====================
    def is_smoothed(self, r: int, c: int) -> bool:
        """해당 셀이 보간으로 채워진 셀인지 반환 (실 데이터 없음).

        Args:
            r: 그리드 행 인덱스.
            c: 그리드 열 인덱스.

        Returns:
            True=보간 채움 셀 (cos_threshold 완화 대상), False=실 데이터 있는 셀.
        """
        r = int(np.clip(r, 0, self.grid_size - 1))       # 범위 보정 (음수·오버플로 방지)
        c = int(np.clip(c, 0, self.grid_size - 1))       # 범위 보정
        return bool(self.smoothed_mask[r, c])             # bool 변환 후 반환

    def get_nearest_direction(self, x: float, y: float):
        """get_interpolated가 None을 반환할 때 가장 가까운 학습된 셀의 방향 벡터를 반환한다.

        미학습 구역(예: 프레임 상단) 차량의 방향 분류 fallback으로 사용.
        가장 가까운 학습 셀(count > 0)을 그리드 좌표 기준 유클리드 거리로 탐색한다.

        Args:
            x: 픽셀 x 좌표 (차량 footpoint).
            y: 픽셀 y 좌표 (차량 footpoint).

        Returns:
            단위 벡터 (numpy array) 또는 학습 셀이 없으면 None.
        """
        fr, fc = self._cell_coords(x, y)               # 차량 위치의 그리드 좌표 (실수)
        best_dist_sq = float("inf")                    # 최소 거리 제곱 (초기 무한대)
        best_v = None                                  # 최근접 셀 벡터 (초기 None)

        for ri in range(self.grid_size):               # 전체 셀 순회 (20×20 = 400회)
            for ci in range(self.grid_size):
                if self.count[ri, ci] <= 0:            # 미학습 셀 건너뜀
                    continue
                dist_sq = (ri - fr) ** 2 + (ci - fc) ** 2  # 거리 제곱 (sqrt 생략)
                if dist_sq < best_dist_sq:             # 더 가까운 셀 발견
                    best_dist_sq = dist_sq             # 거리 갱신
                    best_v = self.flow[ri, ci]         # 벡터 갱신

        if best_v is None:                             # 학습된 셀 없음
            return None
        mag = np.linalg.norm(best_v)                   # 벡터 크기
        return best_v / (mag + 1e-6) if mag > 0.1 else None  # 단위 벡터 반환

    def get_cell_rc(self, px: float, py: float):
        """픽셀 좌표(px, py)를 정수 셀 좌표(r, c)로 변환.

        judge.py에서 smoothed_mask 조회 시 셀 좌표가 필요하므로 편의 메서드 제공.

        Args:
            px: 픽셀 x 좌표.
            py: 픽셀 y 좌표.

        Returns:
            (r, c) 정수 튜플.
        """
        r = int(np.clip(py / self.cell_h, 0, self.grid_size - 1))  # 행 계산 + 범위 보정
        c = int(np.clip(px / self.cell_w, 0, self.grid_size - 1))  # 열 계산 + 범위 보정
        return r, c

    # ==================== bbox 겹침 기반 경계 제거 ====================
    def apply_overlap_erosion(self, contra_threshold: int = 3):
        """학습 중 반대 차선 차량의 bbox가 N회 이상 밟은 셀을 중앙선 경계로 판정해 제거한다.

        apply_boundary_erosion()의 대체·보완 메서드.
        - 기존: 인접 1칸에 반대방향 이웃이 있으면 제거 (단순 형태학적 침식)
        - 개선: 실제 반대 차선 차량의 bbox 발자국 데이터 기반 → 실제 겹친 셀만 제거

        호출 순서:
          apply_spatial_smoothing()   ← 초기 채움
          apply_overlap_erosion()     ← bbox 겹침 셀 제거
          apply_spatial_smoothing()   ← 삭제 후 재채움 (중앙선 불가침)

        Args:
            contra_threshold: 반대방향 bbox 방문 횟수 임계값 (이 이상이면 경계로 판정).
                              기본값 3: 1~2회 우연한 방문(합류로·진입로)은 무시,
                              3회 이상 = 차선 중첩 구간으로 확정.
        """
        erased = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self._bbox_contra_count[r, c] >= contra_threshold:
                    self.flow[r, c]  = 0                  # 벡터 초기화
                    self.count[r, c] = 0                  # 샘플 수 초기화
                    self.eroded_mask[r, c]    = True      # 영구 재학습 금지
                    self.smoothed_mask[r, c]  = False     # smoothed 표시 해제
                    erased += 1
        print(f"   ✂️  overlap_erosion: {erased}셀 제거 "
              f"(bbox 반대방향 방문 ≥ {contra_threshold}회)")

    # ==================== 프레임 스킵 방향 오류 교정 ====================
    def apply_direction_repair(self,
                               repair_cos_threshold: float = -0.3,
                               min_consistent_neighbors: int = 3):
        """카메라 끊김(프레임 스킵)으로 반대 방향으로 학습된 셀을 이웃 평균으로 교정한다.

        apply_boundary_erosion / apply_overlap_erosion이 셀을 '제거'하는 것과 달리,
        이 메서드는 방향만 '교정'하고 셀 자체는 유지한다.

        동작 원리:
          1) 각 셀의 방향을 3×3 이웃 평균과 비교.
          2) 이웃 대부분이 일관된 방향을 가리키는데 (경계 구역이 아님)
             이 셀만 반대 방향(cos < repair_cos_threshold)이면 이웃 평균으로 덮어씀.
          3) 이웃끼리 방향이 불일치(경계 구역 판정)하면 교정하지 않음
             → 실제 중앙선 경계 셀과 프레임 스킵 아티팩트 셀을 구분.

        호출 위치:
          apply_spatial_smoothing(verbose=True)  ← ① 초기 채움
          apply_overlap_erosion()                ← ② 중앙선 경계 제거
          apply_direction_repair()               ← ③ 프레임 스킵 오류 교정  ★ HERE
          apply_spatial_smoothing()              ← ④ 재채움

        Args:
            repair_cos_threshold: 이웃 평균과의 코사인이 이 값 미만이면 교정 대상.
                                  -0.3 ≈ 107° 이상 방향 어긋남 → 명백한 반전 셀만 교정.
            min_consistent_neighbors: 교정 판단을 위한 최소 일관성 이웃 수.
                                      부족하면 데이터 부족으로 보고 교정 건너뜀.
        """
        repaired = 0
        new_flow = self.flow.copy()                        # 교정 결과 버퍼 (원본 유지)
        gs = self.grid_size

        for r in range(gs):
            for c in range(gs):
                if self.eroded_mask[r, c]:                 # 이미 제거된 셀 건너뜀
                    continue

                v    = self.flow[r, c]
                vmag = np.linalg.norm(v)
                if vmag < 0.1:                             # 빈 셀 건너뜀
                    continue

                vn = v / (vmag + 1e-6)                     # 이 셀의 단위 벡터

                # ── 3×3 이웃 수집 ────────────────────────────────────────
                neighbor_vecs = []
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:            # 자기 자신 건너뜀
                            continue
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < gs and 0 <= nc < gs):
                            continue
                        if self.eroded_mask[nr, nc]:       # 제거된 이웃 건너뜀
                            continue
                        nv = self.flow[nr, nc]
                        if np.linalg.norm(nv) > 0.1:
                            neighbor_vecs.append(nv)

                if len(neighbor_vecs) < min_consistent_neighbors:
                    continue                               # 이웃 부족 → 판단 불가

                # ── 이웃 평균 방향 계산 ──────────────────────────────────
                avg_v   = np.mean(neighbor_vecs, axis=0)
                avg_mag = np.linalg.norm(avg_v)
                if avg_mag < 0.1:                          # 이웃 평균이 상쇄됨 → 경계 구역
                    continue

                avg_n = avg_v / (avg_mag + 1e-6)           # 이웃 평균 단위 벡터

                # ── 이웃 일관성 확인 (경계 구역 vs 아티팩트 구분) ────────
                # 이웃들이 이웃 평균과 얼마나 일치하는지 계산.
                # 경계 구역(중앙선 근처): 이웃끼리 반대 방향 존재 → 일관성 낮음 → 건너뜀.
                # 프레임 스킵 아티팩트: 이웃은 모두 한 방향인데 이 셀만 반대 → 일관성 높음 → 교정.
                consistent_count = sum(
                    1 for nv in neighbor_vecs
                    if float(np.dot(avg_n,
                                    nv / (np.linalg.norm(nv) + 1e-6))) > 0.3
                )
                if consistent_count < min_consistent_neighbors:
                    continue                               # 이웃 불일치 → 경계 구역 → 건너뜀

                # ── 이 셀이 이웃 평균과 반대 방향인지 확인 ───────────────
                cos_vs_avg = float(np.dot(vn, avg_n))
                if cos_vs_avg < repair_cos_threshold:      # 반대 방향 → 교정
                    new_flow[r, c] = avg_v                 # 이웃 평균으로 덮어씌움
                    self.smoothed_mask[r, c] = True        # 보간·교정 셀 표시 (판정 완화용)
                    repaired += 1

        self.flow = new_flow                               # 교정 결과 적용
        print(f"   🔧 direction_repair: {repaired}셀 교정 "
              f"(cos<{repair_cos_threshold}, min_neighbors={min_consistent_neighbors})")

    # ==================== 인접 방향 기반 경계 제거 (legacy fallback) ====================
    def apply_boundary_erosion(self, majority_threshold: float = 0.4):
        """방향이 반전되는 경계 셀을 두 단계로 제거한다.

        학습 완료 후 apply_spatial_smoothing() 직후 1회 호출한다.

        1단계 — 인접(4방향) 반대 방향 이웃 존재 시 즉시 제거:
          4방향 직접 이웃 중 cos < -0.5인 셀이 하나라도 있으면 제거.
          중앙 분리대 경계처럼 바로 옆에 반대 방향이 붙어있는 셀을 처리.

        2단계 — 다수결 erosion (1단계 결과 기반):
          3×3 이웃(최대 8셀) 중 유효 벡터를 수집하고,
          현재 셀과 반대 방향(cos < -0.3)인 이웃 비율 ≥ majority_threshold이면 제거.
          주변 대부분의 셀이 반대 방향인 고립·오염 셀을 추가로 제거.

        Args:
            majority_threshold: 다수결 판정 반대방향 비율 임계값 (기본 0.4 = 40%).
        """
        new_flow  = self.flow.copy()                       # 수정 대상 복사본
        new_count = self.count.copy()                      # count도 초기화 대상
        erased_adj = 0                                     # 1단계 제거 수
        erased_maj = 0                                     # 2단계 제거 수

        # ── 1단계: 4방향 직접 이웃에 반대 방향 있으면 즉시 제거 ────────
        for r in range(self.grid_size):                    # 모든 셀 순회
            for c in range(self.grid_size):
                v = self.flow[r, c]                        # 이 셀의 벡터
                vmag = np.linalg.norm(v)                   # 벡터 크기
                if vmag < 0.1:                             # 이미 빈 셀이면 건너뜀
                    continue

                vn = v / (vmag + 1e-6)                     # 단위 벡터

                # 4방향 이웃 검사 (상/하/좌/우)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc                # 이웃 셀 인덱스
                    if not (0 <= nr < self.grid_size       # 범위 밖 건너뜀
                            and 0 <= nc < self.grid_size):
                        continue
                    nv = self.flow[nr, nc]                 # 이웃 벡터
                    nmag = np.linalg.norm(nv)              # 이웃 벡터 크기
                    if nmag < 0.1:                         # 빈 이웃 건너뜀
                        continue
                    nn = nv / (nmag + 1e-6)                # 이웃 단위 벡터
                    if float(np.dot(vn, nn)) < -0.5:       # 반대 방향 이웃 발견
                        new_flow[r, c]  = 0                # 이 셀 벡터 초기화
                        new_count[r, c] = 0                # count도 초기화
                        self.eroded_mask[r, c] = True      # 영구 재학습 금지 마스크 설정
                        self.smoothed_mask[r, c] = False   # erosion 된 셀은 smoothed 해제 (빈 셀)
                        erased_adj += 1                    # 카운터 증가
                        break                              # 한 이웃만 확인해도 충분

        # ── 2단계: 다수결 erosion — 1단계 결과를 기반으로 3×3 이웃 검사 ─
        # new_flow(1단계 반영본)를 읽어 반대방향 비율 계산 → 추가 제거
        flow_after_step1 = new_flow.copy()                 # 1단계 결과 스냅샷 (읽기 전용)
        for r in range(self.grid_size):                    # 모든 셀 순회
            for c in range(self.grid_size):
                if self.eroded_mask[r, c]:                 # 1단계에서 이미 제거된 셀 건너뜀
                    continue
                v = flow_after_step1[r, c]                 # 이 셀의 벡터 (1단계 기반)
                vmag = np.linalg.norm(v)                   # 벡터 크기
                if vmag < 0.1:                             # 빈 셀이면 건너뜀
                    continue

                vn = v / (vmag + 1e-6)                     # 단위 벡터

                # 3×3 이웃 수집 (자기 자신 제외)
                total_neighbors = 0                        # 유효 이웃 총 수
                opposite_count  = 0                        # 반대 방향 이웃 수
                for dr in range(-1, 2):                    # 행 방향 ±1
                    for dc in range(-1, 2):                # 열 방향 ±1
                        if dr == 0 and dc == 0:            # 자기 자신 건너뜀
                            continue
                        nr, nc = r + dr, c + dc            # 이웃 셀 인덱스
                        if not (0 <= nr < self.grid_size   # 범위 밖 건너뜀
                                and 0 <= nc < self.grid_size):
                            continue
                        nv = flow_after_step1[nr, nc]      # 이웃 벡터 (1단계 기반)
                        nmag = np.linalg.norm(nv)          # 이웃 벡터 크기
                        if nmag < 0.1:                     # 빈 이웃 건너뜀
                            continue
                        total_neighbors += 1               # 유효 이웃 카운터 증가
                        nn = nv / (nmag + 1e-6)            # 이웃 단위 벡터
                        if float(np.dot(vn, nn)) < -0.3:   # 반대 방향 (150° 이상)
                            opposite_count += 1            # 반대 방향 카운터 증가

                # 유효 이웃이 있고, 반대 방향 비율 ≥ majority_threshold이면 제거
                if total_neighbors > 0:                    # 이웃이 하나라도 있어야 판정
                    opposite_ratio = opposite_count / total_neighbors  # 반대 비율
                    if opposite_ratio >= majority_threshold:            # 다수결 기준 초과
                        new_flow[r, c]  = 0                # 이 셀 벡터 초기화
                        new_count[r, c] = 0                # count도 초기화
                        self.eroded_mask[r, c] = True      # 영구 재학습 금지 마스크
                        self.smoothed_mask[r, c] = False   # erosion 된 셀은 smoothed 해제
                        erased_maj += 1                    # 2단계 카운터 증가

        self.flow  = new_flow                              # 맵 갱신
        self.count = new_count                             # count 갱신
        print(f"   ✂️  boundary_erosion: 1단계(인접)={erased_adj}셀, "
              f"2단계(다수결 ≥{majority_threshold*100:.0f}%)={erased_maj}셀 제거 "
              f"(flow_size={self.grid_size}x{self.grid_size})")

    # ==================== 저장/로드 ====================
    def build_directional_channels(self, ref_dx: float, ref_dy: float):
        """학습 완료 후 글로벌 flow map을 A/B 두 채널로 분리.

        Args:
            ref_dx, ref_dy: A방향 기준 단위 벡터 (detector._ref_direction).
                cos >= 0인 셀 → A채널 / cos < 0인 셀 → B채널

        학습 단계에서는 글로벌 맵(self.flow)에 모든 차량이 학습됨.
        학습 완료 후 이 메서드를 호출하면 각 셀의 방향을 기준 방향과 비교해
        A/B 채널로 분리. 이후 get_interpolated(direction='a'/'b') 사용 가능.

        중앙선 오염 방지 원리:
          - A차량은 flow_a만 조회 → B차량이 A셀을 오염시켜도 영향 없음
          - B차량은 flow_b만 조회 → A차량이 B셀을 오염시켜도 영향 없음
          - 학습 기반 분리라 EMA 게이팅과 독립적으로 동작
        """
        self.flow_a[:] = 0
        self.count_a[:] = 0
        self.flow_b[:] = 0
        self.count_b[:] = 0
        self._ref_dx = ref_dx                             # 기준 방향 저장 (get_interpolated fallback 필터용)
        self._ref_dy = ref_dy

        a_cells = 0
        b_cells = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.count[r, c] < self.min_samples:
                    continue                               # 미확립 셀 건너뜀
                v = self.flow[r, c]
                cos = float(v[0] * ref_dx + v[1] * ref_dy)
                if cos >= 0:                              # A방향 셀
                    self.flow_a[r, c]  = v
                    self.count_a[r, c] = self.count[r, c]
                    a_cells += 1
                else:                                     # B방향 셀
                    self.flow_b[r, c]  = v
                    self.count_b[r, c] = self.count[r, c]
                    b_cells += 1

        print(f"✅ 양방향 채널 구축: A={a_cells}셀, B={b_cells}셀 "
              f"(글로벌 {a_cells+b_cells}/{self.grid_size**2}셀 분리)")

    def save(self, path: Path):
        """학습된 flow_map, count, speed_ref, smoothed_mask를 .npy 파일로 저장.

        Args:
            path: 저장 파일 경로 (.npy).
        """
        path.parent.mkdir(parents=True, exist_ok=True)    # 저장 폴더 생성
        data = {                                           # 저장할 데이터 딕셔너리
            "version":       4,                           # 포맷 버전 (4=양방향 채널 추가)
            "flow":          self.flow,                   # 흐름 벡터 배열 (글로벌)
            "count":         self.count,                  # 셀별 샘플 수 배열
            "speed_ref":     self.speed_ref,              # 셀별 정상 속도 배열
            "smoothed_mask": self.smoothed_mask,          # 보간 채움 셀 마스크
            "flow_a":        self.flow_a,                 # A방향 채널 벡터
            "count_a":       self.count_a,                # A방향 채널 카운터
            "flow_b":        self.flow_b,                 # B방향 채널 벡터
            "count_b":       self.count_b,                # B방향 채널 카운터
        }
        np.save(path, data)                               # .npy 파일로 저장
        print(f"✅ flow_map 저장: {path}")

    def load(self, path: Path) -> bool:
        """기존에 저장된 flow_map 파일이 있으면 불러와서 사용.

        Returns:
            bool: 로드 성공 여부.
        """
        if not path.exists():                             # 파일 없으면
            return False                                  # 로드 실패
        data = np.load(path, allow_pickle=True).item()    # .npy 로드 (dict)
        loaded = data["flow"]                             # flow 배열 추출
        if loaded.shape[0] != self.grid_size:             # grid_size 불일치이면
            print("⚠️ grid 불일치, 초기화")
            return False                                  # 로드 실패
        self.flow  = loaded                               # flow_map 로드
        self.count = data["count"]                        # count 로드

        version = data.get("version", 1)                  # 포맷 버전 확인 (없으면 1)
        if version >= 2 and "speed_ref" in data:          # 버전 2+ & speed_ref 있으면
            self.speed_ref = data["speed_ref"]            # 셀별 속도 기준 로드
        if "smoothed_mask" in data:                       # smoothed_mask가 저장되어 있으면
            self.smoothed_mask = data["smoothed_mask"]    # 보간 마스크 로드
        if version >= 4:                                  # 버전 4+: 양방향 채널 로드
            if "flow_a" in data:
                self.flow_a  = data["flow_a"]
                self.count_a = data["count_a"]
            if "flow_b" in data:
                self.flow_b  = data["flow_b"]
                self.count_b = data["count_b"]
        _ch_a = int(np.sum(self.count_a > 0))
        _ch_b = int(np.sum(self.count_b > 0))
        print(f"✅ flow_map 로드 ({self.count.sum()} 샘플, ver={version}, "
              f"smoothed={int(np.sum(self.smoothed_mask))}셀, "
              f"채널 A={_ch_a}셀 B={_ch_b}셀)")
        return True                                       # 로드 성공
