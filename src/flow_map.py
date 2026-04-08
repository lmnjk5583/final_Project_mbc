# 파일 경로: C:\final_pj\src\flow_map.py
# 15x15 그리드 기반 정상 흐름장
# EMA 학습 + 이중 선형 보간(Bilinear Interpolation) + 방향 일관성 공간 평활화
# smoothed_mask: 보간으로 채워진 셀(실 데이터 없음) 추적 — judge.py에서 cos_threshold 완화에 사용

import numpy as np                                       # 수치 계산
from collections import deque                             # BFS 큐 (flood-fill용)
from pathlib import Path                                  # 경로 조작


class FlowMap:
    def __init__(self, grid_size: int, alpha: float, min_samples: int):
        self.grid_size = grid_size                        # 흐름 맵을 나눌 격자 크기 (N x N)
        self.alpha = alpha                                # EMA 학습 속도 (새 데이터 반영 비율)
        self.min_samples = min_samples                    # 셀당 최소 학습 샘플 수 (이하이면 공간 보정)

        # 각 셀의 정상 이동 방향 벡터 (ndx, ndy)
        self.flow = np.zeros((grid_size, grid_size, 2), np.float32)
        # 각 셀의 학습 데이터 개수
        self.count = np.zeros((grid_size, grid_size), np.int32)

        self.frame_w = 0                                  # 영상 너비
        self.frame_h = 0                                  # 영상 높이
        self.cell_w = 1.0                                 # 셀 하나의 너비
        self.cell_h = 1.0                                 # 셀 하나의 높이

        # Phase 1 정체 탐지용 — 셀별 정상 normalized_speed EMA
        self.speed_ref = np.zeros((grid_size, grid_size), np.float32)  # 셀별 정상 norm_speed 기준값

        # apply_boundary_erosion()이 제거한 셀 기록 — learn_step에서 재학습 금지
        self.eroded_mask = np.zeros((grid_size, grid_size), dtype=bool)  # True=영구 빈 셀

        # ── 개선 1: smoothed_mask — 보간으로 채워진 셀 추적 ──────────────
        # apply_spatial_smoothing()에서 count=0 셀이 이웃 평균으로 채워지면 True
        # learn_step()에서 실 데이터가 들어오면 False로 해제
        # judge.py에서 이 마스크가 True인 셀은 cos_threshold를 완화하여 오탐 방지
        self.smoothed_mask = np.zeros((grid_size, grid_size), dtype=bool)  # True=보간 채움, 실 데이터 없음

        self._learn_call_count = 0                        # learn_step 호출 횟수 (디버그용)

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
        self._learn_call_count = 0                        # 호출 카운터 초기화

    # ==================== 좌표 변환 ====================
    def _cell_coords(self, x, y):
        """픽셀 좌표(x, y)를 그리드 좌표(r, c)로 변환"""
        r = (y / self.cell_h) - 0.5                       # 행 인덱스 실수 값 (0~grid_size-1 부근)
        c = (x / self.cell_w) - 0.5                       # 열 인덱스 실수 값
        return r, c

    # ==================== EMA 기반 학습 ====================
    def learn_step(self, x1, y1, x2, y2, min_move):
        """한 차량의 이동 벡터를 flow_map에 반영 (EMA 기반 학습)"""

        dx, dy = x2 - x1, y2 - y1                        # 이동 벡터
        mag = np.sqrt(dx ** 2 + dy ** 2)                  # 크기(속도)
        if mag < min_move:                                # 너무 작은 움직임은 무시
            return

        ndx, ndy = dx / mag, dy / mag                     # 단위 방향 벡터
        r = int((y1 + y2) / 2 / self.cell_h)             # 중간 위치의 셀 행 좌표
        c = int((x1 + x2) / 2 / self.cell_w)             # 중간 위치의 셀 열 좌표
        r = np.clip(r, 0, self.grid_size - 1)             # 범위 보정
        c = np.clip(c, 0, self.grid_size - 1)

        # ── 경계 마스크: erosion으로 제거된 셀은 영구 재학습 금지 ────────
        # apply_boundary_erosion()이 설정한 셀에는 enable_online_flow_update=True
        # 상태에서도 차량이 지나가며 다시 채우지 못하도록 완전 차단.
        if self.eroded_mask[r, c]:                        # 제거된 경계 셀이면
            return                                        # 학습 거부

        # ── 방향 게이팅: 확립된 셀에 반대 방향 진입 차단 ────────────────
        # count >= min_samples인 셀은 이미 방향이 확립된 것으로 간주.
        # 새 벡터가 기존 방향과 cos < -0.4 (120° 이상 반대)이면 오염으로 판정,
        # EMA 갱신을 거부한다. 중앙 분리대 경계에서 상행 차량이 하행 셀을
        # 오염시키는 것을 원천 차단하는 핵심 로직.
        if self.count[r, c] >= self.min_samples:          # 충분히 학습된 셀만 검사
            existing = self.flow[r, c]                    # 기존 방향 벡터
            emag = np.linalg.norm(existing)               # 기존 벡터 크기
            if emag > 0.1:                                # 기존 방향이 유효하면
                cos_val = float(                          # 기존 방향과의 코사인 유사도
                    ndx * existing[0] / emag
                    + ndy * existing[1] / emag
                )
                if cos_val < -0.4:                        # 120° 이상 반대 → 오염 시도
                    return                                # EMA 갱신 거부

        # EMA: 기존 흐름 벡터에 새 방향을 비율(alpha)만큼 섞어줌
        self.flow[r, c, 0] = (1 - self.alpha) * self.flow[r, c, 0] + self.alpha * ndx
        self.flow[r, c, 1] = (1 - self.alpha) * self.flow[r, c, 1] + self.alpha * ndy
        self.count[r, c] += 1                             # 샘플 수 증가

        # ── 개선 1: 실 데이터 유입 시 smoothed_mask 해제 ────────────────
        # 보간으로 채워진 셀에 실제 차량이 통과하면 더 이상 "보간 전용"이 아님
        # → judge.py에서 이 셀은 정상 cos_threshold(-0.75) 적용으로 복귀
        if self.smoothed_mask[r, c]:                      # 보간으로 채워진 셀이면
            self.smoothed_mask[r, c] = False              # 실 데이터 유입 → 보간 표시 해제

        # 디버그: 100회마다 학습 현황 출력
        self._learn_call_count += 1                       # 호출 카운터 증가
        if self._learn_call_count % 100 == 0:             # 100회마다
            active_cells = int(np.sum(self.count > 0))    # 데이터가 있는 셀 수
            total_samples = int(self.count.sum())          # 전체 샘플 수
            angle = np.degrees(np.arctan2(ndy, ndx))      # 마지막 학습 방향 (도)
            print(f"   📈 learn_step #{self._learn_call_count}: "
                  f"cell[{r},{c}] cnt={self.count[r,c]}, "
                  f"angle={angle:+.0f}°, "
                  f"active_cells={active_cells}/{self.grid_size**2}, "
                  f"total={total_samples}")

    # ==================== 이중 선형 보간 ====================
    def get_interpolated(self, x, y):
        """이중 선형 보간으로 (x, y) 위치의 흐름 벡터 추정.

        충돌 감지 없이 순수 이중 선형 보간만 수행한다.
        apply_spatial_smoothing에서 count=0 셀 오염을 차단하므로
        보간 결과가 엉뚱한 방향이 되는 문제가 발생하지 않는다.
        """

        r, c = self._cell_coords(x, y)                   # 그리드 좌표로 변환 (실수)
        r0, c0 = int(np.floor(r)), int(np.floor(c))      # 좌상단 셀 인덱스 (정수)
        r1, c1 = r0 + 1, c0 + 1                          # 우하단 셀 인덱스
        dr, dc = r - r0, c - c0                           # 소수점 부분 (보간 가중치)

        # grid 범위 안으로 인덱스 보정
        gs = self.grid_size - 1                           # 최대 인덱스
        r0, r1 = np.clip([r0, r1], 0, gs)                # 행 범위 보정
        c0, c1 = np.clip([c0, c1], 0, gs)                # 열 범위 보정

        v00 = self.flow[r0, c0]                           # (r0, c0) 셀의 흐름 벡터
        v01 = self.flow[r0, c1]                           # (r0, c1)
        v10 = self.flow[r1, c0]                           # (r1, c0)
        v11 = self.flow[r1, c1]                           # (r1, c1)

        # 순수 이중 선형 보간 (충돌 감지 제거)
        top    = (1 - dc) * v00 + dc * v01               # 위쪽 두 셀 가로 보간
        bottom = (1 - dc) * v10 + dc * v11               # 아래쪽 두 셀 가로 보간
        final_v = (1 - dr) * top + dr * bottom            # 위/아래 세로 보간

        mag = np.linalg.norm(final_v)                     # 최종 벡터 크기
        return final_v / (mag + 1e-6) if mag > 0.1 else None  # 단위 벡터 또는 None

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

    # ==================== 경계 셀 제거 ====================
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
    def save(self, path: Path):
        """학습된 flow_map, count, speed_ref, smoothed_mask를 .npy 파일로 저장.

        Args:
            path: 저장 파일 경로 (.npy).
        """
        path.parent.mkdir(parents=True, exist_ok=True)    # 저장 폴더 생성
        data = {                                           # 저장할 데이터 딕셔너리
            "version":       3,                           # 포맷 버전 (3=baseline 제거)
            "flow":          self.flow,                   # 흐름 벡터 배열
            "count":         self.count,                  # 셀별 샘플 수 배열
            "speed_ref":     self.speed_ref,              # 셀별 정상 속도 배열
            "smoothed_mask": self.smoothed_mask,          # 보간 채움 셀 마스크
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
        print(f"✅ flow_map 로드 ({self.count.sum()} 샘플, ver={version}, "
              f"smoothed={int(np.sum(self.smoothed_mask))}셀)")
        return True                                       # 로드 성공
