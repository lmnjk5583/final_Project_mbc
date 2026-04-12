# 파일 경로: C:\final_pj\src\gru_module.py
# 역할: Phase 2 GRU 신경망 모듈
#        - 7차원 feature 벡터 시퀀스(T=30)를 입력받아 gru_score(0~1)를 예측
#        - 초기 자기지도 학습 (학습 구간 완료 후 1회)
#        - 온라인 학습 (SMOOTH 구간 매 gru_online_interval 프레임마다)
#        - camera_switch 후 reset() 으로 버퍼·hidden state 초기화
#        - PyTorch 없는 환경 → graceful fallback (predict=None, 학습=no-op)

import collections                                     # deque — 고정 크기 버퍼
import random                                          # replay_buffer 샘플링
import pickle                                          # feature 로그 직렬화 (디스크 저장)

# ── PyTorch graceful import ───────────────────────────────────────────────
try:
    import torch                                       # PyTorch 코어
    import torch.nn as nn                              # 신경망 레이어
    import torch.optim as optim                        # 옵티마이저
    _TORCH_AVAILABLE = True                            # PyTorch 사용 가능 플래그
except ImportError:
    _TORCH_AVAILABLE = False                           # PyTorch 없음 → fallback 모드


# ── feature 딕셔너리 → Tensor 변환 키 순서 ──────────────────────────────
# feature_extractor.py의 반환 딕셔너리와 반드시 일치해야 한다.
# 순서가 바뀌면 학습·예측 불일치 발생 — 키 추가/변경 시 저장된 .pt 삭제 후 재학습 필요
_FEATURE_KEYS = [                                      # 7차원 벡터 고정 순서
    "norm_speed_ratio",                                # [0] 속도 비율 (자기보정 baseline 기준)
    "stop_ratio",                                      # [1] 정지 비율 (nm_history 전체 집계)
    "slow_ratio",                                      # [2] 서행 비율 (nm_history 전체 집계)
    "flow_occupancy",                                  # [3] 차량 점유 셀 / 유효 셀 (순간 밀도)
    "cell_dwell_score",                                # [4] 셀 누적 점유 EMA (핵심 정체 신호)
    "cell_persistence",                                # [5] 30프레임 전·현재 점유셀 Jaccard
    "rule_jam_score",                                  # [6] rule 기반 jam_score (CJ 역주입)
]
_FEATURE_DIM = len(_FEATURE_KEYS)                      # 7 — 고정 입력 차원


# =============================================================================
# _GRUNet — PyTorch 신경망 (GRU + FC 헤드)
# =============================================================================
if _TORCH_AVAILABLE:                                   # PyTorch 있을 때만 정의

    class _GRUNet(nn.Module):
        """GRU 2계층 + 현재 분류 헤드 + Direct 미래 예측 헤드.

        입력: (batch, seq_len=30, input_dim=7) — 프레임 시퀀스
        출력:
          - forward(): (batch, 3) softmax 확률 — 현재 상태 분류
          - direct_heads[i]: 각 예측 시점(1분·3분·5분)의 (batch, 3) 로짓
        """

        def __init__(self, input_dim: int, hidden: int, layers: int,
                     n_horizons: int = 3):
            """신경망 초기화.

            Args:
                input_dim: 입력 feature 차원 (7).
                hidden: GRU hidden state 크기 (64).
                layers: GRU 레이어 수 (2).
                n_horizons: direct 예측 시점 수 (기본 3 — 1·3·5분).
            """
            super().__init__()                         # nn.Module 초기화
            self.gru = nn.GRU(                         # GRU 레이어
                input_size=input_dim,                  # 입력 차원 (7)
                hidden_size=hidden,                    # hidden 크기 (64)
                num_layers=layers,                     # 레이어 수 (2)
                batch_first=True,                      # (batch, seq, feature) 순서
                dropout=0.1 if layers > 1 else 0.0,   # 다층일 때 dropout 적용
            )
            self.fc1 = nn.Linear(hidden, 32)           # FC: hidden(64) → 32
            self.relu = nn.ReLU()                      # 활성화 함수
            self.fc2 = nn.Linear(32, 3)                # FC: 32 → 3클래스 (현재 상태)
            self.softmax = nn.Softmax(dim=-1)          # 확률 분포 정규화
            self.pred_head = nn.Linear(hidden, input_dim)  # 자기회귀 헤드: hidden → feature(7)

            # ── Direct 미래 예측 헤드 (horizon별 독립 분류기) ────────────
            # 각 헤드: hidden(64) → 32 → ReLU → 3클래스 (SMOOTH/SLOW/CONGESTED)
            # 오차 누적 없이 현재 hidden state에서 N분 후를 직접 예측
            self.direct_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, 32),             # hidden → 32
                    nn.ReLU(),                         # 활성화
                    nn.Linear(32, 3),                  # 32 → 3클래스
                )
                for _ in range(n_horizons)             # horizon 수만큼 독립 헤드 생성
            ])

        def forward(self, x, h=None):
            """순전파 계산 (현재 상태 분류).

            Args:
                x: (batch, seq_len, input_dim) 입력 텐서.
                h: (layers, batch, hidden) 초기 hidden state. None이면 zeros.

            Returns:
                probs: (batch, 3) softmax 확률.
                h_new: 갱신된 hidden state.
            """
            out, h_new = self.gru(x, h)                # GRU 순전파 → (batch, seq, hidden)
            last = out[:, -1, :]                       # 마지막 타임스텝 출력 (batch, hidden)
            x2 = self.relu(self.fc1(last))             # FC1 + ReLU → (batch, 32)
            logits = self.fc2(x2)                      # FC2 → (batch, 3) 로짓
            probs = self.softmax(logits)               # Softmax → 확률 (batch, 3)
            return probs, h_new                        # 확률 + 갱신 hidden 반환


# =============================================================================
# GRUModule — 공개 인터페이스
# =============================================================================

class GRUModule:
    """Phase 2 GRU 예측 모듈.

    Parameters
    ----------
    cfg : DetectorConfig
        gru_hidden, gru_layers, gru_seq_len, gru_blend_ratio,
        gru_warmup_frames, gru_replay_size, gru_online_interval, gru_lr.
    """

    def __init__(self, cfg, fps: float = 30.0):
        """GRUModule 초기화.

        PyTorch 없는 환경이면 fallback 모드로 동작한다.
        모든 메서드는 호출 가능하지만 학습/예측은 no-op이 된다.

        Args:
            cfg: DetectorConfig.
            fps: 영상 FPS — horizon 초→프레임 변환, pretrain 최소 데이터 계산용.
        """
        self.cfg = cfg                                 # 설정 저장
        self._fps = fps                                # FPS 저장
        self._torch_ok = _TORCH_AVAILABLE              # PyTorch 사용 가능 여부

        # ── Direct 예측 horizon 계산 (초 → 프레임) ───────────────────
        _horizons_sec = getattr(cfg, "gru_predict_horizons_sec", (60, 180, 300))
        self._horizon_frames = [max(1, int(s * fps)) for s in _horizons_sec]  # 초→프레임
        self._horizon_secs = list(_horizons_sec)       # 초 단위 저장 (결과 반환용)

        # pretrain 최소 데이터: 최소 10분치 feature 쌓인 후 시작
        _pretrain_min_sec = getattr(cfg, "gru_pretrain_min_sec", 600.0)
        self._pretrain_min_frames = max(              # pretrain 트리거 최소 프레임 수
            cfg.gru_seq_len + max(self._horizon_frames),  # 최소: seq_len + 최대 horizon
            int(_pretrain_min_sec * fps)              # 설정값 (기본 10분)
        )

        self._is_direct_trained = False               # direct head 학습 완료 여부

        # ── 고정 크기 feature 시퀀스 버퍼 ──────────────────────────────
        self._feature_buffer = collections.deque(      # 최근 gru_seq_len개만 유지
            maxlen=cfg.gru_seq_len                     # 기본 30
        )

        # ── replay_buffer: 온라인 학습 데이터 저장 ──────────────────────
        # (feature_seq, label) 튜플 저장
        self.replay_buffer = collections.deque(        # 최대 gru_replay_size개
            maxlen=cfg.gru_replay_size                 # 기본 200
        )

        # ── warmup 카운터 ────────────────────────────────────────────────
        self._warmup_remaining = 0                     # camera_switch 후 유예 카운터

        # ── 온라인 학습 step 카운터 ──────────────────────────────────────
        self._step_count = 0                           # push() 호출 횟수

        self._is_pretrained = False                        # pretrain/load 완료 여부 — False면 blend 비활성

        if not self._torch_ok:                         # PyTorch 없으면 여기서 종료
            self._net = None                           # 신경망 None
            self._optimizer = None                     # 옵티마이저 None
            self._hidden = None                        # hidden state None
            return                                     # fallback 모드

        # ── PyTorch 있을 때만 신경망 초기화 ─────────────────────────────
        self._net = _GRUNet(                           # 신경망 생성
            input_dim=_FEATURE_DIM,                    # 7
            hidden=cfg.gru_hidden,                     # 64
            layers=cfg.gru_layers,                     # 2
            n_horizons=len(self._horizon_frames),       # direct 예측 헤드 수 (기본 3)
        )
        self._net.eval()                               # eval 모드 (dropout 비활성)

        self._optimizer = optim.Adam(                  # Adam 옵티마이저
            self._net.parameters(),                    # 모든 파라미터
            lr=cfg.gru_lr                              # 학습률 (기본 1e-3)
        )

        self._hidden = None                            # GRU hidden state (None=zeros)

    # ── feature 딕셔너리 → 리스트 변환 ──────────────────────────────────
    @staticmethod
    def _dict_to_vec(x_t: dict) -> list:
        """feature 딕셔너리를 고정 순서의 float 리스트로 변환한다.

        Args:
            x_t: {"norm_speed_ratio": ..., ..., "rule_jam_score": ...}

        Returns:
            7개 float 리스트 (고정 순서 _FEATURE_KEYS 기준).
        """
        return [float(x_t.get(k, 0.0)) for k in _FEATURE_KEYS]  # 키 순서대로 추출

    # ── push: 매 프레임 호출 ─────────────────────────────────────────────
    def push(self, x_t: dict):
        """feature 벡터를 시퀀스 버퍼에 추가한다.

        Args:
            x_t: 7차원 feature 딕셔너리.
        """
        vec = self._dict_to_vec(x_t)                  # dict → float 리스트
        self._feature_buffer.append(vec)              # 버퍼에 추가 (maxlen 초과 시 자동 제거)
        self._step_count += 1                         # step 카운터 증가

        if self._warmup_remaining > 0:                 # warmup 기간이면
            self._warmup_remaining -= 1                # 카운터 차감

    # ── predict: gru_score 예측 ──────────────────────────────────────────
    def predict(self) -> float | None:
        """현재 feature_buffer로 gru_score를 예측한다.

        Returns:
            gru_score (0.0~1.0) — 버퍼 부족·warmup·PyTorch 없으면 None.
        """
        if not self._torch_ok:                         # PyTorch 없음 → fallback
            return None

        if self._net is None:                          # 신경망 미초기화
            return None

        if not self._is_pretrained:                    # pretrain/load 미완료 → 랜덤 weights로 blend 금지
            return None

        if len(self._feature_buffer) < self.cfg.gru_seq_len:  # 버퍼 부족
            return None

        if self._warmup_remaining > 0:                 # warmup 기간 중
            return None

        # ── Tensor 변환 ────────────────────────────────────────────────
        seq = list(self._feature_buffer)               # deque → list (seq_len × 7)
        x = torch.tensor(                              # float32 텐서 변환
            [seq],                                     # (1, seq_len, 7) batch=1
            dtype=torch.float32
        )

        # ── eval mode 예측 (gradient 계산 없음) ────────────────────────
        self._net.eval()                               # eval 모드 확인
        with torch.no_grad():                          # gradient 비활성화
            probs, self._hidden = self._net(x, self._hidden)  # (1, 3) 확률 + hidden 갱신

        # ── gru_score: p_slow × 0.5 + p_congested × 1.0 ──────────────
        p = probs[0]                                   # (3,) — [p_smooth, p_slow, p_congested]
        gru_score = float(p[1] * 0.5 + p[2] * 1.0)   # 가중 합산
        gru_score = float(max(0.0, min(1.0, gru_score)))  # clip 0~1

        return gru_score                               # 예측값 반환

    # ── predict_direct: N분 후 직접 예측 ────────────────────────────────
    def predict_direct(self) -> list | None:
        """현재 관측 윈도우에서 1·3·5분 후 정체 레벨을 직접 예측한다.

        자기회귀 롤아웃과 달리 오차 누적이 없다.
        GRU hidden state에서 각 horizon 헤드가 독립적으로 3클래스 확률을 출력한다.

        Returns:
            [{"horizon_sec": 60, "horizon_min": 1,
              "p_smooth": 0.1, "p_slow": 0.6, "p_congested": 0.3,
              "predicted_level": "SLOW"}, ...]
            학습 미완료·버퍼 부족·PyTorch 없으면 None.
        """
        if not self._torch_ok or self._net is None:    # PyTorch 없음
            return None
        if not self._is_direct_trained:                # direct head 미학습
            return None
        if len(self._feature_buffer) < self.cfg.gru_seq_len:  # 버퍼 부족
            return None
        if self._warmup_remaining > 0:                 # warmup 기간 중
            return None

        seq = list(self._feature_buffer)               # 현재 시퀀스 복사
        x = torch.tensor([seq], dtype=torch.float32)   # (1, seq_len, 7)

        self._net.eval()
        results = []
        with torch.no_grad():
            out, _ = self._net.gru(x)                  # GRU 순전파: (1, seq_len, hidden)
            last = out[:, -1, :]                       # 마지막 타임스텝: (1, hidden)

            for h_i, (horizon_f, horizon_s) in enumerate(
                zip(self._horizon_frames, self._horizon_secs)
            ):
                logits = self._net.direct_heads[h_i](last)          # (1, 3) 로짓
                probs = torch.softmax(logits, dim=-1)[0]             # (3,) 확률
                pred_idx = int(probs.argmax())                       # 최대 확률 클래스
                results.append({
                    "horizon_sec":      horizon_s,                   # 예측 목표 (초)
                    "horizon_min":      horizon_s // 60,             # 예측 목표 (분)
                    "p_smooth":         float(probs[0]),             # SMOOTH 확률
                    "p_slow":           float(probs[1]),             # SLOW 확률
                    "p_congested":      float(probs[2]),             # CONGESTED 확률
                    "predicted_level":  ["SMOOTH", "SLOW", "CONGESTED"][pred_idx],
                    "confidence":       float(probs[pred_idx]),      # 최고 확률값
                })

        return results                                 # 전체 horizon 예측 결과 반환

    # ── predict_future: N스텝 자기회귀 롤아웃 ───────────────────────────
    def predict_future(self, steps: int | None = None) -> list | None:
        """N스텝 미래 정체 상태를 자기회귀 롤아웃으로 예측한다.

        현재 feature_buffer → GRU 통과 → pred_head로 x_{t+1} 예측 →
        예측값을 다시 GRU 입력으로 반복하여 미래 상태를 추정한다.
        실제 _hidden/_feature_buffer는 변경하지 않는다 (예측 전용).

        Args:
            steps: 예측 스텝 수 (프레임). None이면 cfg.gru_forecast_steps.

        Returns:
            [{"step": 1, "p_smooth": ..., "p_slow": ..., "p_congested": ...,
              "gru_score": ...}, ...]
            버퍼 부족·warmup·PyTorch 없으면 None.
        """
        if not self._torch_ok or self._net is None:    # PyTorch 없음 → fallback
            return None                                # 예측 불가

        if len(self._feature_buffer) < self.cfg.gru_seq_len:  # 시퀀스 버퍼 부족
            return None                                # 예측 불가

        if self._warmup_remaining > 0:                 # warmup 기간 중
            return None                                # 예측 불가

        if steps is None:                              # 기본 스텝 수 사용
            steps = self.cfg.gru_forecast_steps         # config에서 읽기

        # ── 현재 버퍼 → Tensor 변환 (실제 상태 변경 없음) ──────────────
        seq = list(self._feature_buffer)               # deque → list 복사
        x = torch.tensor([seq], dtype=torch.float32)   # (1, seq_len, 7)

        self._net.eval()                               # eval 모드 확인
        results = []                                   # 예측 결과 리스트

        with torch.no_grad():                          # gradient 비활성화
            # ── 현재 시퀀스로 초기 hidden state 획득 ──────────────────
            gru_out, h = self._net.gru(x)              # (1, seq_len, 64), (layers, 1, 64)
            cur_out = gru_out[:, -1, :]                # (1, 64) — 마지막 타임스텝 출력

            for step_i in range(1, steps + 1):         # 1 ~ steps 반복
                # ── 분류 확률 계산 (현재 스텝) ──────────────────────
                fc_out = self._net.relu(               # FC1 + ReLU
                    self._net.fc1(cur_out)             # (1, 32)
                )
                logits = self._net.fc2(fc_out)         # (1, 3) 로짓
                probs = self._net.softmax(logits)      # (1, 3) softmax 확률
                p = probs[0]                           # (3,) — [smooth, slow, congested]
                gru_score = float(                     # 가중 합산 gru_score
                    p[1] * 0.5 + p[2] * 1.0            # p_slow×0.5 + p_congested×1.0
                )

                results.append({                       # 스텝 결과 저장
                    "step": step_i,                    # 예측 스텝 번호
                    "p_smooth": float(p[0]),           # SMOOTH 확률
                    "p_slow": float(p[1]),             # SLOW 확률
                    "p_congested": float(p[2]),        # CONGESTED 확률
                    "gru_score": max(0.0, min(1.0, gru_score)),  # clip 0~1
                })

                # ── 다음 feature 예측 (자기회귀) ───────────────────
                pred_feat = self._net.pred_head(cur_out)  # (1, 7) — 다음 프레임 feature
                next_input = pred_feat.unsqueeze(1)    # (1, 1, 7) — GRU 입력 형태
                gru_step_out, h = self._net.gru(       # 단일 타임스텝 GRU 순전파
                    next_input, h                      # 예측 feature + 현재 hidden
                )
                cur_out = gru_step_out[:, -1, :]       # (1, 64) — 다음 루프용 출력

        return results                                 # 전체 예측 리스트 반환

    # ── pretrain: 학습 구간 완료 후 자기지도 학습 ─────────────────────────
    def pretrain(self, feature_sequence: list) -> list | None:
        """feature 시퀀스로 x_{t+1} 예측 자기지도 학습을 수행한다.

        학습 구간 완료 직후 1회 호출한다.

        Args:
            feature_sequence: [feature_dict, ...] — 학습 구간 전체 feature 목록.

        Returns:
            epoch별 loss 리스트 (PyTorch 없으면 None).
        """
        if not self._torch_ok or self._net is None:    # fallback 모드
            return None

        if len(feature_sequence) < self.cfg.gru_seq_len + 1:  # 데이터 부족
            return None                                # 학습 불가

        # ── 슬라이딩 윈도우로 (input, target) 쌍 생성 ─────────────────
        vecs = [self._dict_to_vec(f) for f in feature_sequence]  # dict → vec 변환
        inputs, targets = [], []                       # 입력·타겟 리스트
        seq_len = self.cfg.gru_seq_len                 # 시퀀스 길이 (30)

        for i in range(len(vecs) - seq_len):           # 가능한 모든 슬라이딩 윈도우
            inputs.append(vecs[i: i + seq_len])        # 입력: t~t+29
            targets.append(vecs[i + seq_len])          # 타겟: t+30 (다음 프레임)

        if not inputs:                                 # 윈도우가 없으면
            return None                                # 학습 불가

        # ── Tensor 변환 ────────────────────────────────────────────────
        X = torch.tensor(inputs, dtype=torch.float32)  # (N, seq_len, 7)
        Y = torch.tensor(targets, dtype=torch.float32) # (N, 7) — 다음 타임스텝

        # ── MSE Loss로 자기지도 학습 (1~2 epoch) ──────────────────────
        criterion = nn.MSELoss()                       # 평균 제곱 오차 손실
        self._net.train()                              # train 모드
        epoch_losses = []                              # epoch별 loss 기록

        for epoch in range(2):                         # 2 epoch
            self._optimizer.zero_grad()                # gradient 초기화
            out, _ = self._net.gru(X)                  # GRU 출력: (N, seq_len, hidden)
            last_out = out[:, -1, :]                   # 마지막 타임스텝: (N, hidden)
            # pred_head(64→7)로 다음 프레임 feature 예측 — GRU + pred_head 가중치 갱신
            pred = self._net.pred_head(last_out)       # (N, 7) — 정식 예측 헤드 사용
            loss = criterion(pred, Y)                  # MSE 손실 계산
            loss.backward()                            # 역전파
            self._optimizer.step()                     # 파라미터 갱신
            epoch_losses.append(float(loss.item()))    # loss 기록

        self._net.eval()                               # 자기지도 학습 후 eval 모드 복원
        self._is_pretrained = True                     # pretrain 완료 → blend 활성화

        # ── Phase 2: Direct Prediction Head 학습 ──────────────────────
        # 입력: feature_sequence[i:i+seq_len]  (현재 관측 윈도우)
        # 타겟: feature_sequence[i+seq_len+horizon_f]의 rule_jam_score → 3클래스
        # 각 horizon 헤드를 독립적으로 CrossEntropy 학습
        smooth_thr = getattr(self.cfg, "smooth_jam_threshold", 0.25)  # SMOOTH 임계값
        slow_thr   = getattr(self.cfg, "slow_jam_threshold",   0.55)  # SLOW 임계값
        n_epochs   = getattr(self.cfg, "gru_direct_epochs",    10)    # 학습 epoch 수
        criterion_ce = nn.CrossEntropyLoss()           # 분류 손실 함수

        for h_i, horizon_f in enumerate(self._horizon_frames):  # 각 horizon 순회
            # ── (입력, 타겟) 쌍 생성 ──────────────────────────────────
            # i+seq_len+horizon_f 인덱스가 유효한 범위만 사용
            inputs_d, targets_d = [], []
            for i in range(len(vecs) - seq_len - horizon_f):
                future_jam = float(                    # horizon_f 프레임 후 실제 jam_score
                    feature_sequence[i + seq_len + horizon_f].get("rule_jam_score", 0.0)
                )
                # rule_jam_score → 3클래스 레이블 변환
                if future_jam < smooth_thr:
                    label = 0                          # SMOOTH
                elif future_jam < slow_thr:
                    label = 1                          # SLOW
                else:
                    label = 2                          # CONGESTED
                inputs_d.append(vecs[i: i + seq_len])
                targets_d.append(label)

            if not inputs_d:                           # 데이터 부족 — 해당 horizon 스킵
                continue

            X_d = torch.tensor(inputs_d,  dtype=torch.float32)  # (N, seq_len, 7)
            Y_d = torch.tensor(targets_d, dtype=torch.long)      # (N,) 클래스 레이블

            # ── direct head만 학습 (GRU 가중치 함께 갱신) ───────────────
            self._net.train()
            direct_losses = []
            for _ in range(n_epochs):                  # epoch 반복
                self._optimizer.zero_grad()
                out_d, _ = self._net.gru(X_d)          # GRU 출력: (N, seq_len, hidden)
                last_d = out_d[:, -1, :]               # 마지막 타임스텝: (N, hidden)
                logits_d = self._net.direct_heads[h_i](last_d)  # (N, 3) 로짓
                loss_d = criterion_ce(logits_d, Y_d)   # CrossEntropy 손실
                loss_d.backward()
                self._optimizer.step()
                direct_losses.append(float(loss_d.item()))

            self._net.eval()
            h_sec = self._horizon_secs[h_i]            # 초 단위 horizon
            print(f"  └─ {h_sec//60}분 후 헤드: "
                  f"loss {direct_losses[0]:.4f}→{direct_losses[-1]:.4f} "
                  f"({len(inputs_d)}쌍)")              # 학습 결과 출력

        self._is_direct_trained = True                 # direct head 학습 완료
        return epoch_losses                            # 자기지도 phase loss 반환

    # ── save: GRU weights 저장 ───────────────────────────────────────────
    def save(self, path) -> bool:
        """학습된 GRU weights를 디스크에 저장한다.

        Args:
            path: 저장 경로 (Path 또는 str, .pt 확장자 권장).

        Returns:
            True(성공) / False(실패).
        """
        if not self._torch_ok or self._net is None:    # PyTorch 없거나 신경망 미초기화
            return False
        if not self._is_pretrained:                    # pretrain 미완료 — 저장 의미 없음
            return False
        try:
            torch.save(self._net.state_dict(), path)   # state_dict만 저장 (경량)
            return True
        except Exception:
            return False

    # ── load: GRU weights 로드 ───────────────────────────────────────────
    def load(self, path) -> bool:
        """저장된 GRU weights를 로드한다.

        Args:
            path: 로드 경로 (Path 또는 str).

        Returns:
            True(성공) / False(실패 또는 PyTorch 없음).
        """
        if not self._torch_ok or self._net is None:    # PyTorch 없거나 신경망 미초기화
            return False
        try:
            state = torch.load(path, map_location="cpu")  # CPU로 로드 (CUDA 없어도 동작)
            self._net.load_state_dict(state)           # weights 적용
            self._net.eval()                           # eval 모드 복원
            self._is_pretrained = True                 # 로드 성공 → blend 활성화
            return True
        except Exception:
            return False

    # ── online_step: SMOOTH 구간 온라인 학습 ─────────────────────────────
    def online_step(self, label: int):
        """현재 버퍼 상태로 CrossEntropy 온라인 학습을 수행한다.

        gru_online_interval 프레임마다 1회 gradient step 실행.

        Args:
            label: int — SMOOTH=0, SLOW=1, CONGESTED=2.
        """
        if not self._torch_ok or self._net is None:    # fallback 모드
            return                                     # no-op

        if len(self._feature_buffer) < self.cfg.gru_seq_len:  # 버퍼 부족
            return                                     # 학습 불가

        # ── 현재 시퀀스를 replay_buffer에 저장 ─────────────────────────
        seq = list(self._feature_buffer)               # 현재 시퀀스 복사
        self.replay_buffer.append((seq, label))        # (시퀀스, 레이블) 저장

        # ── online_interval 도달 시만 gradient step ──────────────────
        if self._step_count % self.cfg.gru_online_interval != 0:  # 주기 미달
            return                                     # 이번 프레임 학습 생략

        if len(self.replay_buffer) < 4:                # replay 데이터 부족
            return                                     # 최소 4개 필요

        # ── replay_buffer에서 미니배치 샘플링 ─────────────────────────
        batch_size = min(8, len(self.replay_buffer))   # 배치 크기 (최대 8)
        samples = random.sample(list(self.replay_buffer), batch_size)  # 랜덤 샘플
        seqs, labels = zip(*samples)                   # 시퀀스·레이블 분리

        # ── Tensor 변환 ────────────────────────────────────────────────
        X = torch.tensor(list(seqs), dtype=torch.float32)     # (batch, seq_len, 7)
        Y = torch.tensor(list(labels), dtype=torch.long)      # (batch,) 정수 레이블

        # ── CrossEntropy Loss로 분류 학습 ─────────────────────────────
        criterion = nn.CrossEntropyLoss()              # 크로스엔트로피 손실
        self._net.train()                              # train 모드
        self._optimizer.zero_grad()                    # gradient 초기화
        probs, _ = self._net(X)                        # (batch, 3) 예측
        loss = criterion(probs, Y)                     # 손실 계산
        loss.backward()                                # 역전파
        self._optimizer.step()                         # 파라미터 갱신
        self._net.eval()                               # eval 모드 복원

    # ── 누적 로그 저장 (세션 종료 시 호출) ──────────────────────────────
    def append_feature_log(self, features: list, log_path) -> int:
        """새 feature 목록을 디스크 로그에 추가하고 총 누적 개수를 반환한다.

        기존 로그가 있으면 이어 붙이고, 없으면 새로 생성한다.
        세션이 달라져도 누적이 유지되므로 며칠치 데이터가 쌓인다.

        Args:
            features: 이번 세션에서 수집한 feature dict 목록.
            log_path: 로그 파일 경로 (.pkl).

        Returns:
            누적된 총 feature 개수.
        """
        log_path = str(log_path)
        existing = []
        try:
            with open(log_path, "rb") as f:
                existing = pickle.load(f)              # 기존 로그 로드
        except (FileNotFoundError, Exception):
            pass                                       # 없으면 빈 리스트로 시작

        existing.extend(features)                      # 새 데이터 이어붙이기

        try:
            with open(log_path, "wb") as f:
                pickle.dump(existing, f)               # 통합 로그 저장
        except Exception as e:
            print(f"[GRU] feature 로그 저장 실패: {e}")

        return len(existing)                           # 총 누적 개수 반환

    # ── 누적 로그 로드 ────────────────────────────────────────────────
    def load_feature_log(self, log_path) -> list:
        """디스크에서 누적 feature 로그를 로드한다.

        Args:
            log_path: 로그 파일 경로 (.pkl).

        Returns:
            feature dict 목록. 파일 없으면 빈 리스트.
        """
        try:
            with open(str(log_path), "rb") as f:
                data = pickle.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, Exception):
            return []                                  # 로그 없으면 빈 리스트

    # ── 누적 로그 기반 재학습 ─────────────────────────────────────────
    def retrain_from_log(self, log_path) -> bool:
        """디스크 누적 로그 전체를 사용해 GRU를 재학습한다.

        세션이 바뀌어도 로그가 쌓여있으면 점점 정확해진다.
        데이터가 충분하면 (gru_pretrain_min_frames 이상) pretrain()을 호출한다.

        Args:
            log_path: 로그 파일 경로 (.pkl).

        Returns:
            True(재학습 완료) / False(데이터 부족·실패).
        """
        features = self.load_feature_log(log_path)    # 누적 로그 전체 로드
        total = len(features)

        if total < self._pretrain_min_frames:          # 데이터 부족
            print(f"[GRU] 누적 데이터 부족: {total}/{self._pretrain_min_frames}프레임 "
                  f"({total / max(self._fps, 1) / 60:.1f}분 / "
                  f"{self._pretrain_min_frames / max(self._fps, 1) / 60:.1f}분 필요)")
            return False

        print(f"[GRU] 누적 {total}프레임 ({total / max(self._fps, 1) / 60:.1f}분) → 재학습 시작")
        losses = self.pretrain(features)               # 전체 로그로 pretrain
        if losses:
            print(f"[GRU] 재학습 완료: loss {losses[0]:.4f}→{losses[-1]:.4f}")
            return True
        return False

    # ── reset: camera_switch 후 호출 ─────────────────────────────────────
    def reset(self):
        """feature_buffer·hidden_state를 초기화하고 warmup을 시작한다.

        카메라 전환(camera_switch) 감지 시 detector.py에서 호출.
        """
        self._feature_buffer.clear()                   # 시퀀스 버퍼 초기화
        self._hidden = None                            # GRU hidden state 초기화 (zeros)
        self._warmup_remaining = self.cfg.gru_warmup_frames  # warmup 카운터 재설정
        self._step_count = 0                           # step 카운터 초기화
        # replay_buffer는 유지 (이전 학습 데이터 재활용)
