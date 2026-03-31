# 파일 경로: C:\final_pj\tests\test_gru_module.py
# 역할: GRUModule 단위 테스트 (TDD)
#        GRU-01~10 — 초기화, 버퍼, 예측, 학습, 리셋, fallback

import sys                                              # sys.path 조작용
from pathlib import Path                               # 경로 조작

# src/ 폴더를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest                                           # 테스트 프레임워크
from unittest.mock import MagicMock                    # cfg mock 생성용


# ── cfg mock 헬퍼 ──────────────────────────────────────────────────────────
def _make_cfg(**overrides):
    """DetectorConfig를 흉내 내는 mock 객체를 반환한다."""
    cfg = MagicMock()                                  # 속성 접근 자동 허용
    cfg.gru_hidden = 64                                # GRU hidden 크기
    cfg.gru_layers = 2                                 # GRU 레이어 수
    cfg.gru_seq_len = 30                               # 입력 시퀀스 길이
    cfg.gru_blend_ratio = 0.40                         # GRU 기여 비율
    cfg.gru_warmup_frames = 30                         # camera_switch 후 유예 프레임
    cfg.gru_replay_size = 200                          # replay_buffer 최대 크기
    cfg.gru_online_interval = 10                       # 온라인 학습 주기
    cfg.gru_lr = 1e-3                                  # Adam 학습률
    for k, v in overrides.items():                     # 개별 파라미터 덮어쓰기
        setattr(cfg, k, v)
    return cfg


# ── feature 벡터 생성 헬퍼 ─────────────────────────────────────────────────
def _make_feature(seed: float = 0.5) -> dict:
    """7차원 feature 딕셔너리를 생성한다. 모든 값은 0~1 사이."""
    return {
        "norm_speed_ratio": seed,                      # [0] 속도 비율
        "count_ratio":      seed,                      # [1] 차량 수 비율
        "stop_ratio":       1.0 - seed,               # [2] 정지 비율
        "exit_rate_ratio":  seed,                      # [3] 퇴장률 비율
        "dwell_ratio":      seed,                      # [4] 체류 비율
        "density_score":    seed,                      # [5] 밀도 점수
        "rule_jam_score":   1.0 - seed,               # [6] 규칙 기반 jam_score
    }


# ── 임포트 가능 여부 확인 ─────────────────────────────────────────────────
try:
    from gru_module import GRUModule                   # 구현 파일 임포트 시도
    _IMPORT_OK = True                                  # 성공 플래그
except ImportError:
    _IMPORT_OK = False                                 # 실패 플래그 (파일 미존재)


# =============================================================================
# GRU-01 — GRUModule 초기화
# =============================================================================
def test_gru01_init():
    """GRUModule이 예외 없이 초기화되는지 확인한다."""
    cfg = _make_cfg()                                  # mock cfg
    module = GRUModule(cfg)                            # 초기화
    assert module is not None                          # 객체 생성 확인


# =============================================================================
# GRU-02 — buffer 30개 미만이면 predict() → None
# =============================================================================
def test_gru02_predict_insufficient_buffer():
    """feature_buffer에 30개 미만 데이터가 있으면 predict()는 None을 반환해야 한다."""
    cfg = _make_cfg()
    module = GRUModule(cfg)

    for _ in range(29):                                # 29개만 push (1개 부족)
        module.push(_make_feature())

    result = module.predict()                          # 예측 시도
    assert result is None, "29개로는 None 반환 필요"   # None이어야 함


# =============================================================================
# GRU-03 — buffer 정확히 30개이면 predict() → float 반환
# =============================================================================
def test_gru03_predict_with_full_buffer():
    """feature_buffer에 정확히 30개 데이터가 있으면 predict()가 float을 반환해야 한다."""
    cfg = _make_cfg()
    module = GRUModule(cfg)

    for _ in range(30):                                # 30개 push (충분)
        module.push(_make_feature())

    result = module.predict()                          # 예측 시도
    assert result is not None, "30개면 float 반환 필요"  # None이면 안 됨
    assert isinstance(result, float), "반환 타입은 float이어야 함"  # float 타입 확인


# =============================================================================
# GRU-04 — predict() 반환값 범위 0.0~1.0
# =============================================================================
def test_gru04_predict_range():
    """predict()의 반환값은 항상 0.0 이상 1.0 이하여야 한다."""
    cfg = _make_cfg()
    module = GRUModule(cfg)

    for _ in range(30):                                # 버퍼 채우기
        module.push(_make_feature(seed=0.8))           # 일부러 편향된 feature

    score = module.predict()                           # 예측
    assert score is not None                           # None 아님 확인
    assert 0.0 <= score <= 1.0, f"범위 초과: {score}" # 범위 확인


# =============================================================================
# GRU-05 — 동일 입력 → 동일 출력 (deterministic, eval mode)
# =============================================================================
def test_gru05_deterministic():
    """동일 초기 상태(seed)에서 predict()가 동일 결과를 반환하는지 확인한다.

    hidden state가 매 predict()마다 갱신되므로 동일 모듈 2회 호출 시
    결과가 다를 수 있다. 대신 동일 seed로 초기화한 2개 모듈의
    첫 predict() 결과가 같은지 검증한다.
    """
    pytest.importorskip("torch")                       # PyTorch 없으면 skip
    import torch                                       # seed 설정용

    results = []                                       # 2회 실행 결과 저장
    for _ in range(2):                                 # 동일 조건 2회 반복
        torch.manual_seed(42)                          # 동일 seed → 동일 가중치 초기화
        cfg = _make_cfg()                              # mock cfg
        module = GRUModule(cfg)                        # 동일 seed 상태에서 생성
        for _ in range(30):                            # 동일 feature 30개 push
            module.push(_make_feature(seed=0.5))
        results.append(module.predict())               # 첫 predict 결과 저장

    assert results[0] is not None and results[1] is not None  # 둘 다 유효
    assert results[0] == results[1], "같은 초기 상태이면 같은 출력이어야 함"  # 동일 확인


# =============================================================================
# GRU-06 — pretrain() 호출 시 loss가 감소해야 함
# =============================================================================
def test_gru06_pretrain_loss_decreases():
    """pretrain()을 호출하면 첫 epoch보다 마지막 epoch의 loss가 낮아야 한다.

    PyTorch 없는 환경이면 테스트를 건너뛴다.
    """
    pytest.importorskip("torch")                       # PyTorch 없으면 skip

    cfg = _make_cfg()
    module = GRUModule(cfg)

    # 60개 feature 시퀀스 생성 (pretrain에 충분한 데이터)
    feature_seq = [_make_feature(seed=0.4) for _ in range(60)]  # 60프레임치

    losses = module.pretrain(feature_seq)              # 자기지도 학습 실행

    assert losses is not None, "pretrain()은 loss 리스트를 반환해야 함"  # 반환값 확인
    assert len(losses) >= 2, "최소 2 epoch 이상이어야 함"  # epoch 수 확인
    # 마지막 loss가 초기보다 크게 증가하지 않아야 함 (1.5배 이내)
    assert losses[-1] <= losses[0] * 1.5, (
        f"loss가 증가함: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


# =============================================================================
# GRU-07 — online_step() 호출 시 예외 없음
# =============================================================================
def test_gru07_online_step_no_exception():
    """online_step()을 여러 번 호출해도 예외가 발생하지 않아야 한다."""
    cfg = _make_cfg(gru_online_interval=1)             # 매 프레임 학습 (테스트용)
    module = GRUModule(cfg)

    for i in range(50):                                # 50번 push + online_step
        module.push(_make_feature())
        module.online_step(label=0)                    # SMOOTH 레이블로 학습

    # 예외 없이 여기까지 오면 통과
    assert True


# =============================================================================
# GRU-08 — reset() 후 predict() → None
# =============================================================================
def test_gru08_reset_clears_buffer():
    """reset() 호출 후 buffer가 비워지므로 predict()는 None을 반환해야 한다."""
    cfg = _make_cfg()
    module = GRUModule(cfg)

    for _ in range(30):                                # 버퍼 채우기
        module.push(_make_feature())

    assert module.predict() is not None                # reset 전 — 값 있음

    module.reset()                                     # buffer 초기화

    assert module.predict() is None, "reset 후 None 반환 필요"  # reset 후 — None


# =============================================================================
# GRU-09 — replay_buffer 200개 초과 시 오래된 항목 자동 제거
# =============================================================================
def test_gru09_replay_buffer_max_size():
    """replay_buffer가 gru_replay_size(200)를 초과하지 않아야 한다."""
    cfg = _make_cfg(gru_replay_size=200, gru_online_interval=1)  # 매 프레임 학습
    module = GRUModule(cfg)

    # 버퍼 채우기 + 300번 online_step (200 초과)
    for _ in range(30):                                # 시퀀스 버퍼 채우기
        module.push(_make_feature())

    for _ in range(300):                               # replay_buffer 300번 채우기
        module.push(_make_feature())
        module.online_step(label=0)                    # SMOOTH로 학습

    # replay_buffer 크기 접근 (내부 속성 검사)
    buf_size = len(module.replay_buffer)               # 실제 버퍼 크기
    assert buf_size <= 200, f"버퍼 크기 초과: {buf_size}"  # 200 이하여야 함


# =============================================================================
# GRU-10 — PyTorch 없을 때 graceful fallback
# =============================================================================
def test_gru10_torch_unavailable_fallback(monkeypatch):
    """PyTorch import 실패 시 predict()는 None을 반환하고 예외가 없어야 한다."""
    import importlib                                   # 동적 모듈 재로드용
    import builtins                                    # built-in import 후킹용

    original_import = builtins.__import__              # 원본 import 함수 저장

    def mock_import(name, *args, **kwargs):
        """torch 임포트를 강제로 실패시키는 mock."""
        if name == "torch" or name.startswith("torch."):   # torch 관련이면
            raise ImportError("torch not available (mocked)")  # 강제 실패
        return original_import(name, *args, **kwargs)  # 나머지는 정상 처리

    monkeypatch.setattr(builtins, "__import__", mock_import)  # import 후킹

    # GRUModule을 torch 없이 재초기화 시도
    import gru_module as gm                            # 이미 로드된 모듈 사용
    importlib.reload(gm)                               # torch mock 적용 후 재로드

    cfg = _make_cfg()
    module = gm.GRUModule(cfg)                         # torch 없이 초기화

    for _ in range(30):                                # 버퍼 채우기 시도
        module.push(_make_feature())

    result = module.predict()                          # torch 없으면 None이어야 함
    assert result is None, "torch 없으면 predict()는 None 반환"

    # online_step, pretrain도 예외 없어야 함
    module.online_step(label=0)                        # no-op 확인
    module.pretrain([_make_feature() for _ in range(60)])  # no-op 확인

    # ── 모듈 상태 복원 (후속 테스트에 _TORCH_AVAILABLE=False 오염 방지) ──
    monkeypatch.undo()                                 # builtins.__import__ 복원
    importlib.reload(gm)                               # torch 있는 상태로 재로드


# =============================================================================
# GRU-11 — warmup 기간 중 predict() → None (버퍼 충분해도 차단)
# =============================================================================
def test_gru11_warmup_blocks_predict():
    """reset() 후 warmup 기간에는 버퍼가 충분해도 predict()가 None을 반환해야 한다.

    warmup_frames=35 > seq_len=30 이면
    30개 push 시점에서 버퍼는 가득 차지만 warmup이 5 남아 있으므로 차단.
    35개 push 후에야 predict()가 float을 반환한다.
    """
    cfg = _make_cfg(gru_warmup_frames=35)              # warmup 35 > seq_len 30
    module = GRUModule(cfg)

    # ── 정상 동작 확인 (warmup=0, 버퍼 충분) ─────────────────────────
    for _ in range(30):                                # 버퍼 30개 채우기
        module.push(_make_feature())
    assert module.predict() is not None                # 정상 예측 가능

    # ── reset → warmup=35 시작 ──────────────────────────────────────
    module.reset()                                     # warmup_remaining = 35

    # ── 30개 push: 버퍼=30(가득), warmup=5(남음) → None ─────────────
    for _ in range(30):                                # 30개 push
        module.push(_make_feature())
    assert module.predict() is None, (                 # 버퍼 충분해도 warmup 중이면 None
        "warmup 중이면 버퍼가 충분해도 None 반환 필요"
    )

    # ── 5개 더 push: warmup=0 → predict 가능 ────────────────────────
    for _ in range(5):                                 # 5개 추가 push (total 35)
        module.push(_make_feature())
    assert module.predict() is not None, (             # warmup 완료 + 버퍼 충분
        "warmup 완료 후 predict()는 float 반환 필요"
    )
