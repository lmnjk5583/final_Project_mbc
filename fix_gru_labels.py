# 파일 경로: C:\final_pj\fix_gru_labels.py
# 역할: feature_log_*.pkl 안의 rule_jam_score를 현재 공식으로 재계산하고
#        gru_*.pt를 삭제해 다음 실행 시 올바른 라벨로 재학습하게 한다.
# 실행: python fix_gru_labels.py
#
# ※ raw feature(cell_dwell_score, flow_occupancy 등)는 보존됨
# ※ gru_*.pt만 삭제됨 (feature_log.pkl은 보존)
# ※ flow_map.npy는 건드리지 않음

import pickle
import sys
from pathlib import Path

# ── 프로젝트 루트 → src 경로 추가 ──────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from congestion_judge import compute_jam_score_fallback  # 현재 공식 임포트


def fix_pkl(pkl_path: Path) -> int:
    """pkl 안의 rule_jam_score를 현재 공식으로 재계산한다.

    Returns:
        재계산된 feature 수 (0이면 파일 없음 또는 빈 파일).
    """
    if not pkl_path.exists():
        print(f"  [스킵] {pkl_path.name} — 파일 없음")
        return 0

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not data:
        print(f"  [스킵] {pkl_path.name} — 데이터 없음")
        return 0

    before_values = [d.get("rule_jam_score", 0.0) for d in data]

    for feat in data:
        new_jam = compute_jam_score_fallback(feat)  # 현재 공식으로 재계산
        feat["rule_jam_score"] = new_jam            # 덮어쓰기

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    after_values = [d.get("rule_jam_score", 0.0) for d in data]

    # 변화 요약 출력
    before_avg = sum(before_values) / len(before_values)
    after_avg  = sum(after_values)  / len(after_values)
    before_max = max(before_values)
    after_max  = max(after_values)
    n_jam_before = sum(1 for v in before_values if v >= 0.55)
    n_jam_after  = sum(1 for v in after_values  if v >= 0.55)

    print(f"  ✅ {pkl_path.name}: {len(data)}개 재계산 완료")
    print(f"     rule_jam 평균: {before_avg:.3f} → {after_avg:.3f}")
    print(f"     rule_jam 최대: {before_max:.3f} → {after_max:.3f}")
    print(f"     JAM(≥0.55) 비율: {n_jam_before/len(data)*100:.1f}% → {n_jam_after/len(data)*100:.1f}%")

    return len(data)


def delete_gru_weights(road_dir: Path):
    """gru_*.pt 파일을 삭제한다 (재학습 트리거)."""
    for name in ["gru_a.pt", "gru_b.pt"]:
        p = road_dir / name
        if p.exists():
            p.unlink()
            print(f"  🗑  {name} 삭제 완료")
        else:
            print(f"  [스킵] {name} — 없음")


def main():
    print("=" * 60)
    print("  GRU feature_log rule_jam_score 재계산 도구")
    print("=" * 60)

    flow_maps_dir = _HERE / "flow_maps"
    if not flow_maps_dir.exists():
        print(f"[오류] flow_maps 폴더 없음: {flow_maps_dir}")
        return

    # flow_maps/ 하위 모든 CCTV 폴더 탐색
    road_dirs = [d for d in flow_maps_dir.iterdir() if d.is_dir()]
    if not road_dirs:
        print("[오류] flow_maps/ 안에 CCTV 폴더가 없습니다.")
        return

    print(f"\n발견된 CCTV 폴더: {len(road_dirs)}개")
    for d in road_dirs:
        print(f"  - {d.name}")

    print("\n진행하면 각 폴더의 feature_log_*.pkl이 수정되고")
    print("gru_*.pt가 삭제됩니다 (flow_map.npy는 유지).")
    ans = input("\n계속하시겠습니까? (y/n): ").strip().lower()
    if ans != "y":
        print("취소.")
        return

    total_fixed = 0
    for road_dir in road_dirs:
        print(f"\n[{road_dir.name}]")
        for pkl_name in ["feature_log_a.pkl", "feature_log_b.pkl"]:
            n = fix_pkl(road_dir / pkl_name)
            total_fixed += n
        delete_gru_weights(road_dir)

    print(f"\n완료: 총 {total_fixed}개 feature 재계산.")
    print("다음 run_its_live.py 실행 시 GRU가 올바른 라벨로 재학습됩니다.")
    print("(pretrain은 gru_pretrain_min_sec 분량 데이터 수집 후 자동 시작)")


if __name__ == "__main__":
    main()
