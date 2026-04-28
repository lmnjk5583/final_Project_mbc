# 역할: hist_jam_b.csv(UP) vs hist_jam_a.csv(DOWN) 를 비교해
#        B(UP) 가 A(DOWN) 보다 jam이 낮으면 두 파일을 교환한다.
#        count를 가중치로 사용 — 데이터가 많은 슬롯을 더 신뢰.
#
# 양재 기준: A=하행(DOWN), B=상행(UP), B가 항상 더 혼잡
#
# 사용:
#   python fix_hist_direction.py                          # flow_maps/ 하위 전체
#   python fix_hist_direction.py "flow_maps/[경부선] 양재"
#   python fix_hist_direction.py --dry-run                # 분석만
#   python fix_hist_direction.py --threshold 0.55         # 교환 기준 (기본 0.55)

import csv
import shutil
import argparse
from pathlib import Path


def _load(path: Path) -> dict:
    """(hour, minute_start) → (avg_jam, count)"""
    result = {}
    if not path.exists():
        return result
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                count = int(row["count"])
                if count == 0:
                    continue
                key = (int(row["hour"]), int(row["minute_start"]))
                result[key] = (float(row["jam_sum"]) / count, count)
            except (KeyError, ValueError):
                continue
    return result


def _fmt(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"


def analyze(directory: Path, dry_run: bool, threshold: float) -> None:
    a_path = directory / "hist_jam_a.csv"
    b_path = directory / "hist_jam_b.csv"

    if not a_path.exists() or not b_path.exists():
        print(f"[건너뜀] {directory} — hist_jam_a/b.csv 없음")
        return

    a_data = _load(a_path)
    b_data = _load(b_path)

    common = sorted(set(a_data) & set(b_data))
    if not common:
        print(f"[{directory.name}] 공통 슬롯 없음")
        return

    # count 가중 비교
    # 각 슬롯의 신뢰도 = min(count_a, count_b)
    # B < A 인 슬롯의 가중합 / 전체 가중합 = 교환 필요 비율
    w_b_lower  = 0.0   # B(UP) < A(DOWN) 가중합
    w_total    = 0.0   # 전체 가중합
    b_lower    = []    # B < A 슬롯 목록
    b_higher   = []    # B >= A 슬롯 목록

    for key in common:
        avg_a, cnt_a = a_data[key]
        avg_b, cnt_b = b_data[key]
        w = min(cnt_a, cnt_b)   # 더 적은 쪽 count = 신뢰도
        w_total += w
        if avg_b < avg_a:
            w_b_lower += w
            b_lower.append((key, avg_a, cnt_a, avg_b, cnt_b))
        else:
            b_higher.append((key, avg_a, cnt_a, avg_b, cnt_b))

    ratio = w_b_lower / w_total if w_total > 0 else 0.0

    # ── 러시아워 구간 요약 ───────────────────────────────────────────
    rush_slots = [(k, a_data[k], b_data[k]) for k in common
                  if (6 <= k[0] <= 9) or (15 <= k[0] <= 19)]
    rush_b_lower = sum(1 for k, a, b in rush_slots if b[0] < a[0])

    print(f"\n{'='*60}")
    print(f"폴더      : {directory}")
    print(f"공통 슬롯  : {len(common)}개")
    print(f"B(UP) < A(DOWN) : {len(b_lower)}개  /  B >= A: {len(b_higher)}개")
    print(f"가중 B<A 비율   : {ratio:.1%}  (교환 기준: {threshold:.0%})")
    if rush_slots:
        print(f"러시아워(6~9/15~19시) B<A: {rush_b_lower}/{len(rush_slots)}개")

    # 대표 샘플 출력
    if b_lower:
        print(f"\n[B(UP) < A(DOWN) 샘플 — 최대 5개]")
        for key, avg_a, cnt_a, avg_b, cnt_b in b_lower[:5]:
            print(f"  {_fmt(*key)}  A={avg_a:.3f}(n={cnt_a})  B={avg_b:.3f}(n={cnt_b})"
                  f"  diff={avg_b-avg_a:+.3f}")
    if b_higher:
        print(f"\n[B(UP) >= A(DOWN) 샘플 — 최대 5개]")
        for key, avg_a, cnt_a, avg_b, cnt_b in b_higher[:5]:
            print(f"  {_fmt(*key)}  A={avg_a:.3f}(n={cnt_a})  B={avg_b:.3f}(n={cnt_b})"
                  f"  diff={avg_b-avg_a:+.3f}")

    # ── 교환 판정 ────────────────────────────────────────────────────
    if ratio >= threshold:
        if dry_run:
            print(f"\n⚠️  [DRY-RUN] 가중 B<A {ratio:.1%} ≥ {threshold:.0%}"
                  f" → 실행 시 A↔B 교환됩니다.")
        else:
            tmp = directory / "_hist_tmp.csv"
            shutil.copy2(str(a_path), str(tmp))
            shutil.copy2(str(b_path), str(a_path))
            shutil.copy2(str(tmp), str(b_path))
            tmp.unlink()
            print(f"\n✅ 교환 완료: A↔B  →  B(UP)가 더 높은 jam 이력 보유")
    else:
        print(f"\n✔  교환 불필요: 가중 B<A {ratio:.1%} < {threshold:.0%}  (정상)")


def main():
    p = argparse.ArgumentParser(description="hist_jam B(UP)>A(DOWN) 방향 교정")
    p.add_argument("folder", nargs="?", default=None,
                   help="대상 폴더 (생략 시 flow_maps/ 하위 전체)")
    p.add_argument("--dry-run", action="store_true",
                   help="분석만, 파일 교환 안 함")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="교환 기준 가중 비율 (기본 0.55)")
    args = p.parse_args()

    root = Path(__file__).resolve().parent

    if args.folder:
        candidates = [Path(args.folder)]
    else:
        base = root / "flow_maps"
        if not base.exists():
            print(f"[오류] flow_maps/ 없음: {base}")
            raise SystemExit(1)
        candidates = [base] + [d for d in base.iterdir() if d.is_dir()]

    found = False
    for d in candidates:
        if (d / "hist_jam_a.csv").exists() and (d / "hist_jam_b.csv").exists():
            analyze(d, dry_run=args.dry_run, threshold=args.threshold)
            found = True

    if not found:
        print("hist_jam_a/b.csv 쌍을 찾지 못했습니다.")


if __name__ == "__main__":
    main()
