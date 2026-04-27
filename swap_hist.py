# 역할: hist_jam_a.csv ↔ hist_jam_b.csv 를 교환한다.
#        장면 전환 후 Up/Down 예측이 뒤집힌 경우 1회 실행하면 복구된다.
# 사용: python swap_hist.py [폴더경로]
#        폴더경로 생략 시 flow_maps/ 를 기본으로 사용.

import sys
import shutil
from pathlib import Path

def swap_hist(directory: Path) -> None:
    a = directory / "hist_jam_a.csv"
    b = directory / "hist_jam_b.csv"

    if not a.exists() and not b.exists():
        print(f"[오류] {directory} 에 hist_jam_a.csv / hist_jam_b.csv 없음")
        return

    tmp = directory / "_hist_jam_tmp.csv"

    # Windows: rename이 덮어쓰기 안 함 → shutil.move 사용
    if a.exists() and b.exists():
        shutil.copy2(str(a), str(tmp))   # a → tmp
        shutil.copy2(str(b), str(a))     # b → a
        shutil.copy2(str(tmp), str(b))   # tmp → b
        tmp.unlink()
        print(f"✅ 교환 완료: {a.name} ↔ {b.name}  ({directory})")
    elif a.exists() and not b.exists():
        shutil.copy2(str(a), str(b))
        a.unlink()
        print(f"✅ {a.name} → {b.name} 이름 변경 완료  ({directory})")
    elif b.exists() and not a.exists():
        shutil.copy2(str(b), str(a))
        b.unlink()
        print(f"✅ {b.name} → {a.name} 이름 변경 완료  ({directory})")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = PROJECT_ROOT / "flow_maps"

    # 지정 폴더 + 하위 1단계 폴더 모두 처리
    candidates = [target] + [d for d in target.iterdir() if d.is_dir()] if target.is_dir() else [target.parent]

    found = False
    for d in candidates:
        if (d / "hist_jam_a.csv").exists() or (d / "hist_jam_b.csv").exists():
            swap_hist(d)
            found = True

    if not found:
        print("교환할 CSV 파일을 찾지 못했습니다.")
        print("사용법: python swap_hist.py [폴더경로]")
