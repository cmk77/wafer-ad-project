#!/usr/bin/env python
"""fab_bin_join.py — 로그 좌표계(X,Y) ↔ 맵 격자(행,열) 방향 정합 검증 + 다이 단위 bin 조인.

리테스트 FAILS CSV(잔존 fail 다이 = 최종맵의 fail 문자여야 함)를 최종맵에
8가지 방향 후보로 대조해 적중률이 가장 높은 변환을 찾고, 그 방향에서
'사인(첫 fail 테스트) 계열 ↔ 맵 bin 문자' 교차표를 출력한다.

  python src/fab_bin_join.py \
      --csv datasets/fab/rendered/param_08RT_FAILS.csv \
      --map "<장비산출_최종맵>.txt"
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fab_map_ingest import parse_wafer_txt  # noqa: E402


def category(test: str) -> str:
    if test.startswith("OS_"):
        return "OS"
    if test.startswith("TRIM_"):
        return "TRIM"
    if test.startswith("HVS_"):
        return "HVS"
    return "기타"


# (X, Y, H, W) -> (row, col). 1-기반 좌표 가정 + 상하/좌우 반전 + 축교환 후보.
TRANSFORMS = {
    "(r,c)=(Y-1, X-1)":          lambda x, y, H, W: (y - 1, x - 1),
    "(r,c)=(Y-1, W-X)":          lambda x, y, H, W: (y - 1, W - x),
    "(r,c)=(H-Y, X-1)":          lambda x, y, H, W: (H - y, x - 1),
    "(r,c)=(H-Y, W-X)":          lambda x, y, H, W: (H - y, W - x),
    "(r,c)=(X-1, Y-1) [축교환]": lambda x, y, H, W: (x - 1, y - 1),
    "(r,c)=(X-1, W-Y) [축교환]": lambda x, y, H, W: (x - 1, W - y),
    "(r,c)=(H-X, Y-1) [축교환]": lambda x, y, H, W: (H - x, y - 1),
    "(r,c)=(H-X, W-Y) [축교환]": lambda x, y, H, W: (H - x, W - y),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True,
                    help="fab_paramlog --fails 산출 CSV (리테스트 로그 권장)")
    ap.add_argument("--map", type=Path, required=True, help="최종맵 .txt")
    a = ap.parse_args()

    _wid, chars, grid, _bins, _dies = parse_wafer_txt(a.map)
    H, W = grid.shape
    with open(a.csv, newline="") as f:
        pts = [(int(r["X"]), int(r["Y"]), r["test"]) for r in csv.DictReader(f)]
    print(f"[join] fail 다이 {len(pts)}개 vs 맵 {H}행 x {W}열 — 8개 방향 후보 대조")

    results = []
    for name, fn in TRANSFORMS.items():
        hit = oob = 0
        for x, y, _t in pts:
            r, c = fn(x, y, H, W)
            if 0 <= r < H and 0 <= c < W:
                hit += int(grid[r, c] == 2)
            else:
                oob += 1
        results.append((hit, name, fn))
        print(f"  {name:<28} 적중 {hit:>3}/{len(pts)}" + (f" (범위밖 {oob})" if oob else ""))

    hit, name, fn = max(results, key=lambda t: t[0])
    rate = 100.0 * hit / max(len(pts), 1)
    print(f"\n[판정] 최적 변환 = {name} | 적중률 {rate:.1f}%")
    if rate < 90:
        print("  경고: 적중률 90% 미만 — 좌표계 가정(원점/오프셋) 재검토 필요")

    tab: dict[str, Counter] = {}
    mismatch = []
    for x, y, t in pts:
        r, c = fn(x, y, H, W)
        inb = 0 <= r < H and 0 <= c < W
        ch = chars[r, c] if inb else "OOB"
        tab.setdefault(category(t), Counter())[ch] += 1
        if not (inb and grid[r, c] == 2):
            mismatch.append((t, x, y, ch))

    print("\n[사인 계열 ↔ 맵 bin 문자 교차표]")
    for cat, cnt in sorted(tab.items(), key=lambda kv: -sum(kv[1].values())):
        body = ", ".join(f"'{ch}':{n}" for ch, n in cnt.most_common())
        print(f"  {cat:<4} ({sum(cnt.values()):>3}건) → {body}")

    if mismatch:
        print(f"\n[불일치] fail 문자가 아닌 칸에 착지 {len(mismatch)}건 (최대 10건 표시):")
        for t, x, y, ch in mismatch[:10]:
            print(f"  {t} @ X{x},Y{y} → '{ch}'")
    else:
        print("\n[불일치] 없음 — 좌표 정합 + bin 조인 완전 일치")


if __name__ == "__main__":
    main()
