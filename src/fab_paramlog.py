#!/usr/bin/env python
"""fab_paramlog.py — Advantest T2000 계열 ASCII 데이터로그(.log)에서 특정 테스트의
다이별 측정값을 추출해 파라메트릭 웨이퍼맵과 통계를 만든다 (파라메트릭 트랙 1차).

600MB급 로그도 스트리밍으로 처리한다. 블록 구조 가정:
  "DUT   X    Y" 헤더 → DUT별 좌표 행들 → 테스트 결과 행
  (테스트 행: TestID  이름  Index  결과  값단위  상한  하한  DUT  Pin — 공백 9토큰)

사용 예:
  python src/fab_paramlog.py --log "<테스터_데이터로그>.log" \
      --test <테스트명> --tag 09 \
      --map "<장비산출_최종맵>.txt"

산출:
  <out>/param_<tag>_<test>.csv   (X,Y,DUT,결과,값,단위 — 전체 행)
  <out>/param_<tag>_<test>.png   (값 히트맵: 어두울수록 낮음, 흰 점=FAIL, 검정=미측정)
  콘솔: 다이수/합불/평균±표준편차, Y 상·하반 비교, (--map 시) 맵 bin'2' 상·하 대조
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

DUT_HDR = re.compile(r"^\s*DUT\s+X\s+Y\s*$")
DUT_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$")
NUM = re.compile(r"^(-?\d+(?:\.\d+)?)([A-Za-z%\u00b5]*)$")


def stream_extract(log_path: Path, test_name: str | None = None,
                   fails_only: bool = False):
    """로그를 한 줄씩 스트리밍하며 (X, Y, DUT, 테스트, 결과, 값, 단위) 레코드 수집.

    fails_only=True면 테스트명과 무관하게 결과가 *FAIL 인 행을 전부 수집한다.
    """
    recs: list[tuple[int, int, int, str, str, float, str]] = []
    dutmap: dict[str, tuple[int, int]] = {}
    pending_hdr = False
    with open(log_path, errors="replace") as f:
        for line in f:
            if DUT_HDR.match(line):
                dutmap = {}
                pending_hdr = True
                continue
            if pending_hdr:
                m = DUT_ROW.match(line)
                if m:
                    dutmap[m.group(1)] = (int(m.group(2)), int(m.group(3)))
                    continue
                pending_hdr = False
            parts = line.split()
            if len(parts) >= 9:
                if fails_only:
                    if not parts[3].endswith("FAIL"):
                        continue
                elif parts[1] != test_name:
                    continue
                m = NUM.match(parts[4])
                xy = dutmap.get(parts[7])
                if m and xy:
                    recs.append((xy[0], xy[1], int(parts[7]), parts[1], parts[3],
                                 float(m.group(1)), m.group(2)))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--test", default="OS_NEG_LDOA0")
    ap.add_argument("--fails", action="store_true",
                    help="특정 테스트 대신 로그 내 모든 *FAIL 행을 수집(테스트별 요약)")
    ap.add_argument("--tag", required=True, help="출력 파일 접미(예: 08, 09)")
    ap.add_argument("--map", type=Path, default=None,
                    help="(선택) 최종맵 .txt — bin'2' 상·하 분포와 교차 대조")
    ap.add_argument("--out", type=Path, default=Path("datasets/fab_real/rendered"))
    a = ap.parse_args()

    print(f"[paramlog] {a.log.name} 스트리밍 중 (수백MB면 수십 초)...")
    recs = stream_extract(a.log, test_name=a.test, fails_only=a.fails)
    if not recs:
        raise SystemExit("[paramlog] 조건에 맞는 레코드를 찾지 못했습니다"
                         + ("" if a.fails else f" — 테스트명 '{a.test}' 확인"))

    a.out.mkdir(parents=True, exist_ok=True)
    suffix = "FAILS" if a.fails else a.test
    csv_path = a.out / f"param_{a.tag}_{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "DUT", "test", "result", "value", "unit"])
        w.writerows(recs)

    if a.fails:
        ymid = (max(r[1] for r in recs) + 1) // 2
        by_test: dict[str, list] = {}
        for x, y, _d, t, res, val, u in recs:
            by_test.setdefault(t, []).append((x, y, res, val, u))
        print(f"[{a.tag}] FAIL 행 {len(recs)}건 | 테스트 {len(by_test)}종 (첫 fail = 그 다이의 사인)")
        for t, rows in sorted(by_test.items(), key=lambda kv: -len(kv[1])):
            vs = np.array([r[3] for r in rows])
            up = sum(1 for r in rows if r[1] < ymid)
            print(f"  {t:<20} {len(rows):>4}건 | 상/하 {up}/{len(rows)-up}"
                  f" | 값범위 [{vs.min():.2f}, {vs.max():.2f}]{rows[0][4]}")
        print(f"  저장 → {csv_path.name}")
        return

    # 다이별 첫 레코드 기준 통계 (재측정 중복 방지)
    per_die: dict[tuple[int, int], tuple[str, float]] = {}
    for x, y, _d, _t, res, val, _u in recs:
        per_die.setdefault((x, y), (res, val))
    xs = np.array([k[0] for k in per_die])
    ys = np.array([k[1] for k in per_die])
    res = np.array([v[0] for v in per_die.values()])
    vals = np.array([v[1] for v in per_die.values()])
    unit = recs[0][6]
    fails = res != "PASS"

    print(f"[{a.tag}] {a.test}: 다이 {len(vals):,} | FAIL {int(fails.sum())} | "
          f"평균 {vals.mean():.2f}{unit} ± {vals.std():.2f} | "
          f"범위 [{vals.min():.2f}, {vals.max():.2f}]")
    ymid = (ys.max() + 1) // 2
    up, lo = ys < ymid, ys >= ymid
    for name, m in (("상반(Y<%d)" % ymid, up), ("하반", lo)):
        if m.sum():
            print(f"  {name}: 다이 {int(m.sum()):,} | FAIL {int((fails & m).sum())} | "
                  f"평균 {vals[m].mean():.2f}{unit} ± {vals[m].std():.2f}")

    if a.map:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from fab_map_ingest import parse_wafer_txt
        _wid, chars, grid, _b, _d = parse_wafer_txt(a.map)
        rmid = grid.shape[0] // 2
        b2r = np.where(chars == "2")[0]
        print(f"  [교차대조] 맵 bin'2' 상/하 = {int((b2r < rmid).sum())}/{int((b2r >= rmid).sum())} "
              f"vs 로그 {a.test} FAIL 상/하 = {int((fails & up).sum())}/{int((fails & lo).sum())} "
              f"(방향 일치 여부 확인용)")

    # 값 히트맵 (5~95 분위 정규화, FAIL=흰색, 미측정=검정)
    H, W = int(ys.max()) + 1, int(xs.max()) + 1
    canvas = np.full((H, W), np.nan)
    canvas[ys, xs] = vals
    lo_p, hi_p = np.nanpercentile(canvas, [5, 95])
    norm = np.nan_to_num(np.clip((canvas - lo_p) / max(hi_p - lo_p, 1e-9), 0, 1))
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[np.isnan(canvas)] = (0, 0, 0)
    img[ys[fails], xs[fails]] = (255, 255, 255)
    png_path = a.out / f"param_{a.tag}_{a.test}.png"
    cv2.imwrite(str(png_path), cv2.resize(img, (W * 10, H * 10),
                                          interpolation=cv2.INTER_NEAREST))
    print(f"  저장 → {csv_path.name}, {png_path.name} (색=값 크기, 흰점=FAIL)")


if __name__ == "__main__":
    main()
