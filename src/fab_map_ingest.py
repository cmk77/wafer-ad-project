#!/usr/bin/env python
"""fab_map_ingest.py — 프로버 최종맵(.txt, WAFER_MAP ASCII 형식) 어댑터.

테스터 산출물 중 `WAFER_MAP = {` 헤더를 가진 .txt를 파싱해
① WM-811K 팔레트(0=칩없음/128=양품/255=불량) 격자 PNG — 기존 트랙 A 모델 입력과 동일 규약
② fail bin 문자별 컬러맵 PNG — 어떤 불량 bin이 어디에 모이는지 진단용
③ 다이/수율/빈별 통계 + 헤더(DIES/BIN) 대사
를 만들고, `predict`로 학습된 분류기(best.pt)의 패턴 판정을 출력한다.

사용 예:
  python src/fab_map_ingest.py parse \
      --txt "<장비산출_최종맵>.txt" --out datasets/fab/rendered
  python src/fab_map_ingest.py predict \
      --png "datasets/fab/rendered/<WAFER_ID>_64.png" \
      --ckpt results/wm_cls/best.pt
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

import cv2
import numpy as np

# 0=칩 없음, 1=양품, 2=불량, 3=특수마커('!' 등 — 헤더 DIES/BIN 카운트에서 제외되는 기준 다이)
PALETTE = np.array([0, 128, 255, 128], dtype=np.uint8)  # 모델 입력에서 마커는 양품 취급
SCALE = 10  # 사람 눈 확인용 확대 배율

# 검증 확정 사항 (fab_bin_join — 실측 리테스트 FAILS 좌표 대조 100% 적중, 불일치 0):
#   - 텍스트 행 i = 웨이퍼 Y=i+1 (위→아래), 행 내 문자 j = X=j+1 (왼→오)
#   - 로그 좌표 (X,Y) → 격자 (row,col) = (Y-1, X-1). 축교환/반전 없음.
#   - 헤더 ROWS/COLUMNS 는 텍스트 배열과 전치 표기 (ROWS=행 길이=X칸수)
#   - '!' 는 매 웨이퍼 동일 위치(기준 다이 마커) — DIES/BIN 카운트에 미포함
#   - bin 의미: '2'=OS 사인, '5'=TRIM 사인, '4'=HVS 사인 (사인=첫 fail 테스트 계열)
MARKER_CHARS = {"!"}

# fail bin 문자별 표시 색 (BGR) — 최대 10종 순환
BIN_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255),
              (255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 0, 255),
              (203, 192, 255), (255, 255, 255)]


def parse_wafer_txt(path: Path, return_header: bool = False):
    """WAFER_MAP = { ... } 형식 파서. (wafer_id, 문자격자, 0/1/2/3격자, 선언빈, 선언다이수[, 헤더]) 반환."""
    text = path.read_text(errors="replace")

    def kv(key: str, default=None):
        m = re.search(rf'^{key}\s*=\s*"?([^"\r\n]+)"?\s*$', text, re.M)
        return m.group(1).split("//")[0].strip() if m else default

    wafer_id = (kv("WAFER_ID") or path.stem).strip()
    null_bin = (kv("NULL_BIN") or ".")[0]
    dies_declared = kv("DIES")
    dies_declared = int(dies_declared) if dies_declared else None
    # BIN = "T" 65 "Fail" "..." — 카운트와 Pass/Fail 종별까지 수집
    declared_bins = {m.group(1): int(m.group(2))
                     for m in re.finditer(r'^BIN\s*=\s*"(.+?)"\s+(\d+)', text, re.M)}
    bin_kinds = {m.group(1): m.group(3)
                 for m in re.finditer(r'^BIN\s*=\s*"(.+?)"\s+(\d+)\s+"(\w+)"', text, re.M)}

    m = re.search(r'^MAP\s*=\s*\{\s*$(.*?)^\s*\}', text, re.M | re.S)
    if not m:
        raise SystemExit(f"[parse] MAP = {{ ... }} 블록을 찾지 못했습니다: {path}")
    rows = [ln.rstrip("\r") for ln in m.group(1).splitlines() if ln.strip()]
    width = max(len(r) for r in rows)
    chars = np.array([list(r.ljust(width, null_bin)) for r in rows])

    grid = np.where(chars == null_bin, 0,
                    np.where(chars == "1", 1,
                             np.where(np.isin(chars, list(MARKER_CHARS)), 3, 2))).astype(np.uint8)
    if not return_header:
        return wafer_id, chars, grid, declared_bins, dies_declared
    header = {"rows": kv("ROWS"), "columns": kv("COLUMNS"),
              "x_size": kv("X_SIZE"), "y_size": kv("Y_SIZE"),
              "wafer_size": kv("WAFER_SIZE"), "flat_notch": kv("FLAT_NOTCH"),
              "bin_kinds": bin_kinds}
    return wafer_id, chars, grid, declared_bins, dies_declared, header


def cmd_parse(a) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    wid, chars, grid, declared_bins, dies_declared, hdr = \
        parse_wafer_txt(a.txt, return_header=True)
    h, w = grid.shape

    # 마커('!' 등)는 다이/수율 통계에서 제외 — 헤더 DIES/BIN 카운트와 동일 기준
    n_marker = int((grid == 3).sum())
    n_pass = int((grid == 1).sum())
    fail_chars = collections.Counter(chars[grid == 2].tolist())
    n_fail = sum(fail_chars.values())
    n_die = n_pass + n_fail
    yld = 100.0 * n_pass / max(n_die, 1)

    print(f"[{wid}] 격자 {h}행 x {w}열 | 다이 {n_die:,} (pass {n_pass:,} / fail {n_fail:,}"
          + (f" / 마커 {n_marker}" if n_marker else "") + f") | 수율 {yld:.1f}%")

    # ── 헤더 대사 (정합성 리포트) ─────────────────────────────────────
    ok = lambda b: "✓" if b else "✗ 불일치"
    if dies_declared is not None:
        print(f"  [대사] DIES 선언 {dies_declared} vs 실측 {n_die} … {ok(dies_declared == n_die)}")
    kinds = hdr.get("bin_kinds", {})
    decl_pass = sum(c for ch, c in declared_bins.items() if kinds.get(ch) == "Pass")
    decl_fail = sum(c for ch, c in declared_bins.items() if kinds.get(ch) == "Fail")
    if kinds:
        print(f"  [대사] Pass 선언 {decl_pass} vs 실측 {n_pass} … {ok(decl_pass == n_pass)}")
        print(f"  [대사] Fail 선언 {decl_fail} vs 실측 {n_fail} … {ok(decl_fail == n_fail)}")
    if hdr.get("rows") and hdr.get("columns"):
        r_d, c_d = int(hdr["rows"]), int(hdr["columns"])
        if {r_d, c_d} == {h, w}:
            note = " (헤더는 전치 표기 — 텍스트 행=Y, 문자=X가 물리 배치)" if (r_d, c_d) != (h, w) else ""
            print(f"  [대사] 격자 크기 {{{r_d},{c_d}}} vs 텍스트 {{{h},{w}}} … ✓{note}")
        else:
            print(f"  [대사] 격자 크기 헤더 ({r_d},{c_d}) vs 텍스트 ({h},{w}) … ✗ 불일치")
    print("  [규약] 로그 (X,Y) → 격자 (row,col) = (Y-1, X-1)  [fab_bin_join 검증 100%]")

    for ch, cnt in sorted(fail_chars.items(), key=lambda x: -x[1]):
        d = declared_bins.get(ch)
        print(f"  fail bin '{ch}': {cnt:,}개" + (f" (헤더 선언 {d})" if d is not None else ""))
    if n_marker:
        mpos = [f"(r{r},c{c})" for r, c in zip(*np.where(grid == 3))]
        print(f"  마커 {sorted(MARKER_CHARS)} 위치: {', '.join(mpos)} — 통계 제외, 모델 입력에선 양품 취급")

    # ① WM-811K 규약 PNG (원본 격자 + 확대본 + 64px 모델 입력본)
    img = PALETTE[grid]
    stem = wid.replace("/", "_")
    cv2.imwrite(str(a.out / f"{stem}.png"), img)
    cv2.imwrite(str(a.out / f"{stem}_view.png"),
                cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(a.out / f"{stem}_64.png"),
                cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST))

    # ② fail bin 문자별 컬러맵 (배경 어둡게, pass 회색, bin별 원색, 마커 흰색)
    color = np.zeros((h, w, 3), np.uint8)
    color[grid == 1] = (70, 70, 70)
    color[grid == 3] = (255, 255, 255)
    legend = []
    for i, (ch, cnt) in enumerate(sorted(fail_chars.items(), key=lambda x: -x[1])):
        c = BIN_COLORS[i % len(BIN_COLORS)]
        color[chars == ch] = c
        legend.append(f"'{ch}'={cnt} (BGR{c})")
    if n_marker:
        legend.append(f"마커={n_marker} (흰색)")
    cv2.imwrite(str(a.out / f"{stem}_bins.png"),
                cv2.resize(color, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST))
    if legend:
        print("  bin 컬러맵 범례: " + ", ".join(legend))
    print(f"  저장 → {a.out}/{stem}{{.png,_view.png,_64.png,_bins.png}}")


def cmd_predict(a) -> None:
    import json

    import torch
    from wafermap_classifier import WaferNet, to_tensor  # 학습과 동일 전처리 보장

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = WaferNet(len(classes))
    model.load_state_dict(ck["state"])
    model.eval()

    img = cv2.imread(str(a.png), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"[predict] 이미지를 읽지 못했습니다: {a.png}")
    with torch.no_grad():
        prob = torch.softmax(model(to_tensor(img).unsqueeze(0))[0], dim=0)
    topk = torch.topk(prob, k=min(3, len(classes)))
    pairs = [(classes[i], float(p)) for p, i in zip(topk.values.tolist(),
                                                    topk.indices.tolist())]
    print(f"[predict] {Path(a.png).name}")
    for name, p in pairs:
        print(f"  {name:<10} {p*100:5.1f}%")

    if a.emit_json:
        # gemma_reporter --ad-json 이 그대로 소비하는 ad_result 스키마
        top1, p1 = pairs[0]
        p_none = float(prob[classes.index("none")]) if "none" in classes else 0.0
        result = {
            "image": str(a.png), "category": Path(a.png).stem,
            "source": "track_A_wafermap", "anomaly_score": round(1.0 - p_none, 4),
            "is_anomaly": top1 != "none", "defect_type": top1,
            "type_confidence": round(p1, 4),
            "topk": [{"label": n, "prob": round(p, 4)} for n, p in pairs],
        }
        a.emit_json.parent.mkdir(parents=True, exist_ok=True)
        a.emit_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"  → ad_result JSON 저장: {a.emit_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse", help=".txt 최종맵 → PNG/통계")
    p.add_argument("--txt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("datasets/fab_real/rendered"))

    q = sub.add_parser("predict", help="렌더링 PNG → 분류기 패턴 판정")
    q.add_argument("--png", type=Path, required=True)
    q.add_argument("--ckpt", type=Path, default=Path("results/wm_cls/best.pt"))
    q.add_argument("--emit-json", type=Path, default=None,
                   help="ad_result JSON 저장 경로 (gemma_reporter --ad-json 입력용)")

    a = ap.parse_args()
    cmd_parse(a) if a.cmd == "parse" else cmd_predict(a)


if __name__ == "__main__":
    main()
