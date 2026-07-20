#!/usr/bin/env python
"""train_multiclass_dinomaly.py — Phase 1: 70클래스 통합 1모델.

Dinomaly(DINOv2 재구성 기반)는 멀티클래스 통합 설정에서 MVTec 99%대 성능을
보이는 대표 모델로, 클래스별 모델 70개를 하나로 통합해 운영 비용을 줄인다.
anomalib>=2.1 필요.

동작:
  1) fab_root/<클래스>/... 의 정상/불량을 심볼릭 링크로 combined/ 에 통합
  2) 통합 데이터로 Dinomaly 1회 학습
  3) 클래스별 test 세트로 per-class AUROC 리포트

    python src/train_multiclass_dinomaly.py --root datasets/fab --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine

try:
    from anomalib.models import Dinomaly
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Dinomaly를 찾을 수 없습니다. anomalib>=2.1 필요: pip install -U anomalib"
    ) from e


def build_combined(root: Path, combined: Path, classes: list[Path]) -> None:
    """클래스별 폴더를 하나의 anomalib Folder 레이아웃으로 심링크 통합."""
    for d in ("train/good", "test/good", "test/defect"):
        (combined / d).mkdir(parents=True, exist_ok=True)
    n = 0
    for cls in classes:
        for sub in ("train/good", "test/good", "test/defect"):
            src_dir = cls / sub
            if not src_dir.exists():
                continue
            for p in src_dir.glob("*"):
                dst = combined / sub / f"{cls.name}__{p.name}"
                if not dst.exists():
                    os.symlink(p.resolve(), dst)
                    n += 1
    print(f"[combined] {len(classes)} classes, {n} links → {combined}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("datasets/fab"))
    ap.add_argument("--combined", type=Path, default=Path("datasets/_combined"))
    ap.add_argument("--out", type=Path, default=Path("results/dinomaly"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8,
                    help="ViT-B @ 392~448px 기준 48GB에서 8~16 권장")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    classes = sorted(p for p in args.root.iterdir() if (p / "train/good").exists())
    if not classes:
        raise SystemExit(f"클래스 폴더 없음: {args.root}")
    build_combined(args.root, args.combined, classes)

    dm = Folder(
        name="fab_multiclass",
        root=str(args.combined),
        normal_dir="train/good",
        abnormal_dir="test/defect",
        train_batch_size=args.batch,
        eval_batch_size=args.batch,
        num_workers=args.workers,
    )
    model = Dinomaly()  # 기본: DINOv2 ViT 인코더 + 디코더 재구성
    engine = Engine(default_root_dir=str(args.out), max_epochs=args.epochs)
    engine.fit(model=model, datamodule=dm)

    # 통합 지표
    overall = engine.test(model=model, datamodule=dm)
    print(f"[overall] {overall}")

    # 클래스별 지표 (test만 개별 실행)
    rows = []
    for cls in classes:
        try:
            dm_c = Folder(
                name=f"eval_{cls.name}",
                root=str(cls),
                normal_dir="train/good",   # setup 요구용 (평가에는 test만 사용)
                abnormal_dir="test/defect",
                eval_batch_size=args.batch,
                num_workers=args.workers,
            )
            r = engine.test(model=model, datamodule=dm_c)
            m = {k: float(v) for k, v in (r[0] if r else {}).items()}
            m["class"] = cls.name
            rows.append(m)
            print(f"[{cls.name}] {m}")
        except Exception as e:  # noqa: BLE001
            rows.append({"class": cls.name, "error": str(e)})

    keys = sorted({k for r in rows for k in r})
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"클래스별 요약: {args.out/'per_class.csv'}")
    print("AUROC 미달 클래스는 PatchCore 개별 모델로 보완(하이브리드) — PLAN.md 3절")


if __name__ == "__main__":
    main()
