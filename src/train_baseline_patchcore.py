#!/usr/bin/env python
"""train_baseline_patchcore.py — Phase 0/1 기준선.

fab_root/<클래스>/{train/good, test/good, test/defect} 구조를 순회하며
클래스별 PatchCore를 학습(=정상 피처 뱅크 구축, gradient 학습 없음)하고
AUROC 집계 + OpenVINO 내보내기까지 수행한다.

RTX 6000 Ada 기준 클래스당 수 분. 신규 제품 클래스 온보딩도 이 스크립트를
해당 클래스에만 실행하면 끝난다 (--classes 새클래스명).

    python src/train_baseline_patchcore.py --root datasets/fab --classes A B C
    python src/train_baseline_patchcore.py --root datasets --classes wm811k_ad   # WM-811K 데모
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

try:
    from anomalib.deploy import ExportType
except ImportError:  # 구버전 호환
    ExportType = None


def make_datamodule(root: Path, name: str, batch: int, workers: int) -> Folder:
    kwargs = dict(
        name=name,
        root=str(root),
        normal_dir="train/good",
        abnormal_dir="test/defect",
        train_batch_size=batch,
        eval_batch_size=batch,
        num_workers=workers,
    )
    try:  # 정상 테스트 폴더가 있으면 사용 (버전에 따라 인자 상이)
        return Folder(normal_test_dir="test/good", **kwargs)
    except TypeError:
        return Folder(**kwargs)


def run_class(cls_dir: Path, out_root: Path, args) -> dict:
    name = cls_dir.name
    print(f"\n===== [{name}] PatchCore =====")
    dm = make_datamodule(cls_dir, name, args.batch, args.workers)
    model = Patchcore(
        backbone=args.backbone,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=args.coreset,
    )
    engine = Engine(default_root_dir=str(out_root / name))
    engine.fit(model=model, datamodule=dm)
    results = engine.test(model=model, datamodule=dm)
    metrics = results[0] if results else {}
    print(f"[{name}] metrics: {metrics}")

    if args.export and ExportType is not None:
        try:
            path = engine.export(model=model, export_type=ExportType.OPENVINO)
            print(f"[{name}] OpenVINO export → {path}")
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] export 실패(선택 기능): {e}")

    row = {"class": name}
    row.update({k: float(v) for k, v in metrics.items()})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("datasets/fab"),
                    help="클래스 폴더들의 부모 디렉토리")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="지정 시 해당 클래스만. 미지정 시 root 하위 전부(70종 일괄)")
    ap.add_argument("--out", type=Path, default=Path("results/patchcore"))
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--coreset", type=float, default=0.1,
                    help="메모리뱅크 coreset 비율. 70클래스 운영 시 0.01~0.1 권장")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--export", action="store_true", help="OpenVINO 내보내기")
    args = ap.parse_args()

    if args.classes:
        class_dirs = [args.root / c for c in args.classes]
    else:
        class_dirs = sorted(p for p in args.root.iterdir() if (p / "train/good").exists())
    if not class_dirs:
        raise SystemExit(f"클래스 폴더 없음: {args.root}/<cls>/train/good 확인")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in class_dirs:
        try:
            rows.append(run_class(d, args.out, args))
        except Exception as e:  # noqa: BLE001
            print(f"[{d.name}] 실패: {e}")
            rows.append({"class": d.name, "error": str(e)})

    keys = sorted({k for r in rows for k in r})
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n요약 저장: {args.out/'summary.csv'}  "
          f"(image_AUROC < 0.97 클래스는 PLAN.md 4절에 따라 개별 보완)")


if __name__ == "__main__":
    main()
