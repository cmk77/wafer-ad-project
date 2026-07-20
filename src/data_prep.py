#!/usr/bin/env python
"""data_prep.py — 공개 웨이퍼 맵 데이터 전처리.

1) WM-811K (LSWMD.pkl) → 클래스별 PNG + train/test 분할
2) MixedWM38 (Wafer_Map_Datasets.npz) → 클래스별 PNG
3) anomalib Folder 규약 레이아웃 생성 (normal='none', abnormal=결함패턴)

사용 예:
    python src/data_prep.py wm811k    --pkl datasets/wm811k/LSWMD.pkl --out datasets/wm811k_png
    python src/data_prep.py mixedwm38 --npz datasets/mixedwm38/Wafer_Map_Datasets.npz --out datasets/mixedwm38_png
    python src/data_prep.py anomalib-layout --src datasets/wm811k_png --out datasets/wm811k_ad
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np

WM811K_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
                  "Random", "Scratch", "Near-full", "none"]

# 값 인코딩: 0=칩 없음(웨이퍼 밖), 1=양품 다이, 2=불량 다이
PALETTE = np.array([0, 128, 255], dtype=np.uint8)


def _to_png_array(wafer: np.ndarray, size: int) -> np.ndarray:
    import cv2
    img = PALETTE[np.clip(wafer, 0, 2).astype(np.int64)]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def _label_str(x) -> str | None:
    """LSWMD의 failureType은 [['Center']] / [] 등 numpy 중첩 배열 형태."""
    try:
        if x is None:
            return None
        arr = np.asarray(x, dtype=object).ravel()
        if arr.size == 0:
            return None
        return str(arr[0])
    except Exception:  # noqa: BLE001
        return None


def _read_lswmd(pkl: Path):
    """LSWMD.pkl은 2017년경 pandas 0.x로 저장된 레거시 피클.

    pandas 3.x는 구버전 피클 호환 계층(pickle_compat)을 제거해
    "No module named 'pandas.indexes'" 오류가 난다.
    1차: 일반 로드 → 2차: 레거시 모듈 경로를 현행 경로로 매핑하는
    언피클러로 재시도 → 그래도 실패하면 pandas 2.2.x 설치 안내.
    """
    import pandas as pd

    try:
        return pd.read_pickle(pkl)
    except (ModuleNotFoundError, AttributeError, ImportError) as first_err:
        import pickle

        class _LegacyUnpickler(pickle.Unpickler):
            _MAP = {
                ("pandas.indexes.base", "Index"): ("pandas.core.indexes.base", "Index"),
                ("pandas.indexes.base", "_new_Index"): ("pandas.core.indexes.base", "_new_Index"),
                ("pandas.indexes.numeric", "Int64Index"): ("pandas.core.indexes.base", "Index"),
                ("pandas.indexes.numeric", "Float64Index"): ("pandas.core.indexes.base", "Index"),
                ("pandas.indexes.range", "RangeIndex"): ("pandas.core.indexes.range", "RangeIndex"),
                ("pandas.core.index", "Index"): ("pandas.core.indexes.base", "Index"),
                ("pandas.core.index", "Int64Index"): ("pandas.core.indexes.base", "Index"),
                ("pandas.tslib", "Timestamp"): ("pandas._libs.tslibs.timestamps", "Timestamp"),
            }

            def find_class(self, module, name):
                module, name = self._MAP.get((module, name), (module, name))
                if module.startswith("pandas.indexes"):
                    module = module.replace("pandas.indexes", "pandas.core.indexes")
                return super().find_class(module, name)

        try:
            with open(pkl, "rb") as f:
                return _LegacyUnpickler(f).load()
        except Exception:
            raise SystemExit(
                f"[wm811k] 레거시 pkl 로드 실패 ({type(first_err).__name__}: {first_err})\n"
                '  해결: venv에서  pip install "pandas>=2.2,<3"  실행 후 재시도하세요.\n'
                "  (pandas 3.x는 2017년산 피클 호환 계층이 제거되어 LSWMD.pkl을 못 읽습니다)"
            ) from first_err


def prep_wm811k(pkl: Path, out: Path, size: int, limit_per_class: int | None) -> None:
    import cv2  # noqa: F401  (설치 확인)

    print(f"[wm811k] loading {pkl} ... (약 2GB, 수 분 소요)")
    df = _read_lswmd(pkl)
    print(f"[wm811k] total rows: {len(df):,}")

    counts: dict[str, int] = {}
    for split in ("train", "test"):
        for c in WM811K_CLASSES:
            (out / split / c).mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(df.itertuples(index=False)):
        label = _label_str(getattr(row, "failureType", None))
        if label is None or label not in WM811K_CLASSES:
            continue  # 미라벨 638K장은 스킵 (능동학습 단계에서 활용)
        if limit_per_class and counts.get(label, 0) >= limit_per_class:
            continue
        # 주의: 원본 pkl의 컬럼명은 'trianTestLabel' (데이터셋 자체 오타). 양쪽 다 대응.
        raw_split = (_label_str(getattr(row, "trianTestLabel", None))
                     or _label_str(getattr(row, "trainTestLabel", None)) or "train")
        split = "train" if raw_split.lower().startswith("tra") else "test"
        img = _to_png_array(np.asarray(row.waferMap), size)
        import cv2
        cv2.imwrite(str(out / split / label / f"{i:06d}.png"), img)
        counts[label] = counts.get(label, 0) + 1
        if sum(counts.values()) % 20000 == 0:
            print(f"  saved {sum(counts.values()):,} ...")

    print("[wm811k] per-class:", dict(sorted(counts.items())))
    print(f"[wm811k] done → {out}")


def prep_mixedwm38(npz: Path, out: Path, size: int) -> None:
    import cv2
    data = np.load(npz)
    maps, labels = data["arr_0"], data["arr_1"]  # (N,52,52), (N,8) multi-hot(고유 조합 38 = 38클래스)
    vals = np.unique(maps)
    print(f"[mixedwm38] {maps.shape[0]:,} maps, label dim {labels.shape[1]}, 픽셀값 {vals.tolist()}")
    if vals.max() > 2:
        n_over = int((maps > 2).sum())
        print(f"  주의: 값 {vals[vals > 2].tolist()} 픽셀 {n_over:,}개 발견 → 불량 다이(2)로 간주해 클립 처리합니다.")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "labels.npy", labels)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    for i in range(maps.shape[0]):
        cv2.imwrite(str(img_dir / f"{i:06d}.png"), _to_png_array(maps[i], size))
    print(f"[mixedwm38] done → {out} (labels.npy: (N,8) multi-hot, 고유 조합 38클래스)")


def anomalib_layout(src: Path, out: Path, val_ratio: float = 0.1, seed: int = 0,
                    max_normal: int = 2000, max_test_good: int = 1500,
                    max_test_defect: int = 3000) -> None:
    """분류 폴더(wm811k_png) → anomalib Folder 규약.

    normal = 'none' 패턴, abnormal = 나머지 결함 패턴 전부.
    결과: out/{train/good, test/good, test/defect}

    중요: PatchCore는 train/good 전체의 패치 피처를 메모리에 올려 코어셋을
    뽑으므로 무제한 복사(WM-811K 'none' 약 14.7만 장) 시 OOM이 난다.
    기본 상한(2000/1500/3000)은 48GB GPU에서 안전하며, 0을 주면 무제한.
    """
    rng = random.Random(seed)
    for d in ("train/good", "test/good", "test/defect"):
        (out / d).mkdir(parents=True, exist_ok=True)

    normals = sorted((src / "train" / "none").glob("*.png"))
    rng.shuffle(normals)
    n_test_good = min(max_test_good, len(normals) // 10) if max_test_good else max(1, int(len(normals) * val_ratio))
    test_good = normals[:n_test_good]
    train_pool = normals[n_test_good:]
    train_good = train_pool[:max_normal] if max_normal else train_pool
    for p in train_good:
        shutil.copy(p, out / "train/good" / p.name)
    for p in test_good:
        shutil.copy(p, out / "test/good" / p.name)

    defects: list[tuple[str, Path]] = []
    for c in WM811K_CLASSES:
        if c == "none":
            continue
        defects += [(c, p) for p in (src / "test" / c).glob("*.png")]
    rng.shuffle(defects)
    if max_test_defect:
        defects = defects[:max_test_defect]
    for c, p in defects:
        shutil.copy(p, out / "test/defect" / f"{c}_{p.name}")

    print(f"[layout] train/good={len(train_good)}, test/good={len(test_good)}, "
          f"test/defect={len(defects)} → {out}")
    if max_normal:
        print(f"  (상한 적용: --max-normal {max_normal} / --max-test-good {max_test_good}"
              f" / --max-test-defect {max_test_defect}, 0=무제한)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("wm811k")
    p1.add_argument("--pkl", type=Path, required=True)
    p1.add_argument("--out", type=Path, default=Path("datasets/wm811k_png"))
    p1.add_argument("--size", type=int, default=64)
    p1.add_argument("--limit-per-class", type=int, default=None,
                    help="빠른 실험용 클래스당 최대 장수 (예: 2000)")

    p2 = sub.add_parser("mixedwm38")
    p2.add_argument("--npz", type=Path, required=True)
    p2.add_argument("--out", type=Path, default=Path("datasets/mixedwm38_png"))
    p2.add_argument("--size", type=int, default=64)

    p3 = sub.add_parser("anomalib-layout")
    p3.add_argument("--src", type=Path, required=True)
    p3.add_argument("--out", type=Path, default=Path("datasets/wm811k_ad"))
    p3.add_argument("--max-normal", type=int, default=2000,
                    help="train/good 최대 장수 (0=무제한, PatchCore OOM 방지 기본 2000)")
    p3.add_argument("--max-test-good", type=int, default=1500)
    p3.add_argument("--max-test-defect", type=int, default=3000)

    a = ap.parse_args()
    if a.cmd == "wm811k":
        prep_wm811k(a.pkl, a.out, a.size, a.limit_per_class)
    elif a.cmd == "mixedwm38":
        prep_mixedwm38(a.npz, a.out, a.size)
    else:
        anomalib_layout(a.src, a.out, max_normal=a.max_normal,
                        max_test_good=a.max_test_good,
                        max_test_defect=a.max_test_defect)


if __name__ == "__main__":
    main()
