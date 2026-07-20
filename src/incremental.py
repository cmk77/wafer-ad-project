#!/usr/bin/env python
"""incremental.py — 신규 불량/신규 클래스 적응 체계 (PLAN.md 4절 L2·L3 구현).

L2. DefectTypeRegistry
    비지도 AD가 잡아낸 '미지의 불량'에 이름을 붙이고 자동 분류하기 위한
    few-shot 임베딩 뱅크. DINOv2 전역 임베딩의 클래스 프로토타입을 저장하고
    코사인 유사도 kNN으로 분류한다. **gradient 재학습이 전혀 없어**
    신규 불량 유형 등록이 5~20장 + 수 분이면 끝난다.

L3. 리플레이 버퍼
    Dinomaly 통합 모델에 신규 제품 클래스를 추가할 때, 기존 클래스당
    N장(기본 100)을 보존해 재파인튜닝 시 파국적 망각을 방지한다.

CLI:
  # 신규 불량 유형 등록 (예: 'edge-burn'이라는 처음 보는 불량 12장)
  python src/incremental.py register --name edge-burn --images defects/edge_burn/*.png
  # 분류 (미등록 유형이면 unknown-new-defect 반환 → 등록 유도)
  python src/incremental.py classify --image query.png
  # 리플레이 버퍼 구축/갱신
  python src/incremental.py build-replay --root datasets/fab --per-class 100
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

DEFAULT_ENCODER = "vit_base_patch14_reg4_dinov2.lvd142m"


class DefectTypeRegistry:
    """few-shot 결함 유형 분류기 (프로토타입 + kNN, 학습 없음)."""

    def __init__(self, store: Path, encoder: str = DEFAULT_ENCODER,
                 unknown_threshold: float = 0.55, device: str | None = None):
        import torch
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.unknown_threshold = unknown_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder_name = encoder
        self._model = None
        self._tf = None
        self.protos: dict[str, np.ndarray] = {}   # name -> (k, D) shot 임베딩
        self._load()

    # ---------- encoder ----------
    def _ensure_model(self):
        if self._model is not None:
            return
        import timm
        import torch
        self._model = timm.create_model(self._encoder_name, pretrained=True,
                                        num_classes=0).eval().to(self.device)
        cfg = timm.data.resolve_model_data_config(self._model)
        self._tf = timm.data.create_transform(**cfg, is_training=False)
        torch.set_grad_enabled(False)

    def embed(self, image_paths: list[str]) -> np.ndarray:
        from PIL import Image
        import torch
        self._ensure_model()
        feats = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            x = self._tf(img).unsqueeze(0).to(self.device)
            f = self._model(x)
            f = torch.nn.functional.normalize(f, dim=-1)
            feats.append(f.squeeze(0).cpu().numpy())
        return np.stack(feats)

    # ---------- persistence ----------
    def _load(self):
        meta = self.store / "registry.json"
        if meta.exists():
            names = json.loads(meta.read_text())["names"]
            for n in names:
                self.protos[n] = np.load(self.store / f"{n}.npy")

    def _save(self):
        (self.store / "registry.json").write_text(
            json.dumps({"names": sorted(self.protos)}, ensure_ascii=False, indent=2))
        for n, v in self.protos.items():
            np.save(self.store / f"{n}.npy", v)

    # ---------- API ----------
    def register(self, name: str, image_paths: list[str]) -> int:
        """신규(또는 기존) 불량 유형에 shot 추가. 즉시 분류 가능해진다."""
        emb = self.embed(image_paths)
        self.protos[name] = (np.concatenate([self.protos[name], emb])
                             if name in self.protos else emb)
        self._save()
        return len(self.protos[name])

    def classify(self, image_path: str, topk: int = 3):
        if not self.protos:
            return {"label": "unknown-new-defect", "score": 0.0, "topk": []}
        q = self.embed([image_path])[0]
        scored = []
        for name, bank in self.protos.items():
            sims = bank @ q                      # 코사인 (정규화 완료)
            k = min(3, len(sims))
            scored.append((name, float(np.sort(sims)[-k:].mean())))
        scored.sort(key=lambda t: -t[1])
        best, s = scored[0]
        label = best if s >= self.unknown_threshold else "unknown-new-defect"
        return {"label": label, "score": s, "topk": scored[:topk]}


# ---------------------------------------------------------------- replay
def build_replay(root: Path, out: Path, per_class: int, seed: int = 0) -> None:
    """클래스별 정상 이미지 경로 N개를 보존 (Dinomaly 리플레이 재파인튜닝용)."""
    import random
    rng = random.Random(seed)
    buf: dict[str, list[str]] = {}
    for cls in sorted(p for p in root.iterdir() if (p / "train/good").exists()):
        imgs = [str(p) for p in (cls / "train/good").glob("*")]
        rng.shuffle(imgs)
        buf[cls.name] = imgs[:per_class]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(buf, ensure_ascii=False, indent=1))
    print(f"[replay] {len(buf)} classes × ≤{per_class} → {out}")
    print("신규 클래스 추가 시: (리플레이 전 클래스 + 신규 클래스)로 "
          "train_multiclass_dinomaly.py 를 --epochs 3~5 로 짧게 재실행하세요. "
          "전면 재학습이 아니라 체크포인트에서 이어 학습하면 수 시간 내 완료됩니다.")


def onboard_new_class_patchcore(class_dir: Path) -> None:
    """PatchCore 운영 시 신규 제품 클래스 온보딩 = 그 클래스만 뱅크 구축."""
    import subprocess
    import sys
    print(f"[onboard] {class_dir.name}: PatchCore 피처 뱅크 구축 (gradient 학습 없음)")
    subprocess.run([sys.executable, "src/train_baseline_patchcore.py",
                    "--root", str(class_dir.parent), "--classes", class_dir.name],
                   check=True)


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="신규 불량 유형 few-shot 등록")
    r.add_argument("--name", required=True)
    r.add_argument("--images", nargs="+", required=True)
    r.add_argument("--store", type=Path, default=Path("results/defect_registry"))

    c = sub.add_parser("classify", help="결함 크롭의 유형 분류")
    c.add_argument("--image", required=True)
    c.add_argument("--store", type=Path, default=Path("results/defect_registry"))

    b = sub.add_parser("build-replay")
    b.add_argument("--root", type=Path, default=Path("datasets/fab"))
    b.add_argument("--out", type=Path, default=Path("results/replay_buffer.json"))
    b.add_argument("--per-class", type=int, default=100)

    o = sub.add_parser("new-class", help="신규 제품 클래스 온보딩(PatchCore)")
    o.add_argument("--class-dir", type=Path, required=True)

    a = ap.parse_args()
    if a.cmd == "register":
        paths = [p for pat in a.images for p in glob.glob(pat)]
        n = DefectTypeRegistry(a.store).register(a.name, paths)
        print(f"[registry] '{a.name}' 총 {n} shots 등록 완료 — 즉시 분류 가능")
    elif a.cmd == "classify":
        print(json.dumps(DefectTypeRegistry(a.store).classify(a.image),
                         ensure_ascii=False, indent=2))
    elif a.cmd == "build-replay":
        build_replay(a.root, a.out, a.per_class)
    elif a.cmd == "new-class":
        onboard_new_class_patchcore(a.class_dir)


if __name__ == "__main__":
    main()
