#!/usr/bin/env python
"""wafermap_classifier.py — 트랙 A: 웨이퍼 맵 패턴 분류 + 증분 클래스 확장.

- 기본 학습: WM-811K 9클래스 (data_prep.py 산출 폴더 사용)
- 신규 패턴 등장 시: 출력층만 확장(기존 가중치 보존) + 리플레이 파인튜닝
  → 전면 재학습 없이 새 패턴 클래스를 추가한다.

    # 기본 학습
    python src/wafermap_classifier.py train --data datasets/wm811k_png --epochs 20
    # 신규 패턴 'Zigzag' 증분 추가 (새 데이터 폴더 + 기존 데이터 리플레이)
    python src/wafermap_classifier.py add-class --ckpt results/wm_cls/best.pt \
        --new-name Zigzag --new-dir datasets/new_patterns/Zigzag \
        --replay-data datasets/wm811k_png --epochs 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMG_SIZE = 64


# ------------------------------------------------------------------ preprocess
def to_tensor(img: np.ndarray) -> torch.Tensor:
    """그레이스케일 uint8 맵 → (1,H,W) float 텐서.

    학습과 추론(fab_map_ingest.py predict 포함)이 반드시 이 함수를 공유해
    전처리 불일치를 원천 차단한다. 팔레트 보존을 위해 NEAREST만 사용.
    """
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        import cv2
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).unsqueeze(0)


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """WM-811K 9클래스는 모두 회전/반전 불변(방사·형상 위상으로 정의된 패턴)이라
    dihedral(90°×k + 반전) + 자유각 회전이 라벨을 바꾸지 않는 안전한 증강이다.
    팔레트(0/128/255) 값 보존을 위해 INTER_NEAREST만 쓴다."""
    import cv2
    k = rng.randrange(4)
    if k:
        img = np.rot90(img, k)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:  # 자유각 — 스크래치/국부 클러스터 방향 다양화
        ang = rng.uniform(-180.0, 180.0)
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), ang, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return np.ascontiguousarray(img)


# ------------------------------------------------------------------ data
class FolderDataset(Dataset):
    """<root>/<split>/<class>/*.png 를 읽는 단순 데이터셋 (1채널)."""

    def __init__(self, root: Path, split: str, classes: list[str],
                 per_class_limit: int | None = None, seed: int = 0,
                 aug: bool = False):
        self.items: list[tuple[Path, int]] = []
        rng = random.Random(seed)
        for i, c in enumerate(classes):
            files = sorted((root / split / c).glob("*.png"))
            rng.shuffle(files)
            if per_class_limit:
                files = files[:per_class_limit]
            self.items += [(p, i) for p in files]
        rng.shuffle(self.items)
        self.classes = classes
        self.aug = aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import cv2
        p, y = self.items[idx]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.aug:
            img = augment(img, random.Random())  # 워커별 OS 엔트로피 시드
        return to_tensor(img), y

    def sample_weights(self) -> torch.Tensor:
        """WeightedRandomSampler용 샘플별 가중치 = 클래스 빈도 역수의 제곱근.

        순수 역빈도는 Near-full(54장) 같은 극희소 클래스를 매 배치 과다 반복시켜
        과적합·정밀도 붕괴를 부른다. sqrt 완화가 실효 균형점.
        """
        import collections
        cnt = collections.Counter(y for _, y in self.items)
        w = torch.tensor([(1.0 / cnt[y]) ** 0.5 for _, y in self.items])
        return w


# ------------------------------------------------------------------ model
class WaferNet(nn.Module):
    """ResNet18(1채널) 백본 + 교체 가능한 분류 헤드."""

    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    @torch.no_grad()
    def expand_head(self, n_new: int) -> None:
        """출력층에 신규 클래스 노드 추가 — 기존 가중치는 그대로 복사."""
        old = self.head
        new = nn.Linear(self.feat_dim, old.out_features + n_new)
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
        self.head = new.to(old.weight.device)


# ------------------------------------------------------------------ train
def run_epochs(model, loader, val_loader, epochs, lr, device,
               freeze_backbone=False, class_weights=None) -> float:
    """학습 루프. 종료 시 model에 '최고 성능 시점' 가중치를 복원해 두므로
    호출부에서 model.state_dict()를 저장하면 곧 best 체크포인트가 된다.

    선택 기준은 macro-F1 — WM-811K처럼 극단적 불균형에서 raw accuracy는
    다수 클래스(none)에 지배되어 희소 클래스 성능을 반영하지 못한다.
    """
    import copy
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    best_f1, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        acc, mf1 = evaluate(model, val_loader, device)
        mark = ""
        if mf1 > best_f1:
            best_f1, best_state = mf1, copy.deepcopy(model.state_dict())
            mark = "  ← best"
        print(f"  epoch {ep+1}/{epochs}  val_acc={acc:.4f}  macro_F1={mf1:.4f}{mark}",
              flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)  # 마지막이 아닌 '최고' 가중치 확정
    return best_f1


@torch.no_grad()
def evaluate(model, loader, device, details: bool = False):
    """(전체 정확도, macro-F1[, 혼동행렬]) 반환 — 혼동 카운트로 직접 계산."""
    model.eval()
    n_cls = model.head.out_features
    cm = torch.zeros(n_cls, n_cls, dtype=torch.long)  # cm[정답, 예측]
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t, p] += 1
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    present = cm.sum(1) > 0  # 검증셋에 실존하는 클래스만 평균
    acc = float(tp.sum() / cm.sum().clamp(min=1))
    mf1 = float(f1[present].mean())
    return (acc, mf1, cm) if details else (acc, mf1)


def class_weights_from(ds: FolderDataset, device) -> torch.Tensor:
    """WM-811K 불균형 대응 — 역빈도의 제곱근 + 상한 20배 클램프.

    순수 역빈도는 none:희소클래스 가중 비가 수백 배가 되어 모델이 희소
    클래스를 남발(정밀도 붕괴)하게 만든다. sqrt로 완화하고 상한을 둔다.
    """
    import collections
    cnt = collections.Counter(y for _, y in ds.items)
    n = len(ds.classes)
    w = torch.tensor([(len(ds) / (n * cnt.get(i, 1))) ** 0.5 for i in range(n)])
    w = (w / w.min().clamp(min=1e-8)).clamp(max=20.0)
    return w.to(device)


def save_report(out: Path, classes: list[str], cm: torch.Tensor) -> dict:
    """최종 전수 평가 산출물: per_class.csv + confusion.txt. per-class dict 반환."""
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    prec = (tp / (tp + fp).clamp(min=1)).tolist()
    rec = (tp / (tp + fn).clamp(min=1)).tolist()
    f1 = (2 * tp / (2 * tp + fp + fn).clamp(min=1)).tolist()
    n = cm.sum(1).tolist()
    with open(out / "per_class.csv", "w") as f:
        f.write("class,n_test,precision,recall,f1\n")
        for i, c in enumerate(classes):
            f.write(f"{c},{n[i]},{prec[i]:.4f},{rec[i]:.4f},{f1[i]:.4f}\n")
    with open(out / "confusion.txt", "w") as f:
        w = max(len(c) for c in classes) + 1
        f.write(" " * w + "".join(f"{c[:9]:>10}" for c in classes) + "   (행=정답, 열=예측)\n")
        for i, c in enumerate(classes):
            f.write(f"{c:<{w}}" + "".join(f"{int(v):>10}" for v in cm[i]) + "\n")
    return {c: {"n": int(n[i]), "precision": round(prec[i], 4),
                "recall": round(rec[i], 4), "f1": round(f1[i], 4)}
            for i, c in enumerate(classes)}


# ------------------------------------------------------------------ cli
def cmd_train(a) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = sorted(p.name for p in (a.data / "train").iterdir() if p.is_dir())
    print(f"[train] classes({len(classes)}): {classes}")
    tr = FolderDataset(a.data, "train", classes, a.limit, aug=not a.no_aug)
    # 검증 상한: WM-811K test 'none' 11만 장 전수 평가로 에폭이 늘어지는 것 방지.
    # 최종 지표는 학습 종료 후 전수(full) 평가로 따로 산출한다.
    va = FolderDataset(a.data, "test", classes, a.val_limit)
    print(f"[train] train={len(tr):,} (aug={'off' if a.no_aug else 'on'})  "
          f"val={len(va):,} (per-class ≤{a.val_limit})  balance={a.balance}")

    sampler = weights = None
    if a.balance == "sampler":
        sampler = WeightedRandomSampler(tr.sample_weights(), num_samples=len(tr))
    elif a.balance == "weights":
        weights = class_weights_from(tr, device)
    lt = DataLoader(tr, batch_size=a.batch, sampler=sampler, shuffle=sampler is None,
                    num_workers=a.workers, persistent_workers=a.workers > 0)
    lv = DataLoader(va, batch_size=a.batch, num_workers=a.workers)

    model = WaferNet(len(classes)).to(device)
    best = run_epochs(model, lt, lv, a.epochs, a.lr, device, class_weights=weights)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "classes": classes,
                "in_ch": 1, "img_size": IMG_SIZE}, a.out / "best.pt")

    print("[train] 최종 전수 평가 (test 전체) ...", flush=True)
    lf = DataLoader(FolderDataset(a.data, "test", classes),
                    batch_size=max(a.batch, 512), num_workers=a.workers)
    acc, mf1, cm = evaluate(model, lf, device, details=True)
    per_class = save_report(a.out, classes, cm)
    (a.out / "meta.json").write_text(json.dumps(
        {"classes": classes, "best_macro_f1": best,  # 검증(상한) 기준 best
         "final_full_acc": round(acc, 4), "final_full_macro_f1": round(mf1, 4),
         "per_class": per_class}, ensure_ascii=False, indent=2))
    print(f"[train] val best macro_F1={best:.4f} | full test acc={acc:.4f} "
          f"macro_F1={mf1:.4f} → {a.out}/best.pt, per_class.csv, confusion.txt")


def cmd_add_class(a) -> None:
    """증분 확장: 헤드 확장 → (리플레이 + 신규) 짧은 파인튜닝."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    classes: list[str] = ckpt["classes"]
    model = WaferNet(len(classes)).to(device)
    model.load_state_dict(ckpt["state"])
    model.expand_head(1)
    classes = classes + [a.new_name]
    print(f"[add-class] '{a.new_name}' 추가 → 총 {len(classes)} classes")

    # 리플레이(기존 클래스당 소량) + 신규 클래스 데이터 합성 로더
    replay = FolderDataset(a.replay_data, "train", classes[:-1],
                           per_class_limit=a.replay_per_class, aug=True)
    new_ds = _single_class_dataset(a.new_dir, label=len(classes) - 1)
    combo = torch.utils.data.ConcatDataset([replay, new_ds])
    lt = DataLoader(combo, batch_size=a.batch, shuffle=True, num_workers=4)
    lv = DataLoader(FolderDataset(a.replay_data, "test", classes[:-1],
                                  per_class_limit=2000),
                    batch_size=a.batch, num_workers=4)

    # 1단계: 백본 동결로 신규 클래스 안정화 → 2단계: 전체 미세 파인튜닝
    run_epochs(model, lt, lv, max(1, a.epochs // 2), a.lr, device, freeze_backbone=True)
    for p in model.backbone.parameters():
        p.requires_grad = True
    run_epochs(model, lt, lv, a.epochs - a.epochs // 2, a.lr * 0.1, device)

    out = Path(a.ckpt).with_name("best_incremental.pt")
    torch.save({"state": model.state_dict(), "classes": classes,
                "in_ch": 1, "img_size": IMG_SIZE}, out)
    print(f"[add-class] 저장 → {out}")


def _single_class_dataset(folder: Path, label: int) -> Dataset:
    class _DS(Dataset):
        def __init__(self):
            self.files = sorted(Path(folder).glob("*.png"))
        def __len__(self):
            return len(self.files)
        def __getitem__(self, i):
            import cv2
            img = cv2.imread(str(self.files[i]), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            return to_tensor(augment(img, random.Random())), label
    return _DS()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, default=Path("results/wm_cls"))
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--limit", type=int, default=None, help="클래스당 학습 상한(빠른 실험)")
    t.add_argument("--val-limit", type=int, default=2000,
                   help="에폭 중 검증 클래스당 상한 (최종 지표는 전수 평가)")
    t.add_argument("--balance", choices=["sampler", "weights", "none"],
                   default="sampler",
                   help="불균형 대응: sampler=sqrt역빈도 리샘플링(권장), weights=손실 가중")
    t.add_argument("--no-aug", action="store_true", help="회전/반전 증강 끄기")
    t.add_argument("--workers", type=int, default=8)

    c = sub.add_parser("add-class")
    c.add_argument("--ckpt", type=Path, required=True)
    c.add_argument("--new-name", required=True)
    c.add_argument("--new-dir", type=Path, required=True)
    c.add_argument("--replay-data", type=Path, required=True)
    c.add_argument("--replay-per-class", type=int, default=300)
    c.add_argument("--epochs", type=int, default=6)
    c.add_argument("--batch", type=int, default=256)
    c.add_argument("--lr", type=float, default=1e-4)

    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_add_class(a)


if __name__ == "__main__":
    main()
