#!/usr/bin/env python
"""run_demo.py — 웨이퍼 맵 판정 + 온톨로지 리포트 E2E 데모 (CPU, 수 초).

동봉 샘플(WM-811K 테스트셋, 파일명 = 정답 라벨) 18장을 분류하고,
결함 판정 1건에 대해 온톨로지 grounding 리포트(JSON)를 생성한다.

    python demo/run_demo.py            # 로컬
    docker run --rm ghcr.io/cmk77/wafer-ad-project:latest   # Docker
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "llm"))

import cv2  # noqa: E402
import torch  # noqa: E402

from llm.gemma_reporter import fallback_report  # noqa: E402
from llm.ontology import WaferOntology  # noqa: E402
from wafermap_classifier import WaferNet, to_tensor  # noqa: E402

CKPT = ROOT / "models" / "wm_cls_best.pt"
SAMPLES = ROOT / "demo" / "samples"
ONTOLOGY = ROOT / "data" / "ontology" / "wafer_defect_ontology.json"


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = WaferNet(len(classes))
    model.load_state_dict(ck["state"])
    model.eval()
    print(f"모델: {CKPT.name} | 클래스 {len(classes)}종: {', '.join(classes)}")

    files = sorted(SAMPLES.glob("*.png"))
    print(f"\n[1/2] 샘플 {len(files)}장 판정 (파일명 = 정답 라벨)")
    print(f"{'파일':<28}{'정답':<12}{'예측 (확률)':<22}판정")
    n_ok = 0
    worst: tuple[float, dict] | None = None  # 가장 강한 결함 신호 → 리포트 대상
    for p in files:
        true = p.name.split("__")[0]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        with torch.no_grad():
            prob = torch.softmax(model(to_tensor(img).unsqueeze(0))[0], 0)
        top1 = int(prob.argmax())
        pred, conf = classes[top1], float(prob[top1])
        ok = pred == true
        n_ok += ok
        print(f"{p.name:<28}{true:<12}{f'{pred} ({conf*100:.1f}%)':<22}{'✓' if ok else '✗'}")
        p_none = float(prob[classes.index("none")]) if "none" in classes else 0.0
        score = 1.0 - p_none
        if pred != "none" and (worst is None or score > worst[0]):
            worst = (score, {"image": str(p), "category": p.stem,
                             "source": "track_A_wafermap",
                             "anomaly_score": round(score, 4), "is_anomaly": True,
                             "defect_type": pred, "type_confidence": round(conf, 4)})
    print(f"→ 정답률 {n_ok}/{len(files)}")

    if worst is None:
        print("\n[2/2] 결함 판정 샘플 없음 — 리포트 생략")
        return
    ad = worst[1]
    onto = WaferOntology(ONTOLOGY)
    print(f"\n[2/2] 결함 리포트 생성: {Path(ad['image']).name} "
          f"→ {ad['defect_type']} (score {ad['anomaly_score']})")
    print("\n--- 온톨로지 grounding 컨텍스트 ---")
    print(onto.context_block(ad["defect_type"]))
    print("\n--- MES 연동용 구조화 리포트 (rule-based; Gemma 4 연동 시 LLM 생성으로 대체) ---")
    print(json.dumps(fallback_report(ad, onto), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
