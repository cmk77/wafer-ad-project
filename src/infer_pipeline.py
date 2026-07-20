#!/usr/bin/env python
"""infer_pipeline.py — 엔드투엔드 검사 파이프라인 오케스트레이터.

흐름:
  [1] anomalib AD 모델 → 이상 점수/히트맵 (판정)
  [2] 히트맵 피크 기준 결함 영역 크롭
  [3] DefectTypeRegistry(DINOv2 few-shot) → 결함 유형 분류 (unknown 포함)
  [4] WaferOntology → 원인/조치 grounding 컨텍스트
  [5] GemmaReporter → 한국어 리포트 + MES용 구조화 JSON

VRAM 동시 상주(48GB 기준): AD(PatchCore) ~6GB + DINOv2 ~2GB + Gemma 4bit ~18-20GB ≈ 28GB → 여유.

    python src/infer_pipeline.py --image sample.png --category fab_A_layerX \
        --ckpt results/patchcore/fab_A_layerX/latest/weights/lightning/model.ckpt \
        --with-gemma
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "llm"))

from incremental import DefectTypeRegistry  # noqa: E402
from llm.ontology import WaferOntology  # noqa: E402


def run_ad(image_path: Path, ckpt: Path, model_name: str = "Patchcore") -> dict:
    """anomalib 체크포인트로 단일 이미지 예측. 점수/마스크 반환."""
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    import anomalib.models as am

    model_cls = getattr(am, model_name)
    model = model_cls()
    engine = Engine()
    dataset = PredictDataset(path=image_path)
    preds = engine.predict(model=model, dataset=dataset, ckpt_path=str(ckpt))
    p = preds[0]

    def _get(obj, *names):
        for n in names:
            v = getattr(obj, n, None) if not isinstance(obj, dict) else obj.get(n)
            if v is not None:
                return v
        return None

    score = _get(p, "pred_score", "pred_scores")
    label = _get(p, "pred_label", "pred_labels")
    amap = _get(p, "anomaly_map", "anomaly_maps")
    score = float(np.asarray(score).reshape(-1)[0]) if score is not None else -1.0
    label = bool(np.asarray(label).reshape(-1)[0]) if label is not None else score > 0.5
    amap = np.asarray(amap).squeeze() if amap is not None else None
    return {"score": score, "is_anomaly": label, "anomaly_map": amap}


def crop_defect(image_path: Path, amap: np.ndarray | None,
                out_path: Path, box: int = 224) -> Path | None:
    """히트맵 최대점 중심으로 결함 영역 크롭. 히트맵 없으면 중앙 크롭."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if amap is not None and amap.ndim == 2:
        ay, ax = np.unravel_index(np.argmax(amap), amap.shape)
        cx, cy = int(ax * w / amap.shape[1]), int(ay * h / amap.shape[0])
    else:
        cx, cy = w // 2, h // 2
    half = box // 2
    l = max(0, min(cx - half, w - box)); t = max(0, min(cy - half, h - box))
    img.crop((l, t, l + box, t + box)).save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--category", required=True, help="제품/공정 클래스명 (로그용)")
    ap.add_argument("--ckpt", type=Path, required=True, help="anomalib 체크포인트")
    ap.add_argument("--ad-model", default="Patchcore")
    ap.add_argument("--registry", type=Path, default=Path("results/defect_registry"))
    ap.add_argument("--ontology", type=Path,
                    default=Path("data/ontology/wafer_defect_ontology.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/inference"))
    ap.add_argument("--with-gemma", action="store_true",
                    help="Gemma 4 리포트까지 생성 (VRAM +18~20GB)")
    ap.add_argument("--gemma-model", default="google/gemma-4-26B-A4B-it")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # [1] AD 판정
    ad = run_ad(a.image, a.ckpt, a.ad_model)
    result: dict = {
        "image": str(a.image), "category": a.category,
        "anomaly_score": round(ad["score"], 4), "is_anomaly": ad["is_anomaly"],
    }
    print(f"[1/5] AD 판정: score={result['anomaly_score']}  이상={ad['is_anomaly']}")

    crop_path = None
    if ad["is_anomaly"]:
        # [2] 결함 크롭
        crop_path = crop_defect(a.image, ad["anomaly_map"],
                                a.out_dir / f"{a.image.stem}_crop.png")
        print(f"[2/5] 결함 크롭: {crop_path}")

        # [3] 유형 분류 (few-shot 레지스트리, unknown이면 신규 유형 후보)
        reg_file = a.registry / "registry.json"
        if reg_file.exists():
            reg = DefectTypeRegistry(a.registry)
            cls_res = reg.classify(str(crop_path))
            dtype, conf = cls_res["label"], cls_res["score"]
        else:
            dtype, conf = "unknown-new-defect", 0.0
            print("      (레지스트리 없음 — incremental.py register 로 유형 등록 필요)")
        result["defect_type"], result["type_confidence"] = dtype, round(conf, 4)
        print(f"[3/5] 유형 분류: {dtype} (conf={conf:.3f})")
        if dtype == "unknown-new-defect":
            print("      → 신규 불량 후보! 엔지니어 확인 후:")
            print(f"        python src/incremental.py register --name <새유형> --images {crop_path} ...")

        # [4] 온톨로지 grounding
        onto = WaferOntology(a.ontology)
        ctx = onto.context_block(dtype)
        result["ontology_hit"] = onto.resolve(dtype) is not None
    else:
        ctx = ""
        result["defect_type"] = "none"

    out_json = a.out_dir / f"{a.image.stem}_result.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[4/5] 결과 저장: {out_json}")

    # [5] Gemma 리포트 (선택) — 가중치 미확보/OOM 시 온톨로지 rule-based 폴백
    if a.with_gemma and ad["is_anomaly"]:
        from llm.gemma_reporter import (GemmaReporter, build_messages,
                                        extract_json, fallback_report)
        parsed = None
        try:
            reporter = GemmaReporter(a.gemma_model)
            msgs = build_messages(result, ctx, str(crop_path) if crop_path else None)
            resp = reporter.generate(msgs, image_path=str(crop_path) if crop_path else None)
            print("\n===== Gemma 4 리포트 =====\n", resp)
            parsed = extract_json(resp)
        except Exception as e:  # noqa: BLE001
            print(f"[경고] Gemma 사용 불가({type(e).__name__}) — rule-based 폴백")
        if parsed is None:
            parsed = fallback_report(result, onto)
        report_path = out_json.with_suffix(".report.json")
        report_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
        print(f"[5/5] MES용 리포트 JSON: {report_path}")
    elif a.with_gemma:
        print("[5/5] 정상 판정 — 리포트 생략")


if __name__ == "__main__":
    main()
