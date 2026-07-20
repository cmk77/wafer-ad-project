#!/usr/bin/env python
"""gemma_reporter.py — Gemma 4 기반 결함 해석/리포트 생성 (48GB 단일 GPU).

역할 분담(PLAN.md 5절): 판정은 AD 모델이, 해석·보고·원인추론은 Gemma 4가 담당.
온톨로지 컨텍스트로 grounding 하고, MES 연동을 위한 구조화 JSON을 강제한다.
Gemma 4는 멀티모달이므로 결함 크롭 이미지를 직접 입력할 수 있다.

    python src/llm/gemma_reporter.py --ad-json sample_ad_result.json \
        --model google/gemma-4-26B-A4B-it --image crop.png

VRAM 가이드(4bit NF4): 26B-A4B ≈ 18GB, 31B ≈ 20GB (+KV 캐시).
대량 서빙은 vLLM 권장:
    vllm serve google/gemma-4-26B-A4B-it \
        --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ontology import WaferOntology  # noqa: E402

SYSTEM_PROMPT = """당신은 반도체 팹의 수율/결함 분석 엔지니어 어시스턴트입니다.
규칙:
1. 판정(불량 여부)은 이미 비전 AD 모델이 내렸습니다. 판정을 뒤집지 마세요.
2. 원인 분석은 제공된 [온톨로지] 컨텍스트에 근거하세요. 컨텍스트에 없는
   원인을 언급할 때는 반드시 "추정:" 접두어를 붙이세요.
3. 마지막에 아래 스키마의 JSON 블록을 정확히 출력하세요.
```json
{"defect_type": str, "severity": "low|medium|high|critical",
 "confidence": float, "probable_causes": [str], "suspect_process_steps": [str],
 "recommended_actions": [str], "escalation_required": bool, "summary_ko": str}
```"""


def build_messages(ad_result: dict, onto_ctx: str, image_path: str | None):
    user_text = (
        f"검사 결과를 분석해 결함 리포트를 작성하세요.\n\n"
        f"[AD 모델 출력]\n{json.dumps(ad_result, ensure_ascii=False, indent=2)}\n\n"
        f"{onto_ctx}\n"
    )
    if image_path:
        content = [{"type": "image", "image": image_path},
                   {"type": "text", "text": user_text}]
    else:
        content = user_text
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


class GemmaReporter:
    def __init__(self, model_id: str, load_in_4bit: bool = True,
                 enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.enable_thinking = enable_thinking
        self.processor = AutoProcessor.from_pretrained(model_id)
        kwargs: dict = dict(device_map="auto")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = "auto"  # 31B bf16은 48GB 단일 불가 — 4bit 유지 권장
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def generate(self, messages, image_path: str | None = None,
                 max_new_tokens: int = 1024) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        if image_path:
            from PIL import Image
            inputs = self.processor(text=text, images=Image.open(image_path),
                                    return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=text,
                                    return_tensors="pt").to(self.model.device)
        n_in = inputs["input_ids"].shape[-1]
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95, top_k=64,  # Gemma 4 권장값
        )
        raw = self.processor.decode(out[0][n_in:], skip_special_tokens=False)
        try:  # thinking 블록 등 파싱 (transformers의 Gemma 4 지원)
            return self.processor.parse_response(raw)
        except Exception:  # noqa: BLE001
            return self.processor.decode(out[0][n_in:], skip_special_tokens=True)


def extract_json(text) -> dict | None:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    try:
        start, end = s.index("{"), s.rindex("}") + 1
        return json.loads(s[start:end])
    except Exception:  # noqa: BLE001
        return None


def fallback_report(ad_result: dict, onto: WaferOntology) -> dict:
    """LLM 없이 온톨로지만으로 만드는 rule-based 리포트 (동일 JSON 스키마).

    용도: ① Gemma 가중치 미확보/로드 실패 시 파이프라인 무중단,
         ② LLM 출력 검증용 기준선(온톨로지 밖 내용이 없는 리포트).
    """
    dtype = ad_result.get("defect_type", "unknown")
    g = onto.subgraph(dtype)
    conf = float(ad_result.get("type_confidence", ad_result.get("anomaly_score", 0.0)))
    if not ad_result.get("is_anomaly", True):
        return {"defect_type": "none", "severity": "low", "confidence": conf,
                "probable_causes": [], "suspect_process_steps": [],
                "recommended_actions": [], "escalation_required": False,
                "summary_ko": "정상 판정 — 조치 불필요.", "generator": "fallback"}
    if g is None:
        return {"defect_type": dtype, "severity": "high", "confidence": conf,
                "probable_causes": [], "suspect_process_steps": [],
                "recommended_actions": ["신규 유형 검토 후 온톨로지/레지스트리 등록",
                                        "유사 사례 조회(defect registry)"],
                "escalation_required": True,
                "summary_ko": f"미등록 유형 '{dtype}' 탐지 (신뢰도 {conf:.2f}) — "
                              "엔지니어 확인 및 온톨로지 등록 필요.",
                "generator": "fallback"}
    return {
        "defect_type": g["defect"], "severity": g["severity"], "confidence": conf,
        "probable_causes": g["root_causes"],
        "suspect_process_steps": [s["id"] for s in g["process_steps"]],
        "recommended_actions": g["actions"],
        "escalation_required": g["severity"] in ("high", "critical"),
        "summary_ko": f"{g['label_ko']}({g['defect']}) 판정 (신뢰도 {conf:.2f}). "
                      f"시그니처: {g['signature']}. "
                      f"우선 조치: {g['actions'][0] if g['actions'] else '온톨로지 참조'}.",
        "generator": "fallback",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad-json", type=Path, required=True,
                    help="infer_pipeline.py 가 출력한 AD 결과 JSON")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--ontology", type=Path,
                    default=Path("data/ontology/wafer_defect_ontology.json"))
    ap.add_argument("--image", default=None, help="결함 크롭 (멀티모달 입력, 선택)")
    ap.add_argument("--thinking", action="store_true", help="추론 모드 활성화")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="Gemma 미사용 — 온톨로지 rule-based 리포트만 생성")
    a = ap.parse_args()

    ad = json.loads(a.ad_json.read_text(encoding="utf-8"))
    onto = WaferOntology(a.ontology)
    ctx = onto.context_block(ad.get("defect_type", "unknown"))

    parsed = resp = None
    if not a.no_llm:
        try:
            reporter = GemmaReporter(a.model, load_in_4bit=not a.no_4bit,
                                     enable_thinking=a.thinking)
            resp = reporter.generate(build_messages(ad, ctx, a.image),
                                     image_path=a.image)
            print("\n===== Gemma 4 리포트 =====\n", resp)
            parsed = extract_json(resp)
        except Exception as e:  # noqa: BLE001 — 가중치 미확보/OOM 등
            print(f"[경고] Gemma 로드/생성 실패({type(e).__name__}: {e}) — "
                  "rule-based 폴백으로 전환")
    if parsed is None:
        parsed = fallback_report(ad, onto)
        print("\n===== 온톨로지 rule-based 리포트 (LLM 미사용) =====")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print("\n[grounding 컨텍스트]\n" + ctx)

    out = a.ad_json.with_suffix(".report.json")
    out.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    print(f"\n[MES 연동용 JSON 저장] {out}")


if __name__ == "__main__":
    main()
