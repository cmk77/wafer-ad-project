#!/usr/bin/env python
"""qlora_finetune.py — Gemma 4 QLoRA 파인튜닝 (사내 결함 리포트 도메인 적응).

RTX 6000 Ada 48GB 단일 GPU 기준:
  - gemma-4-26B-A4B-it (MoE): 4bit 로드 ~18GB + LoRA/옵티마이저 → 여유 있음(권장 시작점)
  - gemma-4-31B-it (dense):   4bit 로드 ~20GB + grad ckpt + seq 2048 → 35~46GB, 빠듯하지만 가능
  - 31B bf16 풀파인튜닝:      불가 (가중치만 62GB)

OOM 시 폴백 순서: seq 2048→1024 → batch accum 감소 → 26B-A4B로 전환.
MoE(26B-A4B)는 라우터/expert를 건드리면 불안정해질 수 있어 attention 모듈만 타깃한다.

데이터 형식(jsonl): {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
  → gemma_reporter.py 가 저장한 .report.json + 엔지니어 검수 결과를 변환해 축적.

    python src/llm/qlora_finetune.py --data data/sft/reports.jsonl \
        --model google/gemma-4-31B-it --out runs/qlora-wafer-v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

# dense(31B): MLP 포함 전체 선형층 타깃 / MoE(26B-A4B): attention만
TARGETS_DENSE = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
TARGETS_MOE = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="jsonl (messages 형식)")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--out", type=Path, default=Path("runs/qlora-wafer-v1"))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    is_moe = "A4B" in a.model or "a4b" in a.model
    targets = TARGETS_MOE if is_moe else TARGETS_DENSE
    print(f"[모델] {a.model}  (MoE={is_moe}, LoRA targets={targets})")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map="auto",
        attn_implementation="eager",  # Gemma 계열 안정 경로. FA2 설치 시 "flash_attention_2"
    )
    model.config.use_cache = False  # gradient checkpointing과 충돌 방지
    tokenizer = AutoTokenizer.from_pretrained(a.model)

    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    )

    ds = load_dataset("json", data_files=str(a.data), split="train")
    print(f"[데이터] {len(ds)}건 로드")

    sft_cfg = SFTConfig(
        output_dir=str(a.out),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,      # 유효 배치 16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=a.seq_len,
        bf16=True,
        optim="paged_adamw_8bit",            # 48GB에서 옵티마이저 메모리 절감
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(a.out / "final"))
    print(f"[완료] LoRA 어댑터 저장: {a.out / 'final'}")
    print("추론 시: PeftModel.from_pretrained(base_4bit, adapter_path) 로 결합,")
    print("또는 merge_and_unload() 후 vLLM 서빙 (--enable-lora 로 어댑터 직접 서빙도 가능)")


if __name__ == "__main__":
    main()
