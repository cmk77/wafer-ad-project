#!/usr/bin/env python
"""check_gpu.py — 내일 회사 GPU(RTX 6000 Ada 예상)에서 가장 먼저 실행하는 스크립트.

GPU/드라이버/핵심 라이브러리를 점검하고, 본 프로젝트의 워크로드별
실행 가능 여부를 VRAM 기준으로 판정해 출력한다.

    python src/check_gpu.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys

GB = 1024 ** 3

# (워크로드, 필요 VRAM(GB), 비고)
WORKLOADS = [
    ("PatchCore 학습(피처추출)",              8,  "gradient 학습 없음, 클래스당 수 분"),
    ("Dinomaly ViT-B 멀티클래스 학습",        24, "batch 8~16 @ 392~448px"),
    ("EfficientAD/STFPM/FastFlow 학습",       8,  ""),
    ("WM-811K CNN 분류기 학습",               4,  ""),
    ("AD 추론(배포)",                         4,  "OpenVINO/TensorRT 시 CPU도 가능"),
    ("Gemma 4 26B-A4B 4bit 추론",             22, "가중치 ~18GB + KV캐시"),
    ("Gemma 4 31B 4bit 추론",                 26, "가중치 ~20GB + KV캐시"),
    ("Gemma 4 31B QLoRA 파인튜닝",            46, "NF4+grad ckpt, seq 2k, b1. 여유 적음"),
    ("Gemma 4 31B bf16 추론",                 62, "48GB 단일로는 불가 → 4/8bit 사용"),
    ("동시 운용: AD추론 + Gemma 4bit",        28, "한 장으로 공존"),
]


def _nvidia_smi() -> None:
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        print(f"[nvidia-smi] {out}")
    else:
        print("[nvidia-smi] 없음 — NVIDIA 드라이버 설치 여부 확인 필요")


def main() -> int:
    print("=" * 72)
    print("웨이퍼 AD 프로젝트 — GPU/환경 점검")
    print("=" * 72)
    _nvidia_smi()

    try:
        import torch
    except ImportError:
        print("torch 미설치: pip install -r requirements.txt")
        return 1

    print(f"[torch] {torch.__version__}  CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("!! CUDA 사용 불가 — 드라이버/설치 확인. (CPU로는 PatchCore 추론 정도만 실용적)")
        return 1

    dev = torch.cuda.get_device_properties(0)
    total_gb = dev.total_memory / GB
    free_gb = torch.cuda.mem_get_info()[0] / GB
    cap = f"{dev.major}.{dev.minor}"
    print(f"[GPU] {dev.name} | VRAM {total_gb:.1f} GB (free {free_gb:.1f}) | SM {cap}")
    print(f"[bf16] {'지원' if torch.cuda.is_bf16_supported() else '미지원'}"
          f" | [TF32] matmul 허용={torch.backends.cuda.matmul.allow_tf32}")

    # 간단 벤치마크 (bf16 4096^2 matmul)
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
        tflops = 20 * 2 * 4096 ** 3 / (time.time() - t0) / 1e12
        print(f"[bench] bf16 matmul ≈ {tflops:.0f} TFLOPS")
        del a
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"[bench] 스킵: {e}")

    # 선택 라이브러리
    for mod, why in [("anomalib", "트랙 B 이상탐지"), ("timm", "DefectRegistry"),
                     ("bitsandbytes", "Gemma 4bit"), ("peft", "QLoRA"),
                     ("transformers", "Gemma 4"), ("flash_attn", "(선택) 어텐션 가속")]:
        try:
            m = __import__(mod)
            print(f"[lib] {mod} {getattr(m, '__version__', '?')} — OK ({why})")
        except ImportError:
            print(f"[lib] {mod} 미설치 — ({why})")

    print("-" * 72)
    print(f"{'워크로드':<34}{'필요VRAM':>9}   판정")
    print("-" * 72)
    for name, need, note in WORKLOADS:
        ok = total_gb + 0.5 >= need
        mark = "✅" if ok else "❌"
        print(f"{name:<34}{need:>7}GB   {mark}  {note}")
    print("-" * 72)
    print("판정 기준: 카드 총 VRAM. 실제 free 메모리는 위 [GPU] 라인 참고.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
