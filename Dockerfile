# wafer-ad-project 데모 이미지 — CPU 추론 전용 (GPU/대용량 LLM 의존성 미포함)
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/cmk77/wafer-ad-project" \
      org.opencontainers.image.description="Wafer map defect classification + ontology report demo" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# CPU 전용 torch 휠 → 이미지 슬림화 (CUDA 런타임 미포함)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir opencv-python-headless numpy

COPY src/ src/
COPY demo/ demo/
COPY models/ models/
COPY data/ data/

# fab_map_ingest predict 의 기본 --ckpt 경로에도 모델 배치
RUN mkdir -p results/wm_cls && cp models/wm_cls_best.pt results/wm_cls/best.pt

ENTRYPOINT ["python"]
CMD ["demo/run_demo.py"]
