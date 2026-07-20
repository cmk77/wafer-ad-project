# wafer-ad-project

웨이퍼 맵 불량 패턴 분류(WM-811K 9클래스, ResNet18) + 결함 온톨로지 기반 자동 리포트 파이프라인.

## 1. Docker로 바로 실행 (권장)

```bash
docker run --rm ghcr.io/cmk77/wafer-ad-project:latest
```

동봉된 샘플 웨이퍼 맵 18장을 판정하고, 결함 검출 시 온톨로지 기반 분석 리포트(JSON)까지 출력하는 데모가 실행됩니다. GPU 불필요(CPU 추론).

내 웨이퍼 맵 PNG를 직접 판정하려면:

```bash
docker run --rm -v "$PWD":/work ghcr.io/cmk77/wafer-ad-project:latest \
    src/fab_map_ingest.py predict --png /work/my_map.png --emit-json /work/result.json
```

## 2. 로컬 실행

```bash
# Python 3.10+
pip install torch torchvision opencv-python-headless numpy   # 데모 최소 의존성
python demo/run_demo.py
```

## 3. 전체 데이터로 직접 학습 (선택)

```bash
pip install -r requirements.txt        # 전체 의존성 (anomalib, timm 등)
# Kaggle API 토큰(~/.kaggle/kaggle.json) 필요
bash scripts/download_datasets.sh
python src/data_prep.py wm811k --pkl datasets/wm811k/LSWMD.pkl --out datasets/wm811k_png
python src/wafermap_classifier.py train --data datasets/wm811k_png --epochs 20
```

학습 완료 시 `results/wm_cls/`에 체크포인트(best.pt)·클래스별 F1(per_class.csv)·혼동행렬(confusion.txt)이 생성됩니다. 학습 없이 쓰려면 동봉된 `models/wm_cls_best.pt`(전수 테스트 macro-F1 0.69 / acc 0.95)를 그대로 사용하면 됩니다.
