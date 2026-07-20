#!/usr/bin/env bash
# 웨이퍼 맵 공개 데이터셋 다운로드
# 사용법: bash scripts/download_datasets.sh [저장경로=./datasets]
set -e
DATA_DIR=${1:-./datasets}
mkdir -p "$DATA_DIR" && cd "$DATA_DIR"

echo "== 1) WM-811K (LSWMD.pkl, 약 2GB) =="
# Kaggle CLI 필요: pip install kaggle 후 ~/.kaggle/kaggle.json 토큰 배치
if command -v kaggle >/dev/null 2>&1; then
  kaggle datasets download -d qingyi/wm811k-wafer-map -p wm811k --unzip
else
  echo "  [수동] https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"
  echo "  [원출처] NTU MIR Lab: http://mirlab.org/dataSet/public/  (MIR-WM811K)"
fi
# (대안) PNG 변환본: https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image

echo "== 2) MixedWM38 (38,015장, 38클래스) =="
git clone --depth 1 https://github.com/Junliangwangdhu/WaferMap.git mixedwm38 || true
echo "  repo README의 링크에서 Wafer_Map_Datasets.npz 를 받아 $DATA_DIR/mixedwm38/ 에 두세요."

echo "== 3) 사내 데이터 규약 =="
mkdir -p fab
echo "  fab/<제품클래스>/{train/good, test/good, test/defect}/*.png 형태로 배치"
echo "완료: $DATA_DIR"
