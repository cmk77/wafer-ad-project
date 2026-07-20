# 웨이퍼 불량 검출 및 신규 불량 적응형 모델 개발 계획서

- 대상: 70종 웨이퍼 제품 라인
- 하드웨어: NVIDIA RTX 6000 Ada Generation (48GB GDDR6) × 1
- 핵심 요구사항: ① 70종 제품 전체의 불량 검출, ② **신규 불량 등장 시 전면 재학습 없이** 추가학습/전이학습으로 대응, ③ Gemma 4 오픈모델 + 온톨로지 기반 지식 계층 연동

---

## 1. 문제 정의 — 두 개의 트랙

"웨이퍼 불량"은 서로 다른 두 문제를 포함하므로 트랙을 분리한다.

| 트랙 | 입력 | 문제 유형 | 대표 데이터 | 접근 |
|---|---|---|---|---|
| **A. 웨이퍼 맵(빈맵) 패턴** | 전기 테스트 결과 맵 (52×52 등) | 공간 패턴 **분류** (Center/Donut/Scratch…) | WM-811K, MixedWM38 | CNN/ViT 분류 + 증분 클래스 확장 |
| **B. 표면 결함 이미지** | 광학/SEM 검사 이미지 | 비지도 **이상탐지** (정상만 학습) | 사내 검사 이미지, (참고: MIIC SEM) | anomalib 기반 AD 모델 |

"70종"은 **제품(공정) 클래스** 축이고, "불량 유형"은 별도 축이다. 트랙 B는 정상 이미지만으로 학습하므로 **라벨링 부담이 거의 없고, 한 번도 본 적 없는 불량도 이상(anomaly)으로 탐지**된다 — 이것이 신규 불량 대응의 1차 방어선이다.

---

## 2. 데이터

### 2.1 공개 벤치마크 (파이프라인 검증용)

| 데이터셋 | 규모 | 용도 | 다운로드 |
|---|---|---|---|
| **WM-811K (LSWMD)** | 811,457장 / 라벨 172,950장(≈21%) / 실결함 25,519장 / 9클래스 | 트랙 A 기준 벤치마크 | Kaggle: `https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map` (LSWMD.pkl), 원출처: NTU MIR Lab `http://mirlab.org/dataSet/public/` |
| **WM-811K 이미지 변환본** | PNG 변환본 | pkl 파싱 생략 시 | `https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image` |
| **MixedWM38** | 38,015장 / 52×52 / 1 정상 + 8 단일 + 29 혼합 = 38클래스 | 혼합 패턴 확장 | `https://github.com/Junliangwangdhu/WaferMap` (Wafer_Map_Datasets.npz) |

주의: MVTec AD는 CC BY-NC-SA(비상업)이므로 벤치마크 참고용으로만 쓰고, **최종 검증은 반드시 사내 데이터**로 한다.

### 2.2 사내 데이터 요구사항 (트랙 B)

- 클래스(제품)당 정상 이미지 **100–500장** 권장 (PatchCore는 50장으로도 동작, Dinomaly는 200장+ 권장)
- 검증용 불량 이미지는 클래스당 10–30장이면 AUROC 산출 가능 (없으면 합성 이상 생성으로 대체)
- 디렉토리 규약: `datasets/fab/<제품클래스>/{train/good, test/good, test/defect}/*.png`

---

## 3. 모델 아키텍처 로드맵 (4단계)

### Phase 0 — Day 1 (내일): 환경 검증 + 기준선
1. `python src/check_gpu.py` 로 GPU/드라이버/라이브러리 판정
2. WM-811K 다운로드 → `data_prep.py` 로 전처리
3. 대표 1–3개 클래스에 **PatchCore** 기준선 학습 (클래스당 수 분, gradient 학습 없음)
   - 근거: MVTec 이미지 AUROC 99.1%, 재학습 불필요(피처 뱅크 방식), anomalib(Apache-2.0) 통합
4. WM-811K 9클래스 CNN 분류기 1회 학습 (트랙 A 기준선)

### Phase 1 — 1~4주: 70종 전체 커버
- 후보 비교: ① PatchCore per-class × 70 vs ② **Dinomaly 멀티클래스 1모델**
- **권고: Dinomaly** — 멀티클래스 MVTec 99.6%(ViT-L 99.8%), 한 모델로 70클래스 커버 → 운영/저장 비용 급감. anomalib v2.1+에 통합, Apache-2.0
- 판단 기준: 사내 데이터 이미지 AUROC ≥ 0.97 미달 클래스는 PatchCore 개별 모델로 보완 (하이브리드 운영)
- 로컬라이제이션이 중요한 클래스(SEM류)는 STFPM 병행 평가 (MIIC 벤치마크에서 픽셀 F1 최상)
- 초고속 인라인 요구 시 EfficientAD(~2ms) 검토

### Phase 2 — 1~3개월: 신규 불량/신규 클래스 적응 체계 (4절 참조)

### Phase 3 — 2~4개월: Gemma 4 + 온톨로지 지식 계층 (5절 참조)

### 콜드스타트 보조 트랙
신제품 초기(정상 데이터조차 부족)에는 **WinCLIP zero-shot → few-shot** 으로 트리아지만 수행하고, 데이터가 모이면 본 트랙으로 승격. CLIP류는 반도체 도메인 사전지식이 약하므로 최종 판정이 아닌 **사전선별 용도**로 한정.

---

## 4. 신규 불량 대응 전략 (핵심 요구사항) — 3레벨 설계

| 레벨 | 상황 | 대응 | 재학습 비용 |
|---|---|---|---|
| **L1** | 한 번도 본 적 없는 불량 발생 | 비지도 AD가 정상 분포 이탈로 **자동 탐지** | **0** (설계상 재학습 불필요) |
| **L2** | 그 불량에 이름을 붙이고 자동 분류하고 싶음 | `DefectTypeRegistry` — DINOv2 임베딩 프로토타입에 신규 유형을 **few-shot(5~20장) 등록**, kNN/코사인 매칭으로 즉시 분류 | **~10분** (gradient 학습 없음) |
| **L3-a** | 신규 **제품 클래스** 추가 (PatchCore 운영 시) | 해당 클래스만 신규 피처 뱅크 구축 | **수 분** (피처 추출뿐) |
| **L3-b** | 신규 제품 클래스 추가 (Dinomaly 운영 시) | **리플레이 파인튜닝**: 기존 클래스당 100장 리플레이 버퍼 + 신규 데이터로 짧게 재파인튜닝 | 수 시간 (전면 재학습 아님) |
| **L3-c** | 중기 고도화 | UCAD(frozen 백본+프롬프트, 과거 데이터 불필요) / IUF(증분 통합 재구성) 파일럿 | 신규분만 |

- 트랙 A(웨이퍼 맵)의 신규 패턴: `wafermap_classifier.py`의 **증분 헤드 확장**(기존 가중치 유지 + 출력층 확장) + 리플레이 파인튜닝 + 능동학습(불확실 샘플만 라벨 요청)으로 라벨링 최소화 (Shim et al. 방식)
- 주의: 순수 연속학습(UCAD/IUF)의 절대 성능은 아직 PatchCore/Dinomaly보다 낮을 수 있으므로, **PatchCore+리플레이 절충안과 반드시 병행 벤치마크** 후 채택

---

## 5. Gemma 4 + 온톨로지 연동 설계

### 5.1 역할 분담 원칙
- **탐지·판정 = 전용 비전 AD 모델** (정확도·지연시간·감사가능성 때문에 LLM에 맡기지 않음)
- **해석·보고·원인추론·엔지니어 Q&A = Gemma 4 + 온톨로지** (지식 계층)

### 5.2 파이프라인
```
검사 이미지 ─→ AD 모델(Dinomaly/PatchCore) ─→ 이상 스코어 + 히트맵 + 결함 크롭
                                              │
                       DefectTypeRegistry ────┤ 결함 유형 (예: scratch, 신뢰도 0.91)
                                              ▼
                 온톨로지 조회: 유형 → 공정단계 → 원인 후보 → 조치 (서브그래프)
                                              ▼
        Gemma 4 (4bit) : [AD 결과 + 온톨로지 컨텍스트 (+ 결함 크롭 이미지)] 입력
                                              ▼
        한국어 결함 리포트 / 원인 가설 순위 / MES 연동용 구조화 JSON / 엔지니어 Q&A
```

### 5.3 온톨로지 스키마 (시드: `data/ontology/wafer_defect_ontology.json`)
`불량유형 —(발생공정)→ 공정단계 —(사용장비)→ 장비군`, `불량유형 —(원인후보)→ 근본원인 —(권장조치)→ 조치`
- 시작은 JSON 그래프, 확장 시 RDF/OWL(rdflib) 또는 Neo4j GraphRAG로 이관
- Gemma 출력은 온톨로지에 **grounding** 시켜 환각을 억제 (컨텍스트에 없는 원인은 "추정"으로 명시하도록 시스템 프롬프트 강제)

### 5.4 모델 선택 (48GB 단일 GPU 기준)

Gemma 4 확인 정보(2026-07-08 검증): 2026-03-31 릴리스(4-02 공개 발표), **Apache 2.0**, 전 모델 멀티모달(텍스트+이미지 입력), 26B-A4B/31B는 **256K 컨텍스트**. 26B-A4B는 MoE(전문가 128 + 상시활성 공유 1, 토큰당 8개 활성 ≈ 활성 4B)라 dense 31B보다 훨씬 빠르다. 하이브리드 어텐션(슬라이딩 윈도+글로벌, unified KV, p-RoPE)으로 장문맥 KV 메모리가 절감되어 온톨로지 서브그래프+검사 로그를 대량 주입하기 좋다.

| 용도 | 권장 | 이유 |
|---|---|---|
| 실시간 리포트/Q&A 서빙 | **Gemma 4 26B-A4B-it (4bit)** | MoE, 활성 ~4B → 4B급 속도, ~18GB |
| 고품질 오프라인 분석/파인튜닝 베이스 | **Gemma 4 31B-it (4bit/QLoRA)** | dense 최고 품질 (bf16 원본 62.6GB) |
| bf16 무양자화가 꼭 필요할 때 | Gemma 4 12B-it (2026-06 추가) | ~24GB로 48GB에 bf16 네이티브 탑재 가능, 26B-A4B와 유사 성능 |
| 도메인 어댑터 | QLoRA (r=16, attention+MLP) | 기존 QLoRA 어댑터 프로젝트와 동일 파이프라인 재사용 |

**Hugging Face 리포지토리 (정확한 ID)**

| 모델 | HF ID | 비고 |
|---|---|---|
| 31B instruct (bf16) | `google/gemma-4-31B-it` | safetensors 62.6GB — 48GB 단일 bf16 불가, 4bit 로드용 원본 |
| 26B MoE instruct | `google/gemma-4-26B-A4B-it` | 서빙 기본 권장 |
| 31B 공식 QAT 4bit | `google/gemma-4-31B-it-qat-q4_0-gguf` (llama.cpp/Ollama), `google/gemma-4-31B-it-qat-w4a16-ct` (vLLM 네이티브) | **QAT라 bf16급 품질 유지하며 메모리 ~1/4** — 서빙은 사후 NF4보다 이쪽 우선 |
| 12B instruct | `google/gemma-4-12B-it` | 인코더-프리 Unified 멀티모달 |

- Gemma 4는 멀티모달(텍스트+이미지)이므로 결함 크롭을 **직접 이미지 입력**해 시각적 설명 생성 가능 (31B는 비디오도 프레임 시퀀스로 처리)
- 서빙: vLLM (`--reasoning-parser gemma4 --tool-call-parser gemma4`), 또는 transformers 4bit
- 라이선스: Apache 2.0 → 팹 사내 상용 배포 가능 (anomalib과 라이선스 정합)

---

## 6. GPU 요구사항 분석 — RTX 6000 Ada (48GB) 판정

| 워크로드 | 예상 VRAM | 판정 |
|---|---|---|
| PatchCore 학습(=피처추출, WRN-50) | 4–8GB | ✅ 여유 |
| Dinomaly ViT-B 학습 (batch 8–16, 392–448px) | 12–24GB | ✅ |
| EfficientAD / STFPM / FastFlow 학습 | 2–8GB | ✅ |
| WM-811K CNN 분류기 학습 | <4GB | ✅ |
| AD 추론 (배포) | 2–4GB | ✅ |
| Gemma 4 26B-A4B 4bit 추론 | ~18GB + KV | ✅ 활성 4B라 응답도 빠름 |
| Gemma 4 31B 4bit 추론 (공식 QAT 권장) | ~20GB + KV | ✅ |
| Gemma 4 12B **bf16** 추론 | ~24GB + KV | ✅ 무양자화가 필요하면 이 크기로 |
| Gemma 4 31B **bf16** 추론 | ~62GB | ❌ 단일 48GB 불가 (80GB H100급 필요) → 4/8bit 사용 |
| Gemma 4 31B **QLoRA 파인튜닝** (NF4+grad ckpt, seq 2k, b1×accum) | 35–46GB | ✅ 가능하나 여유 적음 — OOM 시 26B-A4B 또는 seq 1024 |
| **동시 운용**: AD 추론(4GB) + Gemma 4bit(20GB) | ~24–28GB | ✅ 한 장으로 공존 가능 |

결론: **내일 RTX 6000 Ada 한 장으로 Phase 0 전체 + Gemma 4 4bit 데모까지 실행 가능.** 유일한 제약은 31B의 비양자화(bf16) 구동/풀파인튜닝뿐이다.

Day 1 실무 체크리스트: ① `nvidia-smi`로 드라이버(CUDA 12.x 지원, R535 이상) 확인 → ② `python src/check_gpu.py` → ③ 디스크 여유 150GB+ (WM-811K ~2GB, Gemma 4 26B-A4B bf16 원본 ~52GB / 31B ~62.6GB — 4bit로 로드해도 **다운로드는 원본 크기**이므로 사내망 대역폭 확인, 공식 QAT 4bit본을 받으면 ~16-20GB로 절감) → ④ 사내 프록시에서 huggingface.co·kaggle.com 접근 가능 여부 확인, 막혀 있으면 집에서 받아 반입.

---

## 7. 일정 및 마일스톤

| 주차 | 마일스톤 | 산출물 |
|---|---|---|
| W1 (Day1 포함) | 환경검증, 데이터 확보, PatchCore 3클래스 + WM-811K 분류기 기준선 | AUROC 리포트 |
| W2–4 | 사내 데이터 수집 규약 확정, 70클래스 PatchCore vs Dinomaly 비교 | 모델 선정 보고 |
| W5–8 | Dinomaly 70클래스 운영화, OpenVINO/TensorRT 배포, 임계값 캘리브레이션 | 인라인 파일럿 |
| W9–12 | L2/L3 증분 체계 가동 (DefectRegistry, 리플레이), UCAD/IUF 파일럿 | 신규불량 대응 SOP |
| W13–16 | 온톨로지 v1 + Gemma 4 리포트 봇, QLoRA 도메인 어댑터 학습 | 결함 리포트 자동화 |

---

## 8. 리스크와 완화

1. **벤치마크 라이선스**: MVTec은 비상업 → 사내 데이터로 재검증 필수 (anomalib 자체는 Apache-2.0으로 상용 가능)
2. **연속학습 성능 미성숙**: UCAD/IUF는 파일럿으로만, 운영은 PatchCore+리플레이/Dinomaly 리플레이로
3. **클래스 불균형·미세결함**: 고해상도 타일링, 결함 크롭 오버샘플링, 픽셀 지표(F1/PRO) 병행
4. **PatchCore 메모리뱅크 비대화**(70클래스): coreset 1–10% 서브샘플링, 초과 시 Dinomaly로 통합
5. **LLM 환각**: 온톨로지 grounding + 구조화 JSON 출력 + "판정은 AD 모델, LLM은 해석" 원칙 + 사람 최종 승인
6. **도메인 갭(SEM)**: CLIP류 zero-shot은 트리아지 한정, 도메인 적응(SEM-CLIP 방식) 별도 검토

## 9. 성공 지표 (KPI)
- 클래스별 이미지 AUROC ≥ 0.97 (미달 클래스 리스트 관리)
- 신규 제품 클래스 온보딩 소요 < 1시간, 신규 불량 유형 등록 < 10분
- 오탐률(정상 과검) ≤ 목표 수율 조건, 리포트 유용성 휴먼평가 ≥ 4/5
