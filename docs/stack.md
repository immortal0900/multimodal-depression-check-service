# Tech Stack — Multimodal Depression Detection Service

> **담당 범위**: 이미지 모델 학습 · 멀티모달 통합 설계  
> 얼굴 표정 기반 우울증 분류 모델을 직접 학습하고,  
> 이미지 · 음성 · 텍스트 3개 모달리티를 하나의 추론 파이프라인으로 통합했습니다.

---

## 1. Deep Learning & Model Architecture

> 이미지 분류 모델 설계, 학습, 실험에 사용한 핵심 프레임워크

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

| 기술 | 프로젝트 내 활용 |
|------|------------------|
| **ConvNeXt-Base** | 얼굴 표정 → 우울증 이진 분류 (ImageNet Pretrained → Fine-tuning) |
| **Transfer Learning** | Backbone Freezing(Epoch 1~5) → Full Fine-tuning(Epoch 6+) 2단계 전략 |
| **Factory Pattern (BuildModel)** | ResNet18, ConvNeXt(4종), ViT-B/16, EfficientNet-B0 — 7종 모델을 Enum + Factory로 교체 가능하게 설계 |
| **DepressionClassifier** | `ConvNeXt-Base → Dropout(0.5) → Linear(1024→2)` 커스텀 분류기 |

---

## 2. Computer Vision & Face Detection

> 학습 데이터의 얼굴 검출, 전처리, 모델 해석 가능성(XAI)

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logoColor=white)
![GradCAM](https://img.shields.io/badge/Grad--CAM-FF6F00?style=for-the-badge&logoColor=white)

| 기술 | 프로젝트 내 활용 |
|------|------------------|
| **YUNet (DNN)** | 1차 얼굴 검출기 — 높은 정밀도의 DNN 기반 검출 |
| **Haar Cascade** | YUNet 실패 시 Fallback (정면 + 측면 2단계) |
| **회전 보정** | 얼굴 랜드마크 기반 기울기 자동 보정으로 학습 데이터 품질 확보 |
| **Grad-CAM** | 모델 판단 근거 시각화 — 해석 가능한 AI (XAI) |
| **커스텀 Augmentation** | `TrimBorder` · `TimeMask` · `FreqMask` 전처리 클래스 직접 구현 |

---

## 3. Training Pipeline & Optimization

> 학습 효율 극대화, 실험 자동화, 과적합 방지 전략

![Optuna](https://img.shields.io/badge/Optuna-0095D5?style=for-the-badge&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?style=for-the-badge&logo=tensorboard&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

| 기술 | 프로젝트 내 활용 |
|------|------------------|
| **3-Phase Training** | 1 Epoch 테스트 → 하이퍼파라미터 조정 → 본 학습 파이프라인 |
| **Early Stopping** | `patience=8` 기반 과적합 자동 방지 |
| **WeightedRandomSampler** | 우울/비우울 클래스 불균형 해소 |
| **Optuna** | 하이퍼파라미터 자동 탐색 (learning rate, dropout 등) |
| **Face Detection Cache** | JSON 캐시로 반복 학습 시 얼굴 검출 재연산 제거 |
| **TensorBoard** | 학습 곡선 · 손실 함수 실시간 모니터링 |

---

## 4. Multimodal Fusion & Model Orchestration

> 3개 모달리티 모델을 단일 추론 인터페이스로 통합 설계 (담당 핵심)

![LateFusion](https://img.shields.io/badge/Late_Fusion-Weighted_Average-green?style=for-the-badge)
![OOP](https://img.shields.io/badge/Design-Factory_Pattern-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

| 기술 | 프로젝트 내 활용 |
|------|------------------|
| **AutoPredict 클래스** | Image · Sound · Text 3개 모델을 단일 인터페이스(`combind_predict`)로 통합 |
| **Late Fusion** | 각 모달리티 독립 추론 → 가중 평균으로 최종 판정 |
| **Modality 독립 설계** | 개별 모달리티 실패(예: STT 인식 불가) 시에도 나머지로 예측 가능 |
| **모듈 분리 구조** | `combind/image/` · `combind/sound/` · `combind/text/` 독립 패키지 구성 |

> **통합 대상 모달리티**
>
> | 모달리티 | 모델 | 역할 |
> |----------|------|------|
> | Image | ConvNeXt-Base | 얼굴 표정 → 우울증 분류 **(직접 학습)** |
> | Sound | ConvNeXt-Small | 프로소디(멜+피치+RMS) → 감정 → 우울 점수 |
> | Text | KoBERT + Logistic Regression | 발화 텍스트 → 우울증 분류 |

---

## 5. Model Serving & Demo

> 통합된 모델을 서비스로 제공하는 API 및 데모 UI

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-2094F3?style=for-the-badge&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

| 기술 | 프로젝트 내 활용 |
|------|------------------|
| **FastAPI + Lifespan** | Lifespan으로 앱 시작 시 `AutoPredict` 1회 로드 → 매 요청마다 재로드 방지 |
| **POST /predict** | `multipart/form-data` 이미지+오디오 수신 → 가중 평균 연산 → JSON 응답 |
| **Pydantic Response Model** | `PredictionResponse`, `ModalityResult` — 타입 안전 응답 스키마 |
| **Weighted Average Logic** | image(0.4) + sound(0.35) + text(0.25) 가중 평균으로 최종 진단 |
| **File Validation** | 확장자 검사(.png/.jpg/.wav), 파일 크기 제한(10MB), 임시 파일 자동 정리 |
| **Uvicorn ASGI Server** | 비동기 처리로 동시 요청 처리 — Port 8999 |
| **Streamlit UI** | 파일 업로드 분석 + 실시간 카메라/마이크 분석 2-Tab 구성 |
| **Health Check** | `/health` 엔드포인트로 서버 상태 모니터링 |

---

## 6. Development Environment

> 개발 환경 및 도구

![Python](https://img.shields.io/badge/Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logoColor=white)
![dotenv](https://img.shields.io/badge/.env-ECD53F?style=for-the-badge&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logoColor=white)
