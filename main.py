"""
FastAPI 서버 - 멀티모달 우울증 예측 API

엔드포인트:
- POST /predict: 이미지 + 오디오 파일을 입력받아 3개 모델의 통합 예측 결과 반환
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hwa_in.combind.AutoPredict import AutoPredict

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 가중치 설정 (API_FLOW.md 기준)
WEIGHTS = {"image": 0.4, "sound": 0.35, "text": 0.25}

# 허용 파일 확장자
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTENSIONS = {".wav"}

# 최대 파일 크기 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


# Response 스키마
class ModalityResult(BaseModel):
    prediction: int
    percentage: float


class IndividualResults(BaseModel):
    image: ModalityResult
    sound: ModalityResult
    text: ModalityResult


class PredictionResponse(BaseModel):
    final_prediction: str
    depression_percentage: float
    risk_level: str
    individual_results: IndividualResults


# Lifespan: 앱 시작 시 모델 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 AutoPredict 모델을 1회 로드하여 재사용"""
    logger.info("Loading AutoPredict models...")
    try:
        predictor = AutoPredict()
        app.state.predictor = predictor
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    logger.info("Shutting down server...")


# FastAPI 앱 생성
app = FastAPI(
    title="Multimodal Depression Check API",
    description="이미지, 음성, 텍스트를 결합한 우울증 예측 API",
    version="1.0.0",
    lifespan=lifespan,
)


def validate_file(file: UploadFile, allowed_extensions: set, file_type: str) -> None:
    """파일 유효성 검사"""
    if not file.filename:
        raise HTTPException(status_code=400, detail=f"{file_type} 파일명이 없습니다")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"{file_type} 파일 형식이 올바르지 않습니다. 허용: {', '.join(allowed_extensions)}",
        )


def calculate_weighted_prediction(results: dict) -> dict:
    """가중 평균 기반 최종 예측 계산"""
    # 가중 평균 점수 계산 (0~1 범위)
    score = (
        WEIGHTS["image"] * results["image"]["conf"]
        + WEIGHTS["sound"] * results["sound"]["conf"]
        + WEIGHTS["text"] * results["text"]["conf"]
    )

    # 퍼센티지 변환
    depression_percentage = round(score * 100, 1)

    # 최종 진단
    final_prediction = "우울증 의심" if score > 0.5 else "정상"

    # 위험도 계산
    if score >= 0.7:
        risk_level = "높음"
    elif score >= 0.4:
        risk_level = "중간"
    else:
        risk_level = "낮음"

    return {
        "final_prediction": final_prediction,
        "depression_percentage": depression_percentage,
        "risk_level": risk_level,
    }


def convert_to_response_format(results: dict, weighted_result: dict) -> dict:
    """AutoPredict 결과를 API 응답 형식으로 변환"""
    return {
        "final_prediction": weighted_result["final_prediction"],
        "depression_percentage": weighted_result["depression_percentage"],
        "risk_level": weighted_result["risk_level"],
        "individual_results": {
            "image": {
                "prediction": results["image"]["pred"],
                "percentage": round(results["image"]["conf"] * 100, 1),
            },
            "sound": {
                "prediction": results["sound"]["pred"],
                "percentage": round(results["sound"]["conf"] * 100, 1),
            },
            "text": {
                "prediction": results["text"]["pred"],
                "percentage": round(results["text"]["conf"] * 100, 1),
            },
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="얼굴 이미지 파일 (.png, .jpg, .jpeg)"),
    audio: UploadFile = File(..., description="음성 파일 (.wav)"),
):
    """
    멀티모달 우울증 예측 엔드포인트

    Args:
        image: 얼굴 이미지 파일
        audio: 음성 WAV 파일

    Returns:
        PredictionResponse: 최종 진단 및 개별 모델 결과
    """
    # 파일 유효성 검사
    validate_file(image, ALLOWED_IMAGE_EXTENSIONS, "이미지")
    validate_file(audio, ALLOWED_AUDIO_EXTENSIONS, "오디오")

    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    try:
        # 이미지 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            content = await image.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="이미지 파일이 너무 큽니다 (최대 10MB)")
            temp_img.write(content)
            image_path = temp_img.name
            logger.info(f"Image saved to {image_path}")

        # 오디오 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="오디오 파일이 너무 큽니다 (최대 10MB)")
            temp_audio.write(content)
            audio_path = temp_audio.name
            logger.info(f"Audio saved to {audio_path}")

        # AutoPredict로 예측 수행
        predictor: AutoPredict = app.state.predictor
        results = predictor.combind_predict(image_path, audio_path)
        logger.info(f"Prediction results: {results}")

        # 가중 평균 계산
        weighted_result = calculate_weighted_prediction(results)

        # 응답 형식 변환
        response = convert_to_response_format(results, weighted_result)

        return JSONResponse(content=response, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")
    finally:
        # 임시 파일 삭제
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
            logger.info(f"Deleted temp image: {image_path}")
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
            logger.info(f"Deleted temp audio: {audio_path}")


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "multimodal-depression-check"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8999)
