# hwa_in/combind/image/config.py
"""
이미지 모델 설정 모듈.

경로 설정 우선순위:
1. 환경 변수 HWA_IN_IMAGE_MODEL_PATH (설정된 경우)
2. 기본값: hwa_in/model/best_image_model.pth (상대 경로)
"""

import os
from pathlib import Path

import torch


class Settings:
    """애플리케이션의 모든 설정을 관리합니다."""

    # 환경 변수 우선, 없으면 hwa_in/model/ 기준 상대 경로 사용
    # Path(__file__).parents[2] = hwa_in/ (config.py는 hwa_in/combind/image/에 위치)
    _DEFAULT_MODEL_DIR: Path = Path(__file__).resolve().parents[2] / "model"
    _DEFAULT_MODEL_PATH: Path = _DEFAULT_MODEL_DIR / "best_image_model.pth"

    MODEL_PATH: str = os.getenv("HWA_IN_IMAGE_MODEL_PATH", str(_DEFAULT_MODEL_PATH))

    CLASS_NAMES: dict = {
        0: "Non-Depressed",
        1: "Depressed"
    }


settings = Settings()
