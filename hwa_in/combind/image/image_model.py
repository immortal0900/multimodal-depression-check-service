# hwa_in/combind/image/image_model.py
"""이미지 기반 우울증 분류 모델."""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

from .config import settings

# ImageNet 정규화 상수
# 출처: https://pytorch.org/vision/stable/models.html
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DepressionClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # 백본(Backbone)을 ConvNeXt-Base로 교체
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.backbone = convnext_base(weights=weights)

        # ConvNeXt의 분류기는 'classifier'라는 이름의 Sequential 안에 있음
        # 마지막 Linear 레이어의 in_features를 가져옴
        in_features = self.backbone.classifier[-1].in_features
        # 원래 분류기의 마지막 레이어를 제거
        self.backbone.classifier[-1] = nn.Identity()

        # 분류기 정의: 우울/비우울 이진 분류
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self, freeze=True):
        """백본 레이어를 고정하거나 해제합니다."""
        # ConvNeXt의 파라미터를 순회하며 고정/해제
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        # 분류기 부분은 항상 학습 가능하도록 설정
        for param in self.classifier.parameters():
            param.requires_grad = True

def transform_image() -> transforms.Compose:
    """추론에 사용할 이미지 전처리기를 정의합니다."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def load_image_model(device) -> DepressionClassifier:
    """모델 구조를 가져와 학습된 가중치를 로드합니다."""
    try:
        model = DepressionClassifier()
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # 평가 모드로 설정
        print(f"Model loaded successfully from {settings.MODEL_PATH}")
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Error: Model file not found at {settings.MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
