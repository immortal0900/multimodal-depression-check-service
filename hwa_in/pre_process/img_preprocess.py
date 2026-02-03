# hwa_in/pre_process/img_preprocess.py
"""이미지 전처리 유틸리티 모듈."""

import random

import numpy as np
from PIL import Image


class TrimBorder:
    """이미지 가장자리를 고정 마진만큼 잘라냅니다."""

    def __init__(self, margin=10):
        self.margin = margin

    def __call__(self, img):
        width, height = img.size
        margin = self.margin
        return img.crop((margin, margin, width - margin, height - margin))


def compute_mean_std(image_paths):
    """
    이미지 경로 리스트로부터 평균과 표준편차를 계산합니다.

    Grayscale 변환 후 0~1 범위로 정규화하여 계산합니다.

    Args:
        image_paths: 이미지 파일 경로 리스트

    Returns:
        tuple: (평균, 표준편차)
    """
    sum_mean_values = 0.0
    sum_squared_mean = 0.0
    image_count = 0

    for path in image_paths:
        pixel_array = np.array(
            Image.open(path).convert("L"), dtype=np.float32
        ) / 255.0
        sum_mean_values += pixel_array.mean()
        sum_squared_mean += (pixel_array ** 2).mean()
        image_count += 1

    mean = sum_mean_values / image_count
    std = (sum_squared_mean / image_count - mean ** 2) ** 0.5
    return float(mean), float(std)

class TimeMask:
    """가로 방향 띠(mask): 시간 축 마스킹을 적용합니다."""

    def __init__(self, max_width=0.15, probability=0.5, fill_value=0.0):
        self.max_width = max_width
        self.probability = probability
        self.fill_value = fill_value

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        width, height = img.size
        band_width = int(width * random.uniform(0.03, self.max_width))
        x_start = random.randint(0, max(0, width - band_width))

        # 사각형 덮기
        fill_color = tuple(int(255 * self.fill_value) for _ in range(3))
        mask = Image.new("RGB", (band_width, height), fill_color)
        img.paste(mask, (x_start, 0))
        return img


class FreqMask:
    """세로 방향 띠(mask): 주파수 축 마스킹을 적용합니다."""

    def __init__(self, max_height=0.2, probability=0.5, fill_value=0.0):
        self.max_height = max_height
        self.probability = probability
        self.fill_value = fill_value

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        width, height = img.size
        band_height = int(height * random.uniform(0.03, self.max_height))
        y_start = random.randint(0, max(0, height - band_height))

        fill_color = tuple(int(255 * self.fill_value) for _ in range(3))
        mask = Image.new("RGB", (width, band_height), fill_color)
        img.paste(mask, (0, y_start))
        return img
