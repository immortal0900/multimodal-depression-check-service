# hwa_in/combind/AutoPredict.py
"""
멀티모달 우울증 예측 모듈.

이미지(얼굴 표정), 사운드(멜 스펙트로그램), 텍스트(음성→텍스트)를
결합하여 우울증 여부를 판별합니다.
"""

import logging

import numpy as np
import speech_recognition as sr
import torch
from PIL import Image

from .image.image_model import load_image_model, transform_image
from .sound.sound_model import load_sound_model, wav_to_prosody_tensor
from .text.text_model import clean_text, load_text_model_embedder, wav_to_text

logger = logging.getLogger(__name__)


class AutoPredict:
    """이미지, 사운드, 텍스트를 결합한 우울증 예측 클래스."""

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        text_model, embedder = load_text_model_embedder()
        self.text_model = text_model
        self.text_embedder = embedder

        self.image_model = load_image_model(self.device)
        self.sound_model = load_sound_model()

    def text_predict(self, file_path):
        """음성 파일을 텍스트로 변환하여 우울증을 예측합니다."""
        text = wav_to_text(file_path)
        self.text_model.eval()

        with torch.no_grad():
            try:
                cleaned = clean_text(text)
            except (TypeError, AttributeError) as e:
                # text가 None이거나 잘못된 타입일 경우
                logger.warning(f"텍스트 전처리 실패: {e}")
                return {"pred": 0, "conf": 0}

            emb = self.text_embedder.encode([cleaned], device=self.device, convert_to_tensor=True)
            outputs = self.text_model(emb)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            label = int(probs[1] >= 0.5)   # 0 or 1
            conf = float(np.max(probs))    # 최대 확률값

            return {"pred": label, "conf": conf}

    def image_predict(self, image_path) -> tuple[str, float]:
        image = Image.open(image_path)
        image = image.convert("RGB")
        tf = transform_image()
        input_tensor = tf(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        model = self.image_model
        result = {}
        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        print(probabilities)
        return {"pred":predicted_idx.item(), "conf":confidence.item()}

    def sound_predict(self, sound_file):

        wav_tensor = wav_to_prosody_tensor(sound_file)
        x = wav_tensor.unsqueeze(0).to(self.device)

        self.sound_model.eval()
        score = 0
        pred = 0
        result = {}
        with torch.no_grad():
            logits = self.sound_model(x)
            probs = torch.softmax(logits,dim = 1)
            _, pred = torch.max(probs,dim=1)
            happiness, surprise, neutral, fear, disgust, anger, sadness = probs.tolist()[0]

            POS = (happiness + surprise) / 2
            NEG_w = (0.4 * sadness + 0.3 * neutral + 0.1 * fear + 0.1 * disgust + 0.1 * anger) / 5
            score = NEG_w / (NEG_w + POS + 1e-8)
            pred = int(score > 0.5)
            result = {"pred":pred, "conf":score}
        return result

    def combind_predict(self, image_file, sound_file):
        """이미지, 사운드, 텍스트를 결합하여 최종 우울증 예측을 수행합니다."""
        image_result = self.image_predict(image_file)
        sound_result = self.sound_predict(sound_file)
        text_result = self.text_predict(sound_file)

        # 각 모달리티 결과를 딕셔너리로 결합
        result = {
            "image": image_result,
            "sound": sound_result,
            "text": text_result
        }
        return result
