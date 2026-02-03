# hwa_in/combind/sound/sound_model.py
"""
사운드 기반 우울증 예측 모델.

경로 설정 우선순위:
1. 환경 변수 HWA_IN_SOUND_MODEL_PATH / HWA_IN_MEL_SAVE_DIR (설정된 경우)
2. 기본값: hwa_in/model/, hwa_in/combind/sound/trans_img/ (상대 경로)
"""

import os
from datetime import datetime
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.models as M
from torchvision import transforms as T

from hwa_in.data_model.BuildModel import BuildModel
from hwa_in.data_model.GlobalVariable import SoundVariable
from hwa_in.data_model.ModelType import ModelType
from hwa_in.pre_process.img_preprocess import TrimBorder

# 환경 변수 우선, 없으면 상대 경로 사용
# Path(__file__).parents[2] = hwa_in/ (sound_model.py는 hwa_in/combind/sound/에 위치)
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
_DEFAULT_SOUND_MODEL_PATH = _DEFAULT_MODEL_DIR / "sound_convnext_small.pth"
SOUND_MODEL_PATH = os.getenv("HWA_IN_SOUND_MODEL_PATH", str(_DEFAULT_SOUND_MODEL_PATH))

# 멜 스펙트로그램 저장 경로
_DEFAULT_MEL_SAVE_DIR = Path(__file__).resolve().parent / "trans_img"
MEL_SAVE_DIR = Path(os.getenv("HWA_IN_MEL_SAVE_DIR", str(_DEFAULT_MEL_SAVE_DIR)))


def load_sound_model():
    """사운드 모델을 로드합니다."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(SOUND_MODEL_PATH, map_location="cpu")
    model = BuildModel.get_model(
        ModelType.CONVNEXT_SMALL,
        M.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        7,
        checkpoint,
        device
    )

    return model


def sound_to_image(file):
    """오디오 파일을 멜 스펙트로그램 이미지로 변환합니다."""
    temp_y, sr = librosa.load(file, sr=SoundVariable.HZ)
    y, _ = librosa.effects.trim(temp_y, top_db=SoundVariable.TOP_DB)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=SoundVariable.N_MELS, fmax=SoundVariable.F_MAX
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(SoundVariable.FIG_SIZE_W, SoundVariable.FIG_SIZE_H))
    librosa.display.specshow(S_dB, sr=sr, fmax=SoundVariable.F_MAX)

    # 저장 디렉토리 생성 (없으면)
    MEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = MEL_SAVE_DIR / f"{timestamp}_mel.png"
    plt.savefig(str(save_path))
    plt.close()

    return str(save_path)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 프로소디(운율) 분석용 상수 - SoundVariable과 별개의 용도
PROSODY_SAMPLE_RATE = 16000
PROSODY_N_FFT = 1024
PROSODY_HOP_LENGTH = 256
PROSODY_N_MELS = 64
PROSODY_TOP_DB = 70
PROSODY_FMIN_MEL, PROSODY_FMAX_MEL = 50.0, 8000.0
PROSODY_FMIN_F0, PROSODY_FMAX_F0 = 50.0, 800.0
PROSODY_TARGET_SIZE = 224

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=PROSODY_SAMPLE_RATE,
    n_fft=PROSODY_N_FFT,
    hop_length=PROSODY_HOP_LENGTH,
    n_mels=PROSODY_N_MELS,
    f_min=PROSODY_FMIN_MEL,
    f_max=PROSODY_FMAX_MEL,
    power=2.0,
    norm="slaney",
    mel_scale="htk"
).to(device)
to_db = torchaudio.transforms.AmplitudeToDB(top_db=PROSODY_TOP_DB).to(device)


def _avg_pool_2d(x: torch.Tensor, k_freq: int, k_time: int) -> torch.Tensor:
    """2D 평균 풀링을 수행합니다."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = F.avg_pool2d(x, kernel_size=(k_freq, k_time), stride=(k_freq, k_time), ceil_mode=True)
    return x

def _resize_to(x: torch.Tensor, H: int, T: int) -> torch.Tensor:
    # x: [1, h, t]
    x = F.interpolate(x.unsqueeze(0), size=(H,T), mode="bilinear", align_corners=False).squeeze(0)
    return x

def _f0_to_band(y_pos: np.ndarray, H: int, sigma: float = 1.5) -> np.ndarray:
    # y_pos: [T] (0..H-1 위치), 가우시안 띠
    yy = np.arange(H, dtype=np.float32)[:, None]
    dist2 = (yy - y_pos[None,:])**2
    band = np.exp(-dist2/(2*sigma**2))
    return (band - band.min()) / (band.max()-band.min() + 1e-8)



def wav_to_prosody_tensor(path: str) -> torch.Tensor:
    """WAV 파일을 프로소디(운율) 특성 텐서로 변환합니다."""
    # 0) 로드
    y, sr = librosa.load(path, sr=PROSODY_SAMPLE_RATE, mono=True)
    y_t = torch.from_numpy(y).float().to(device)

    # 1) 멜 (저해상도 + 평활화)
    S = mel_spec(y_t)
    Sdb = to_db(S).clamp_(-PROSODY_TOP_DB, 0.0)
    Smel = (Sdb + PROSODY_TOP_DB) / PROSODY_TOP_DB
    Smel = Smel.unsqueeze(0)

    # 음소 내용 희석: 주파수/시간 풀링으로 블러 + 다시 원래 크기로 리사이즈
    # (k_freq, k_time)는 데이터에 맞춰 조정 가능. 값이 클수록 단어 정보 더 흐림.
    k_freq, k_time = 4, 4
    Smel_low = _avg_pool_2d(Smel, k_freq, k_time)
    Smel_smooth = _resize_to(Smel_low, Smel.shape[1], Smel.shape[2])
    ch_mel = Smel_smooth

    # 2) 피치(F0) -> 띠
    f0, vflag, vconf = librosa.pyin(
        y,
        fmin=PROSODY_FMIN_F0,
        fmax=PROSODY_FMAX_F0,
        sr=sr,
        frame_length=PROSODY_N_FFT,
        hop_length=PROSODY_HOP_LENGTH
    )
    T0 = len(f0)
    valid = ~np.isnan(f0)
    if valid.sum() < 3:
        ch_pitch = torch.zeros_like(ch_mel)
        vprob = torch.zeros(Smel.shape[-1], dtype=torch.float32, device=device)
    else:
        idx = np.arange(T0)
        f0i = np.interp(idx, idx[valid], f0[valid]).astype(np.float32)
        ylog = np.log(np.clip(f0i, PROSODY_FMIN_F0, PROSODY_FMAX_F0))
        y_pos = (ylog - np.log(PROSODY_FMIN_F0)) / (np.log(PROSODY_FMAX_F0) - np.log(PROSODY_FMIN_F0)) * (PROSODY_N_MELS - 1)

        # 시간 정렬: 멜 T와 동일하게
        t_src = np.linspace(0, 1, T0, dtype=np.float32)
        t_dst = np.linspace(0, 1, Smel.shape[-1], dtype=np.float32)
        y_pos = np.interp(t_dst, t_src, y_pos)

        band = _f0_to_band(y_pos, PROSODY_N_MELS, sigma=1.5)  # [H,T] 0..1
        ch_pitch = torch.from_numpy(band).unsqueeze(0).float().to(device)
        vprob = np.nan_to_num(vconf, nan=0.0)
        vprob = np.interp(t_dst, t_src, vprob).astype(np.float32)

    # 3) RMS 에너지 -> 밴드
    rms = librosa.feature.rms(y=y, frame_length=PROSODY_N_FFT, hop_length=PROSODY_HOP_LENGTH)[0]

    # 멜 T와 정렬
    t_src2 = np.linspace(0, 1, len(rms), dtype=np.float32)
    t_dst2 = np.linspace(0, 1, Smel.shape[-1], dtype=np.float32)
    rms_t = np.interp(t_dst2, t_src2, rms)

    # 0..1 정규화 (강세/리듬만)
    rms_t = (rms_t - rms_t.min()) / (rms_t.max() - rms_t.min() + 1e-8)
    ch_rms = torch.from_numpy(
        np.tile(rms_t[None, :], (PROSODY_N_MELS, 1))
    ).unsqueeze(0).float().to(device)  # [1,H,T]

    # 4) 스택 & 리사이즈 -> [3, TARGET, TARGET], 스케일 [-1,1]
    x = torch.cat([ch_mel, ch_pitch, ch_rms], dim=0)  # [3,H,T], 0..1
    x = F.interpolate(
        x.unsqueeze(0),
        size=(PROSODY_TARGET_SIZE, PROSODY_TARGET_SIZE),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    x = x * 2 - 1
    return x
