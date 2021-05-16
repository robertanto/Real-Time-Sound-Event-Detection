import numpy as np
import librosa

from .features import mel
from . import params

def preprocess_input(waveform: np.ndarray, sr: int):
    inp = waveform if sr == params.SAMPLE_RATE else librosa.resample(
        waveform, sr, params.SAMPLE_RATE)

    return mel(waveform,params.SAMPLE_RATE)
