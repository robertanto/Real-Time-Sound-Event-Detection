import numpy as np
import librosa
from . import params

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def _np_hann_periodic_window(length):
    if length == 1:
      return np.ones(1)
    odd = length % 2
    if not odd:
      length += 1
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / (length - 1))
    if not odd:
      window = window[:-1]
    return window

def _np_frame(data, window_length, hop_length):
    num_frames = 1 + int(np.floor((len(data) - window_length) // hop_length))
    shape = (num_frames, window_length)
    strides = (data.strides[0] * hop_length, data.strides[0])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def _np_stft(data, fft_length, hop_length, window_length):
    frames = _np_frame(data, window_length, hop_length)
    window = _np_hann_periodic_window(window_length)
    return np.fft.rfft(frames * window, fft_length)

def spec(waveform,sr):
    win_samples = int(round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS))
    hop_samples = int(round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
    n_fft = 2 ** int(np.ceil(np.log(win_samples) / np.log(2.0)))

    inp = waveform if sr == params.SAMPLE_RATE else librosa.resample(
        waveform, sr, params.SAMPLE_RATE)

    return np.abs(_np_stft(waveform,n_fft,hop_samples,win_samples))


def hertz_to_mel(frequencies_hertz):
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0,
                              unused_dtype=None):
  
  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)

  band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)

  mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]

    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix

def mel(waveform,sr):
    spectro = spec(waveform,sr)
    mel_basis = spectrogram_to_mel_matrix(
        params.MEL_BANDS,
        spectro.shape[1],
        params.SAMPLE_RATE,
        params.MEL_MIN_HZ,
        params.MEL_MAX_HZ
    )
    return np.log(np.dot(spectro,mel_basis)+params.LOG_OFFSET)