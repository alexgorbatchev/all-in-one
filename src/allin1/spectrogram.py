import numpy as np
import torch
import librosa
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool


def build_log_filterbank(sr: int = 44100, frame_size: int = 2048, num_bands: int = 12, fmin: float = 30.0, fmax: float = 17000.0, fref: float = 440.0) -> torch.Tensor:
  bin_freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)[1:]  # 1024 bins
  
  left = np.floor(np.log2(fmin / fref) * num_bands)
  right = np.ceil(np.log2(fmax / fref) * num_bands)
  freqs = fref * 2.0 ** (np.arange(left, right) / float(num_bands))
  freqs = freqs[np.searchsorted(freqs, fmin):]
  freqs = freqs[:np.searchsorted(freqs, fmax, 'right')]
  
  indices = bin_freqs.searchsorted(freqs)
  indices = np.clip(indices, 1, len(bin_freqs) - 1)
  left_b = bin_freqs[indices - 1]
  right_b = bin_freqs[indices]
  bins = indices - (freqs - left_b < right_b - freqs)
  bins = np.unique(bins)
  
  num_filters = len(bins) - 2
  filterbank = np.zeros((len(bin_freqs), num_filters), dtype=np.float32)
  
  for i in range(num_filters):
    start = int(bins[i])
    center = int(bins[i + 1])
    stop = int(bins[i + 2])
    if start <= center < stop:
      rel_center = center - start
      rel_stop = stop - start
      data = np.zeros(rel_stop, dtype=np.float32)
      data[:rel_center] = np.linspace(0, 1, rel_center, endpoint=False)
      data[rel_center:] = np.linspace(1, 0, rel_stop - rel_center, endpoint=False)
      if data.sum() > 0:
        data = data / data.sum()
      filterbank[start:stop, i] = data
      
  return torch.from_numpy(filterbank)


def _compute_stem_log_spec(audio_path: Path, fb_t: torch.Tensor, sr: int = 44100, frame_size: int = 2048, fps: int = 100) -> np.ndarray:
  y, _ = librosa.load(audio_path, sr=sr, mono=True)
  hop_length = int(sr / fps)
  y_t = torch.from_numpy(y.astype(np.float32))
  stft_complex = torch.stft(
    y_t,
    n_fft=frame_size,
    hop_length=hop_length,
    win_length=frame_size,
    window=torch.hann_window(frame_size),
    center=True,
    return_complex=True
  )
  stft_mag = torch.abs(stft_complex)[1:, :].T  # drop DC bin 0 -> [time, 1024]
  spec = torch.log10(1.0 + torch.matmul(stft_mag, fb_t)).numpy()
  return spec


def extract_spectrograms(demix_paths: List[Path], spec_dir: Path, multiprocess: bool = True):
  todos = []
  spec_paths = []
  for src in demix_paths:
    dst = spec_dir / f'{src.name}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((src, dst))

  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    fb_t = build_log_filterbank()

    # Process all tracks using multiprocessing.
    if multiprocess and len(todos) > 1:
      pool = Pool()
      map_fn = pool.imap
    else:
      pool = None
      map_fn = map

    iterator = map_fn(_extract_spectrogram, [
      (src, dst, fb_t)
      for src, dst in todos
    ])
    for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
      pass

    if pool:
      pool.close()
      pool.join()

  return spec_paths


def _extract_spectrogram(args: Tuple[Path, Path, torch.Tensor]):
  src, dst, fb_t = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  spec_bass = _compute_stem_log_spec(src / 'bass.wav', fb_t)
  spec_drums = _compute_stem_log_spec(src / 'drums.wav', fb_t)
  spec_others = _compute_stem_log_spec(src / 'other.wav', fb_t)
  spec_vocals = _compute_stem_log_spec(src / 'vocals.wav', fb_t)

  # Truncate stems to equal frame length if off by 1 frame
  min_len = min(len(spec_bass), len(spec_drums), len(spec_others), len(spec_vocals))
  spec_bass = spec_bass[:min_len]
  spec_drums = spec_drums[:min_len]
  spec_others = spec_others[:min_len]
  spec_vocals = spec_vocals[:min_len]

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])  # instruments, frames, bins

  np.save(str(dst), spec)
