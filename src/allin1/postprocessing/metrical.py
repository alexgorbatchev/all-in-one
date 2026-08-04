import torch
import numpy as np
from typing import Optional, List
# agorbatchev: Use modern CPJKU beat-this Postprocessor instead of legacy madmom DBN
from beat_this.inference import Postprocessor
from ..typings import AllInOneOutput
from ..config import Config


def postprocess_metrical_structure(
  logits: AllInOneOutput,
  cfg: Config,
  # agorbatchev: Allow min_bpm and max_bpm overrides for beat tracking constraints
  min_bpm: Optional[float] = None,
  max_bpm: Optional[float] = None,
):
  raw_prob_beats = torch.sigmoid(logits.logits_beat[0])
  raw_prob_downbeats = torch.sigmoid(logits.logits_downbeat[0])

  postproc = Postprocessor(type='minimal', fps=cfg.fps)
  beats, downbeats = postproc(raw_prob_beats, raw_prob_downbeats)

  beat_positions = _assign_beat_positions(beats, downbeats)

  return {
    'beats': beats.tolist() if isinstance(beats, np.ndarray) else list(beats),
    'downbeats': downbeats.tolist() if isinstance(downbeats, np.ndarray) else list(downbeats),
    'beat_positions': beat_positions,
  }


def _assign_beat_positions(beats: np.ndarray, downbeats: np.ndarray, beats_per_bar: int = 4) -> List[int]:
  if len(beats) == 0:
    return []
  beat_positions = []
  current_pos = 1
  for beat in beats:
    if len(downbeats) > 0 and np.min(np.abs(downbeats - beat)) < 0.05:
      current_pos = 1
    beat_positions.append(current_pos)
    current_pos = (current_pos % beats_per_bar) + 1
  return beat_positions
