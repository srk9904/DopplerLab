import os
import re
import glob
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from src.features.extraction import extract_1d_features, extract_2d_features, SR

LABEL_RE = re.compile(
    r'(?P<vehicle>[^_]+)_(?P<path>[^_]+)_'
    r'(?P<speed>[\d\.]+)mps_(?P<dist>[\d\.]+)m_'
)
PATH_TO_ID = {"straight": 0, "parabola": 1, "bezier": 2}
ID_TO_PATH = {v: k for k, v in PATH_TO_ID.items()}

def parse_label(fname):
    m = LABEL_RE.search(fname)
    if m is None:
        raise ValueError(f"Unrecognised filename: {fname}")
    return {
        "path":    m.group("path"),
        "path_id": PATH_TO_ID[m.group("path")],
        "speed":   float(m.group("speed")),
        "dist":    float(m.group("dist")),
    }

def get_label_key(wav_path):
    basename = os.path.basename(wav_path)
    try:
        return parse_label(basename)
    except ValueError:
        parent = os.path.basename(os.path.dirname(wav_path))
        return parse_label(parent)

def build_splits(data_root, n_clips=1000, train_r=0.8, val_r=0.1, seed=42):
    all_wav = sorted(
        glob.glob(f"{data_root}/**/*.wav", recursive=True)
      + glob.glob(f"{data_root}/**/*.WAV", recursive=True)
    )
    assert len(all_wav) > 0, f"No .wav files found under: {data_root}"

    parseable = []
    for f in all_wav:
        try:
            get_label_key(f)
            parseable.append(f)
        except ValueError:
            pass

    buckets = defaultdict(list)
    for f in parseable:
        lbl = get_label_key(f)["path"]
        buckets[lbl].append(f)

    rng = random.Random(seed)
    for cls_files in buckets.values():
        rng.shuffle(cls_files)

    per_class_budget = n_clips // len(buckets)
    per_class = min(per_class_budget, min(len(v) for v in buckets.values()))

    balanced = []
    for cls_files in buckets.values():
        balanced.extend(cls_files[:per_class])
    rng.shuffle(balanced)

    n       = len(balanced)
    n_train = int(train_r * n)
    n_val   = int(val_r   * n)

    return (
        balanced[:n_train],
        balanced[n_train : n_train + n_val],
        balanced[n_train + n_val :],
    )

class _DopplerBase(Dataset):
    def __init__(self, files, augment=False):
        self.files   = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _extract(self, wav, wav_path, augment):
        raise NotImplementedError

    def __getitem__(self, idx):
        fpath      = self.files[idx]
        wav, _     = librosa.load(fpath, sr=SR, mono=True)
        if self.augment:
            wav = wav * np.random.uniform(0.85, 1.15)
        x   = self._extract(wav, fpath, self.augment)
        lbl = get_label_key(fpath)
        return (
            torch.from_numpy(x),
            torch.tensor(lbl["path_id"], dtype=torch.long),
            torch.tensor(lbl["speed"],   dtype=torch.float32),
            torch.tensor(lbl["dist"],    dtype=torch.float32),
            os.path.basename(fpath),
        )

class DopplerDataset1D(_DopplerBase):
    def _extract(self, wav, wav_path, augment):
        return extract_1d_features(wav, wav_path, augment=augment)

class DopplerDataset2D(_DopplerBase):
    def _extract(self, wav, wav_path, augment):
        return extract_2d_features(wav, wav_path)
