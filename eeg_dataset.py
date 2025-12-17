from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowIndex:
    start: int
    section_id: int


def _compute_section_boundaries(section_id: np.ndarray) -> np.ndarray:
    """Return boundary indices [0, ..., N] where section_id changes.

    boundaries[i] .. boundaries[i+1] is a contiguous run (same section_id).
    """
    if section_id.ndim != 1:
        raise ValueError(f"section_id must be 1D, got shape={section_id.shape}")
    if len(section_id) == 0:
        return np.array([0], dtype=np.int64)

    change = np.flatnonzero(section_id[1:] != section_id[:-1]) + 1
    boundaries = np.concatenate([
        np.array([0], dtype=np.int64),
        change.astype(np.int64),
        np.array([len(section_id)], dtype=np.int64),
    ])
    return boundaries


def make_window_index(
    section_id: np.ndarray,
    window_size: int = 2048,
    stride: int = 2048,
) -> list[WindowIndex]:
    """Create window start indices without crossing section boundaries."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    boundaries = _compute_section_boundaries(section_id)
    indices: list[WindowIndex] = []

    for s, e in zip(boundaries[:-1], boundaries[1:]):
        sid = int(section_id[s])
        seg_len = int(e - s)
        if seg_len < window_size:
            continue
        # strictly within [s, e)
        last_start = e - window_size
        starts = range(int(s), int(last_start) + 1, int(stride))
        indices.extend(WindowIndex(start=st, section_id=sid) for st in starts)

    return indices


class EEGWindowDataset(Dataset):
    """Windowed EEG dataset from a single NPZ with raw/clean/section_id.

    Returns tensors shaped (1, window_size).

    Normalization options:
        - per_sample_zscore: z-score raw and clean independently per window.
            (each sample/window uses its own mean/std)
        - per_window_raw_zscore: use raw window mean/std to z-score BOTH raw/clean.
            (keeps raw/clean on the same normalized scale)
    - none: no normalization.
    """

    def __init__(
        self,
        npz_path: str | None,
        indices: list[WindowIndex],
        window_size: int = 2048,
        normalization: Literal["per_sample_zscore", "per_window_raw_zscore", "none"] = "per_sample_zscore",
        eps: float = 1e-6,
        data: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        if data is None and npz_path is None:
            raise ValueError("Either npz_path or data must be provided")

        self.npz_path = npz_path
        self.obj = None

        if data is None:
            obj = np.load(npz_path, allow_pickle=True)
            self.obj = obj
            data = {k: obj[k] for k in obj.files}

        self.raw = data["raw"].astype(np.float32, copy=False)
        self.clean = data["clean"].astype(np.float32, copy=False)
        self.section_id = data["section_id"]

        if len(self.raw) != len(self.clean) or len(self.raw) != len(self.section_id):
            raise ValueError("raw/clean/section_id lengths do not match")

        self.indices = indices
        self.window_size = int(window_size)
        self.normalization = normalization
        self.eps = float(eps)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        wi = self.indices[idx]
        st = wi.start
        ed = st + self.window_size

        raw = self.raw[st:ed]
        clean = self.clean[st:ed]

        if raw.shape[0] != self.window_size or clean.shape[0] != self.window_size:
            raise IndexError("window out of range")

        x = torch.from_numpy(raw).unsqueeze(0)  # (1, T)
        y = torch.from_numpy(clean).unsqueeze(0)

        if self.normalization == "per_sample_zscore":
            mx = x.mean(dim=-1, keepdim=True)
            sx = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
            my = y.mean(dim=-1, keepdim=True)
            sy = y.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
            x = (x - mx) / sx
            y = (y - my) / sy
        elif self.normalization == "per_window_raw_zscore":
            mx = x.mean(dim=-1, keepdim=True)
            sx = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
            x = (x - mx) / sx
            y = (y - mx) / sx
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        return x, y


def split_indices(
    indices: list[WindowIndex],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[WindowIndex], list[WindowIndex]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    order = np.arange(len(indices))
    rng.shuffle(order)

    n_train = int(round(len(indices) * train_ratio))
    train_idx = [indices[i] for i in order[:n_train]]
    test_idx = [indices[i] for i in order[n_train:]]
    return train_idx, test_idx
