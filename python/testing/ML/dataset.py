"""
Dataset for your local Innu fine-tuning clips: <root>/<word>/*.wav.

Uses mfcc.MFCCExtractor so features match the C++ pipeline (and match
whatever config you pretrained with -- pass the same MFCCExtractor
instance, or matching config, that train_pretrain.py used).

No fixed-length cropping: MSWC pretraining now happens on the
variable-length HF dataset (see hf_dataset.py), so this matches that
same variable-length convention. Batches get padded to the longest
clip via hf_dataset.pad_collate -- see the note in train_finetune.py
about AdaptiveAvgPool2d and zero-padded frames.
"""

import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from python.Testing.ML.mfcc import make_extractor

SAMPLE_RATE = 16000


def spec_augment(
    mfcc, freq_mask_param=2, time_mask_param=8, n_freq_masks=1, n_time_masks=2
):
    """SpecAugment on a [1, num_coeffs, T] MFCC tensor."""
    mfcc = mfcc.clone()
    n_coeffs, t = mfcc.shape[-2], mfcc.shape[-1]
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, n_coeffs - f))
        mfcc[..., f0 : f0 + f, :] = 0
    for _ in range(n_time_masks):
        tm = random.randint(0, min(time_mask_param, t))
        t0 = random.randint(0, max(0, t - tm))
        mfcc[..., :, t0 : t0 + tm] = 0
    return mfcc


def add_noise(waveform, snr_db_range=(5, 20)):
    noise = torch.randn_like(waveform)
    snr_db = random.uniform(*snr_db_range)
    sig_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean() + 1e-8
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * (target_noise_power / noise_power).sqrt()
    return waveform + noise


def random_gain(waveform, db_range=(-6, 6)):
    gain_db = random.uniform(*db_range)
    return waveform * (10 ** (gain_db / 20))


def speed_perturb(waveform, sample_rate, rates=(0.9, 1.0, 1.1)):
    rate = random.choice(rates)
    if rate == 1.0:
        return waveform
    effects = [["speed", str(rate)], ["rate", str(sample_rate)]]
    out, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    return out


class FolderDataset(Dataset):
    """<root>/<word>/*.wav (or any torchaudio-readable format)."""

    def __init__(
        self,
        root,
        words,
        sample_rate=SAMPLE_RATE,
        train=True,
        val_fraction=0.2,
        seed=0,
        extractor=None,
    ):
        self.root = Path(root)
        self.words = list(words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.sample_rate = sample_rate
        self.train = train
        self.extractor = extractor or make_extractor(sample_rate=sample_rate)

        rng = random.Random(seed)
        items = []
        for w in self.words:
            files = sorted((self.root / w).glob("*.*"))
            rng.shuffle(files)
            n_val = max(1, int(len(files) * val_fraction)) if len(files) > 1 else 0
            split_files = files[n_val:] if train else files[:n_val]
            items += [(f, self.word_to_idx[w]) for f in split_files]
        self.items = items
        if len(self.items) == 0:
            raise RuntimeError(f"No clips found under {root} for words={self.words}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.squeeze(0)  # [n_samples]

        if self.train:
            waveform = speed_perturb(waveform.unsqueeze(0), self.sample_rate).squeeze(0)
            waveform = random_gain(waveform)
            if random.random() < 0.5:
                waveform = add_noise(waveform)

        # pad up to at least one frame if the clip is very short
        if waveform.shape[-1] < self.extractor.frame_length:
            pad = self.extractor.frame_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        mfcc = self.extractor(waveform.numpy())  # [num_coeffs, T]
        mfcc = torch.from_numpy(mfcc).unsqueeze(0)  # [1, num_coeffs, T]

        if self.train:
            mfcc = spec_augment(mfcc)

        return mfcc, label
