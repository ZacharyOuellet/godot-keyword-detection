"""
Torch Dataset wrapper around the HF dataset repo
'AkashPrasadMishra/multilingual-keywords-900', producing MFCCs via
mfcc.MFCCExtractor.

The repo is *not* a structured HF dataset with audio/word/language
columns -- it's raw files laid out as:

    <lang>/<word>/sample_001.wav
    <lang>/<word>/sample_002.wav
    ...

So instead of `load_dataset(repo_id)` + column-name guessing, we
snapshot-download the repo (or point at an existing local copy with
`local_dir=`) and walk that folder structure ourselves to build the
dataset, with `word` and `language` taken directly from the path.
"""

import os
import re
import random
import zipfile
from collections import Counter

import torch
from torch.utils.data import Dataset

from datasets import Dataset as HFDataset, Audio
from huggingface_hub import snapshot_download

from ML.mfcc import make_extractor

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

# Filenames look like:
#   cs_bracketed_1__-8vdmTPjby-00015-00005998-00006070.wav
#   <locale>_<word>[_<occurrence#>]__<source-clip id>.<ext>
# This is inferred from a single example -- check the printed summary in
# load_keywords_dataset()'s output against a wider sample before trusting it.
FILENAME_RE = re.compile(
    r"^(?P<locale>[^_]+)_(?P<word>[^_]+?)(?:_(?P<occurrence>\d+))?__(?P<clip_id>.+)$"
)


def _extract_if_needed(root):
    """
    If `root` contains a top-level .zip (as this repo does -- the audio
    files are shipped as a single archive), extract it once into a
    sibling `<zip_stem>_extracted/` folder and return that path.
    Idempotent: skips extraction if already done.
    """
    zips = [f for f in os.listdir(root) if f.lower().endswith(".zip")]
    if not zips:
        return root
    if len(zips) > 1:
        print(f"[hf_dataset] multiple zips found ({zips}), using the first one")
    zip_path = os.path.join(root, zips[0])
    extract_dir = os.path.join(root, os.path.splitext(zips[0])[0] + "_extracted")
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"[hf_dataset] using previously extracted data at {extract_dir!r}")
        return extract_dir
    print(f"[hf_dataset] extracting {zip_path!r} -> {extract_dir!r} (one-time)")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _find_audio_files(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                paths.append(os.path.join(dirpath, fname))
    return paths


def _parse_records(paths):
    records = []
    unparsed = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = FILENAME_RE.match(stem)
        if not m:
            unparsed.append(stem)
            continue
        records.append(
            {
                "audio": p,
                "word": m.group("word"),
                "language": m.group("locale"),
            }
        )

    if unparsed:
        print(
            f"[hf_dataset] WARNING: {len(unparsed)}/{len(paths)} filenames didn't "
            f"match the expected <locale>_<word>[_<n>]__<clip_id> pattern. "
            f"Examples: {unparsed[:5]}"
        )

    if records:
        langs = sorted(set(r["language"] for r in records))
        sample_words = sorted(set(r["word"] for r in records))[:10]
        print(
            f"[hf_dataset] parsed {len(records)} files -> "
            f"{len(langs)} locales {langs}, "
            f"{len(set(r['word'] for r in records))} unique words "
            f"(sample: {sample_words})"
        )
    return records


def _walk_dataset_dir(root):
    """
    Recursively walk `root` for audio files and treat the two path
    components immediately above each file as (language, word) --
    i.e. .../<language>/<word>/<file>. This doesn't assume `root`
    itself is the language-folder level, so it's fine if the repo
    snapshot has an extra wrapper directory in between.
    """
    records = []
    broken_symlinks = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        audio_files = [
            f for f in filenames if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]
        if not audio_files:
            continue
        for fname in sorted(audio_files):
            language, word = fname.split("_")[0:2]
            fpath = os.path.join(dirpath, fname)
            if os.path.islink(fpath) and not os.path.exists(fpath):
                broken_symlinks += 1
                continue
            records.append({"audio": fpath, "word": word, "language": language})

    if broken_symlinks:
        print(
            f"[hf_dataset] WARNING: skipped {broken_symlinks} broken symlinks -- "
            f"the local HF cache may be missing blob data. Try re-downloading "
            f"with `huggingface-cli download {os.path.basename(root)} --repo-type dataset`."
        )
    return records


def _print_dir_tree(root, max_depth=3, max_entries=15):
    print(f"[hf_dataset] directory tree under {root!r}:")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep) if dirpath != root else 0
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(
            f"{indent}{os.path.basename(dirpath) or dirpath}/ "
            f"({len(filenames)} files)"
            + (f" e.g. {filenames[:2]}" if filenames else "")
        )
        if len(dirnames) > max_entries:
            print(
                f"{indent}  ... ({len(dirnames)} subfolders total, showing first {max_entries})"
            )
            dirnames[:] = dirnames[:max_entries]


def load_keywords_dataset(
    repo_id="AkashPrasadMishra/multilingual-keywords-900",
    local_dir=None,
    sample_rate=16000,
):
    """
    Downloads (or reuses) a local copy of the dataset repo, walks the
    lang/word/*.wav folder structure, and returns
    (dataset, audio_col, word_col, lang_col) where dataset is an HF
    `Dataset` with an `audio` column decoded at `sample_rate`.

    Pass `local_dir` if you've already cloned/downloaded the repo
    yourself and want to skip the network call.
    """
    root = local_dir or snapshot_download(repo_id=repo_id, repo_type="dataset")
    root = _extract_if_needed(root)
    records = _walk_dataset_dir(root)
    if not records:
        _print_dir_tree(root)
        raise RuntimeError(
            f"No audio files found under {root!r} (see tree printed above). "
            f"Expected .../<lang>/<word>/*.{{wav,flac,mp3,ogg}} somewhere under "
            f"this root -- if the tree looks empty or has broken-looking entries, "
            f"the local cache may be incomplete; try re-downloading with "
            f"`huggingface-cli download {repo_id} --repo-type dataset` first and "
            f"pass its output path in as local_dir=."
        )

    languages = sorted(set(r["language"] for r in records))
    print(
        f"[hf_dataset] found {len(records)} audio files across "
        f"{len(languages)} languages: {languages}"
    )

    ds = HFDataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    return ds, "audio", "word", "language"


def select_words(
    ds, word_col, min_count=100, max_words=None, languages=None, lang_col=None
):
    """Pick classes with at least `min_count` samples, optionally filtered by language."""
    if languages and lang_col:
        ds = ds.filter(lambda ex: ex[lang_col] in languages)
    counts = Counter(ds[word_col])
    counts = {w: c for w, c in counts.items() if c >= min_count}
    words = sorted(counts, key=counts.get, reverse=True)
    if max_words:
        words = words[:max_words]
    return words


# --------------------------------------------------------------------------
# waveform augmentation (same as dataset.py, kept local to avoid import cycles)
# --------------------------------------------------------------------------


def _add_noise(waveform, snr_db_range=(5, 20)):
    noise = torch.randn_like(waveform)
    snr_db = random.uniform(*snr_db_range)
    sig_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean() + 1e-8
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * (target_noise_power / noise_power).sqrt()
    return waveform + noise


def _random_gain(waveform, db_range=(-6, 6)):
    gain_db = random.uniform(*db_range)
    return waveform * (10 ** (gain_db / 20))


class HFKeywordsDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        words,
        audio_col,
        word_col,
        sample_rate=16000,
        train=True,
        extractor=None,
        frame_length=512,
        hop_length=256,
        num_mel_bands=40,
        num_coeffs=13,
    ):
        self.word_to_idx = {w: i for i, w in enumerate(words)}
        self.ds = hf_dataset.filter(lambda ex: ex[word_col] in self.word_to_idx)
        self.audio_col = audio_col
        self.word_col = word_col
        self.sample_rate = sample_rate
        self.train = train
        self.extractor = extractor or make_extractor(
            sample_rate=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            num_mel_bands=num_mel_bands,
            num_coeffs=num_coeffs,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        waveform = torch.tensor(ex[self.audio_col]["array"], dtype=torch.float32)
        label = self.word_to_idx[ex[self.word_col]]

        if self.train:
            waveform = _random_gain(waveform)
            if random.random() < 0.5:
                waveform = _add_noise(waveform)

        mfcc = self.extractor(waveform.numpy())  # [num_coeffs, n_frames]
        if mfcc.shape[-1] == 0:
            # clip shorter than one frame -- pad up to a single frame's worth
            pad = self.extractor.frame_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, max(pad, 0)))
            mfcc = self.extractor(waveform.numpy())

        return torch.from_numpy(mfcc).unsqueeze(0), label  # [1, num_coeffs, n_frames]


def pad_collate(batch):
    """Pads variable-length [num_coeffs, T] MFCC tensors to the batch max T."""
    specs, labels, _lengths_in = zip(*batch)
    max_t = max(s.shape[-1] for s in specs)
    num_coeffs = specs[0].shape[-2]
    out = torch.zeros(len(specs), 1, num_coeffs, max_t)
    lengths = torch.zeros(len(specs), dtype=torch.long)
    for i, s in enumerate(specs):
        t = s.shape[-1]
        out[i, :, :, :t] = s
        lengths[i] = t
    return out, torch.tensor(labels, dtype=torch.long), lengths
