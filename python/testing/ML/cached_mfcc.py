from pathlib import Path
import torch


class CachedMFCCDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, words, audio_col, word_col,
                 cache_dir,train = True, extractor=None, sample_rate=16000):

        self.hf_dataset = hf_dataset
        self.words = words
        self.audio_col = audio_col
        self.word_col = word_col
        self.cache_dir = Path(cache_dir)
        self.extractor = extractor
        self.sample_rate = sample_rate
        self.train = train
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.word_to_idx = {
            word: i for i, word in enumerate(words)
        }

    def __len__(self):
        return len(self.hf_dataset)

    def _cache_path(self, idx):
        return self.cache_dir / f"{idx:08d}.pt"

    def __getitem__(self, idx):
        cache_path = self._cache_path(idx)

        if cache_path.exists():
            data = torch.load(cache_path, weights_only=True)

            return (
                data["spec"],
                data["label"],
                data["length"],
            )

        # Load audio from HF dataset
        ex = self.hf_dataset[idx]

        audio = ex[self.audio_col]

        # Adjust this depending on the structure returned by your HF dataset
        samples = audio["array"]

        # Compute MFCC
        spec = self.extractor(samples)

        if not isinstance(spec, torch.Tensor):
            spec = torch.from_numpy(spec)

        spec = spec.float()

        label = self.word_to_idx[ex[self.word_col]]
        length = spec.shape[0]

        torch.save({
            "spec": spec,
            "label": label,
            "length": length,
        }, cache_path)

        return spec, label, length