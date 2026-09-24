import torchaudio
import torch
import csv
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

PAD_ID, EOS_ID = 0, 1


# ---------- text ----------
def build_vocab(root):
    meta = Path(root) / "LJSpeech-1.1" / "metadata.csv"
    with open(meta, encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        chars = sorted({c for r in rows for c in r[2].lower()})
    stoi = {c: i + 2 for i, c in enumerate(chars)}  # 0 = pad, 1 = eos
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text.lower()] + [EOS_ID], dtype=torch.long)


# same spec as mel from my wavenet
SR = 8000
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=1024, hop_length=100, n_mels=80
)


def default_mel_fn(wav):
    return torch.log(_mel(wav).clamp(min=1e-5).squeeze(0).T)


# dataset
class LJSpeechTTS(Dataset):

    def __init__(self, root, stoi, mel_fn=default_mel_fn, cache_dir="data/mels"):
        torchaudio.datasets.LJSPEECH(root=root, download=True)  # download only
        # load wavs ourselves: torchaudio.load needs torchcodec + ffmpeg
        self.wav_dir = Path(root) / "LJSpeech-1.1" / "wavs"
        with open(self.wav_dir.parent / "metadata.csv", encoding="utf-8") as f:
            self.rows = list(csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE))

        self.stoi, self.mel_fn = stoi, mel_fn
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path = self.cache / f"{idx}.pt"
        if path.exists():
            mel, norm_text = torch.load(path)

        else:
            file_id, _, norm_text = self.rows[idx]
            wav, sr = sf.read(self.wav_dir / f"{file_id}.wav", dtype="float32")
            wav = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
            wav = torchaudio.functional.resample(wav, sr, SR)
            mel = self.mel_fn(wav)
            torch.save((mel, norm_text), path)
        return encode(norm_text, self.stoi), mel


def collate(batch):
    texts, mels = zip(*batch)
    text_lens = torch.tensor([len(t) for t in texts])
    mel_lens = torch.tensor([len(m) for m in mels])

    # (B, T_text)
    texts = pad_sequence(texts, batch_first=True, padding_value=PAD_ID)

    # (B, T_mel, n_mels)
    mels = pad_sequence(mels, batch_first=True)
    stop = torch.zeros(mels.shape[:2])  # (B, T_mel)

    stop[torch.arange(len(mels)), mel_lens - 1] = 1.0
    return texts, text_lens, mels, mel_lens, stop


def get_loaders(root="data", batch_size=16, n_val=300, seed=0):
    Path(root).mkdir(parents=True, exist_ok=True)
    torchaudio.datasets.LJSPEECH(root=root, download=True)  # ensure downloaded first
    stoi, itos = build_vocab(root)
    ds = LJSpeechTTS(root, stoi)
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    train_dl = DataLoader(
        train, batch_size, shuffle=True, collate_fn=collate, num_workers=2
    )
    val_dl = DataLoader(val, batch_size, shuffle=False, collate_fn=collate)
    return train_dl, val_dl, stoi, itos


if __name__ == "__main__":
    train_dl, val_dl, stoi, itos = get_loaders()
    texts, text_lens, mels, mel_lens, stop = next(iter(train_dl))
    print("vocab:", len(stoi) + 2)
    print("texts", texts.shape, "mels", mels.shape, "stop", stop.shape)
