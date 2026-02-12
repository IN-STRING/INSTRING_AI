import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")

class GuitarStyleDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sr=22050,
        duration=4,
        n_mels=128
    ):
        """
        root_dir/
          ├─ arpeggio/
          ├─ fingerstyle/
          └─ stroke/
        """
        self.root_dir=root_dir
        self.sr=sr
        self.duration=duration
        self.n_mels=n_mels
        self.target_len=sr*duration
        
        self.label_map={
            "arpeggio": 0,
            "fingerstyle": 1,
            "stroke": 2
        }
        self.samples = []
        self._load_file_list()

    def _load_file_list(self):
        for label_name, label_idx in self.label_map.items():
            class_dir = os.path.join(self.root_dir, label_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(".wav"):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label_idx))

    def __len__(self):
        return len(self.samples)

    def _load_wav(self, path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)

        if len(y) > self.target_len:
            y = y[:self.target_len]
        else:
            y = np.pad(y, (0, self.target_len - len(y)))

        return y

    def _wav_to_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        y = self._load_wav(path)
        mel = self._wav_to_mel(y)

        # (1, n_mels, time)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return mel, label
    
if __name__ == "__main__":
    dataset = GuitarStyleDataset(DATASET_DIR)
    x, y = dataset[0]
    print(x.shape)
    print(y)
