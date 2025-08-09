import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
import numpy as np

class MusicCapsDataset(Dataset):
    def __init__(self, config, split="train"):
        """
        Initializes the dataset loader for MusicCaps using CLIP for text encoding.

        Args:
            config: The configuration object.
            split (str): The dataset split to load ("train" or "test").
        """
        self.config = config
        
        full_dataset = load_dataset(config.dataset_name, split=split, streaming=True)
        if split == "train":
            subset_size = int(5521 * config.dataset_subset_size)
        else:
            subset_size = 20
        self.dataset = list(full_dataset.take(subset_size))

        # --- Use CLIP Tokenizer and Text Encoder ---
        self.tokenizer = CLIPTokenizer.from_pretrained(config.text_encoder_name)
        self.text_encoder = CLIPTextModel.from_pretrained(config.text_encoder_name).to(config.device).eval()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        ).to(config.device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # --- Audio processing ---
        audio = item['audio']['array'].astype(np.float32)
        audio = torch.from_numpy(audio).to(self.config.device)
        if audio.dim() > 1:
            audio = audio.mean(dim=0)

        mel = self.mel_spectrogram(audio)
        mel = (mel - mel.mean()) / mel.std()

        # --- Padding / Truncating ---
        if mel.shape[1] < self.config.max_mel_len:
            pad_len = self.config.max_mel_len - mel.shape[1]
            mel = F.pad(mel, (0, pad_len))
            mask = torch.cat([torch.ones(mel.shape[1] - pad_len), torch.zeros(pad_len)])
        else:
            mel = mel[:, :self.config.max_mel_len]
            mask = torch.ones(self.config.max_mel_len)
            
        # --- Text processing with CLIP ---
        text = item['caption']
        inputs = self.tokenizer(text, return_tensors="pt", max_length=77, truncation=True, padding="max_length")
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids=inputs.input_ids.to(self.config.device)).last_hidden_state
