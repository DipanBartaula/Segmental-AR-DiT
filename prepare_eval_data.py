import os
import torchaudio
from tqdm.auto import tqdm

# Local imports
from config import get_config
from data_loader import MusicCapsDataset

def prepare_data(config, num_samples=20):
    """
    Saves a subset of the test data (real audio and prompts)
    to be used for evaluation.
    """
    print("Preparing evaluation dataset...")
    
    eval_dir = "eval_data"
    real_audio_dir = os.path.join(eval_dir, "real")
    prompts_file = os.path.join(eval_dir, "prompts.txt")

    os.makedirs(real_audio_dir, exist_ok=True)

    # We need the original audio, not the processed mel spectrograms.
    # So we re-initialize the dataset but process it differently here.
    test_dataset_raw = MusicCapsDataset(config, split="test").dataset

    if len(test_dataset_raw) < num_samples:
        num_samples = len(test_dataset_raw)
        print(f"Warning: Test set is smaller than num_samples. Using {num_samples} samples.")

    with open(prompts_file, "w") as f:
        for i in tqdm(range(num_samples), desc="Saving real audio and prompts"):
            item = test_dataset_raw[i]
            
            # Save prompt
            prompt = item['caption'].strip()
            f.write(f"{prompt}\n")
            
            # Save original audio
            audio_array = item['audio']['array']
            sample_rate = item['audio']['sampling_rate']
            
            # Ensure audio is float32 and 2D tensor for saving
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            output_path = os.path.join(real_audio_dir, f"real_{i:03d}.wav")
            torchaudio.save(output_path, audio_tensor, sample_rate)

    print(f"Successfully saved {num_samples} real audio samples and prompts to '{eval_dir}'.")

if __name__ == "__main__":
    config = get_config()
    # You can change the number of samples for evaluation here
    prepare_data(config, num_samples=20)
