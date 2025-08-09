# eval.py

import os
import argparse
import torch
import torchaudio
from tqdm.auto import tqdm
import torch.nn.functional as F

# Metric-specific imports
from frechet_audio_distance import FrechetAudioDistance
from laion_clap.clap_module import create_model
from panns.models import Cnn14_16k

def get_audio_paths(directory):
    """Returns a sorted list of .wav file paths in a directory."""
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')])

def calculate_fad(real_path, generated_path):
    """Calculates the Frechet Audio Distance (FAD)."""
    print("Calculating Frechet Audio Distance (FAD)...")
    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False, 
        use_activation=False,
        verbose=False
    )
    fad_score = frechet.score(real_path, generated_path, dtype="float32")
    return fad_score

def calculate_clap_score(generated_path, prompts_file):
    """Calculates the CLAP score for text-audio alignment."""
    print("Calculating CLAP Score...")
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]

    generated_files = get_audio_paths(generated_path)
    if len(prompts) != len(generated_files):
        raise ValueError(f"Mismatch between prompts ({len(prompts)}) and generated files ({len(generated_files)}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model, _ = create_model(amodel_name='HTSAT-base', tmodel_name='roberta-base', enable_fusion=False, device=device)
    clap_model.eval()

    text_embeds = clap_model.get_text_embedding(prompts, use_tensor=True)
    
    audio_embeds_list = []
    for audio_file in tqdm(generated_files, desc="CLAP: Getting audio embeddings"):
        audio_waveform, _ = torchaudio.load(audio_file)
        audio_embed = clap_model.get_audio_embedding_from_data(x=audio_waveform, use_tensor=True)
        audio_embeds_list.append(audio_embed)
    audio_embeds = torch.cat(audio_embeds_list, dim=0)

    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)

    similarity = torch.diag(text_embeds @ audio_embeds.T).mean().item()
    return similarity * 100

def calculate_kl_divergence(real_path, generated_path):
    """
    Calculates KL Divergence on AudioSet tag predictions from a PANNs model.
    This measures the similarity of sound event distributions.
    """
    print("Calculating KL Divergence on AudioSet tags...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pre-trained audio tagging model
    panns_model = Cnn14_16k(pretrained=True).to(device)
    panns_model.eval()

    def get_mean_distribution(audio_dir):
        audio_files = get_audio_paths(audio_dir)
        all_preds = []
        for audio_file in tqdm(audio_files, desc=f"PANNs: Processing {os.path.basename(audio_dir)}"):
            waveform, sr = torchaudio.load(audio_file)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.to(device)
            
            with torch.no_grad():
                # Get tag probabilities (softmax over logits)
                preds = panns_model(waveform.mean(dim=0, keepdim=True))['clipwise_output'].softmax(dim=-1)
                all_preds.append(preds)
        
        return torch.cat(all_preds, dim=0).mean(dim=0)

    # Get the mean probability distribution for real and generated audio
    p_real = get_mean_distribution(real_path)
    p_gen = get_mean_distribution(generated_path)

    # Calculate KL Divergence: KL(P_real || P_gen)
    # Add a small epsilon to avoid log(0)
    kl_div = F.kl_div(torch.log(p_gen + 1e-9), p_real + 1e-9, reduction='sum').item()
    return kl_div

def main():
    parser = argparse.ArgumentParser(description="Evaluation script for text-to-music generation.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory containing generated WAV files.")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing real WAV files for comparison.")
    parser.add_argument("--prompts_file", type=str, required=True, help="Text file with one prompt per line.")
    args = parser.parse_args()

    # --- Calculate All Metrics ---
    fad_score = calculate_fad(args.real_dir, args.generated_dir)
    clap_score = calculate_clap_score(args.generated_dir, args.prompts_file)
    kl_score = calculate_kl_divergence(args.real_dir, args.generated_dir)

    # --- Print Results ---
    print("\n" + "="*40)
    print("          EVALUATION RESULTS")
    print("="*40)
    print(f"Fréchet Audio Distance (FAD) ↓  : {fad_score:.4f}")
    print(f"CLAP Score (Text-Audio Sim) ↑ : {clap_score:.4f}")
    print(f"KL Divergence (Audio Content) ↓ : {kl_score:.4f}")
    print("="*40)
    print("↓: Lower is better.  ↑: Higher is better.")

if __name__ == "__main__":
    main()
