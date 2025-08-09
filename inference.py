import torch
import torchaudio
from transformers import T5EncoderModel, T5Tokenizer
from tqdm.auto import tqdm
import argparse

# Local imports
from config import get_config
from model import SegmentalAR_DiT

def generate_audio(model, text_prompt, config, vocoder):
    """
    Generates audio from a text prompt using the trained model.
    """
    print(f"Generating music for prompt: '{text_prompt}'")
    model.eval()
    
    # 1. Prepare Text Conditioning
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    text_encoder = T5EncoderModel.from_pretrained("t5-small").to(config.device).eval()
    
    inputs = tokenizer(text_prompt, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    with torch.no_grad():
        text_cond = text_encoder(input_ids=inputs.input_ids.to(config.device)).last_hidden_state
    
    # 2. Autoregressive Generation with Euler Scheduler
    generated_segments = []
    num_segments_to_gen = config.max_mel_len // config.segment_mel_len
    
    for i in tqdm(range(num_segments_to_gen), desc="Generating Segments"):
        # Initial noise for the current segment
        x_t = torch.randn(1, 1, config.n_mels, config.segment_mel_len, device=config.device)
        
        # Prepare conditioning from previously generated segments
        prev_segments_cond = torch.cat(generated_segments, dim=-1) if len(generated_segments) > 0 else None

        # Euler discrete solver for the flow ODE
        time_steps = torch.linspace(1, 1e-4, config.num_inference_steps, device=config.device)
        dt = -1.0 / config.num_inference_steps

        for t in time_steps:
            with torch.no_grad():
                time_tensor = t.expand(1)
                # Predict velocity v(x_t, t)
                velocity = model(x_t, time_tensor, text_cond, prev_segments_cond)
                # Euler step: x_{t-dt} = x_t + dt * v
                x_t = x_t + velocity * dt
        
        generated_segments.append(x_t)

    # 3. Combine segments and convert to audio
    full_mel = torch.cat(generated_segments, dim=-1)
    
    # Denormalize (using training mean/std would be better, this is an approximation)
    full_mel_denorm = full_mel * 16.0 + 4.5 

    with torch.no_grad():
        waveform = vocoder(full_mel_denorm.to(config.device)).cpu()

    return waveform

def main():
    parser = argparse.ArgumentParser(description="Inference script for Segmental AR DiT Music Generation")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for music generation.")
    parser.add_argument("--output_path", type=str, default="generated_music.wav", help="Path to save the output WAV file.")
    
    args = parser.parse_args()
    
    # --- Setup ---
    config = get_config()
    device = config.device
    
    # --- Load Model ---
    model = SegmentalAR_DiT(config).to(device)
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")
    
    # --- Load Vocoder ---
    vocoder = torchaudio.pipelines.HIFIGAN_VOCODER_V1.get_model().to(device)
    
    # --- Generate Audio ---
    waveform = generate_audio(model, args.prompt, config, vocoder)
    
    # --- Save Audio ---
    torchaudio.save(args.output_path, waveform, config.sample_rate)
    print(f"Generated audio saved to {args.output_path}")


if __name__ == '__main__':
    main()
