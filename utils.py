import torch
import torchaudio
import wandb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def log_generation_to_wandb(model, test_dataset, vocoder, config, epoch):
    """
    Generates an audio sample from a test prompt, logs the mel spectrogram
    and the playable audio file to a wandb.Table.
    """
    model.eval()
    
    # Use a fixed item from the test set for consistent comparison across epochs
    _, text_cond, _, text_prompt = test_dataset[0]
    text_cond = text_cond.to(config.device)

    print(f"\nLogging generation for: '{text_prompt}'")

    # --- Autoregressive Generation with Euler Solver ---
    generated_segments = []
    num_segments_to_gen = config.max_mel_len // config.segment_mel_len
    
    for i in tqdm(range(num_segments_to_gen), desc="Logging Generation", leave=False):
        x_t = torch.randn(1, 1, config.n_mels, config.segment_mel_len, device=config.device)
        prev_segments_cond = torch.cat(generated_segments, dim=-1) if len(generated_segments) > 0 else None
        
        time_steps = torch.linspace(1, 1e-4, config.num_inference_steps, device=config.device)
        dt = -1.0 / config.num_inference_steps
        
        for t in time_steps:
            with torch.no_grad():
                time_tensor = t.expand(1)
                velocity = model(x_t, time_tensor, text_cond, prev_segments_cond)
                x_t = x_t + velocity * dt # Euler step
        generated_segments.append(x_t)

    full_mel = torch.cat(generated_segments, dim=-1)
    
    # --- Create Mel Spectrogram Image for WandB ---
    mel_np = full_mel.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Generated Mel Spectrogram (Epoch {epoch})')
    plt.tight_layout()
    mel_image = wandb.Image(plt)
    plt.close() # Important to close the plot to free memory

    # --- Create Audio for WandB ---
    # Denormalize (this is an approximation, better to use saved stats from training set)
    full_mel_denorm = full_mel * 16.0 + 4.5
    with torch.no_grad():
        waveform = vocoder(full_mel_denorm.to(config.device)).cpu()
    
    audio_log = wandb.Audio(waveform.squeeze().numpy(), caption=text_prompt, sample_rate=config.sample_rate)

    # --- Log to a wandb.Table ---
    # Retrieve the table from the run summary or create a new one
    log_table = wandb.run.summary.get("generation_table", None)
    if log_table is None:
        log_table = wandb.Table(columns=["Epoch", "Prompt", "Mel Spectrogram", "Generated Audio"])
    
    log_table.add_data(epoch, text_prompt, mel_image, audio_log)
    wandb.run.summary["generation_table"] = log_table
    
    model.train() # Set model back to training mode
