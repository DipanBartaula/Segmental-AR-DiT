import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm.auto import tqdm
import wandb
import os
import glob

# Local imports
from config import get_config
from data_loader import MusicCapsDataset
from model import SegmentalAR_DiT
from utils import log_generation_to_wandb

def main():
    """Main training function."""
    config = get_config()

    # --- Initialize WandB ---
    wandb.init(project=config.project_name, config=config)
    
    # --- Data ---
    train_dataset = MusicCapsDataset(config, split="train")
    test_dataset = MusicCapsDataset(config, split="test")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- Model, Optimizer, and Flow Matcher ---
    model = SegmentalAR_DiT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    flow_matcher = ConditionalFlowMatcher(sigma=0.01)
    
    # Load a pre-trained vocoder for generation logging
    vocoder = torchaudio.pipelines.HIFIGAN_VOCODER_V1.get_model().to(config.device)

    # --- Checkpoint Loading Logic ---
    start_epoch = 0
    global_step = 0
    checkpoint_files = glob.glob(os.path.join(config.checkpoint_dir, "model_step_*.pth"))
    if checkpoint_files:
        # Find the checkpoint with the highest step number
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        print(f"Resumed from Epoch {start_epoch}, Global Step {global_step}")
    else:
        print("No checkpoint found, starting training from scratch.")

    wandb.watch(model, log="all", log_freq=500)
    
    # --- Main Training Loop ---
    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(train_dataloader, initial=global_step % len(train_dataloader), total=len(train_dataloader), leave=False)
        
        for mel_batch, text_cond_batch, mask_batch, _ in progress_bar:
            model.train()
            mel_batch = mel_batch.unsqueeze(1).to(config.device)
            text_cond_batch = text_cond_batch.to(config.device)
            
            num_segments = config.max_mel_len // config.segment_mel_len
            total_loss = 0
            
            # Autoregressive training over segments
            for i in range(num_segments):
                optimizer.zero_grad()
                start_idx = i * config.segment_mel_len
                end_idx = start_idx + config.segment_mel_len
                target_segment = mel_batch[:, :, :, start_idx:end_idx]
                
                # Skip segments that are entirely padding
                if mask_batch[:, start_idx].sum() == 0: continue
                
                prev_segments = mel_batch[:, :, :, :start_idx] if i > 0 else None
                
                # Sample from flow matcher
                t, u, v = flow_matcher.sample_location_and_conditional_flow(target_segment)
                
                # Get model prediction
                predicted_v = model(u, t, text_cond_batch, prev_segments)
                
                loss = F.mse_loss(predicted_v, v)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / num_segments if num_segments > 0 else 0
            wandb.log({"loss": avg_loss, "epoch": epoch, "global_step": global_step})
            progress_bar.set_description(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}")
            
            # --- Checkpoint Saving Logic ---
            if (global_step + 1) % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f"model_step_{global_step+1}.pth")
                print(f"\nSaving checkpoint to {checkpoint_path}")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

            global_step += 1
        
        # --- End of Epoch Logging ---
        log_generation_to_wandb(model, test_dataset, vocoder, config, epoch + 1)

    print("Training finished.")
    wandb.finish()


if __name__ == '__main__':
    main()

