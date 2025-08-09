# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# IMPORTANT: Paste your WandB API key here.
# You can get it from your wandb.ai/settings page.
WANDB_API_KEY="08a4c57edfe8bc0393a2a7f093adf84e2a3b8986"

echo "================================================="
echo "Installing dependencies..."
echo "================================================="
pip install -q -r requirements.txt

# Check if the API key is set
if [ "$WANDB_API_KEY" == "YOUR_WANDB_API_KEY_HERE" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! WARNING: Please set your WANDB_API_KEY in the script.  !!!"
    echo "!!! Training will proceed without logging to WandB.          !!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
else
    echo "================================================="
    echo "Logging into Weights & Biases..."
    echo "================================================="
    wandb login $WANDB_API_KEY
fi


echo "================================================="
echo "Starting training process..."
echo "================================================="
# The 'accelerate launch' command can help with distributed training in the future
# but for a single GPU, 'python' is sufficient.
python train.py

echo "================================================="
echo "Training script finished."
echo "================================================="
