# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The script will use the first two arguments passed to it.
# $1: Path to the checkpoint file.
# $2: The text prompt for generation (should be in quotes).

CHECKPOINT_PATH=$1
PROMPT=$2
OUTPUT_PATH="generated_music.wav"

# --- Validation ---
if [ -z "$CHECKPOINT_PATH" ] || [ -z "$PROMPT" ]; then
    echo "Usage: ./infer.sh <path_to_checkpoint> \"<your_text_prompt>\""
    echo "Example: ./infer.sh checkpoints/model_step_500.pth \"A beautiful piano melody\""
    exit 1
fi

echo "================================================="
echo "Starting inference..."
echo "================================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Prompt: $PROMPT"
echo "Output will be saved to: $OUTPUT_PATH"
echo "-------------------------------------------------"

python infer.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --prompt "$PROMPT" \
  --output_path "$OUTPUT_PATH"

echo "================================================="
echo "Inference finished. Audio saved to $OUTPUT_PATH"
echo "================================================="
