# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CHECKPOINT_PATH=$1
EVAL_DIR="eval_data"
REAL_AUDIO_DIR="$EVAL_DIR/real"
GEN_AUDIO_DIR="$EVAL_DIR/generated"
PROMPTS_FILE="$EVAL_DIR/prompts.txt"

# --- Validation ---
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: ./eval.sh <path_to_checkpoint>"
    exit 1
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "================================================="
echo "           STARTING EVALUATION PIPELINE"
echo "================================================="

# --- 1. Prepare Evaluation Data ---
echo "STEP 1: Preparing real audio samples and prompts..."
python prepare_eval_data.py

# --- 2. Generate Audio for Each Prompt ---
echo "STEP 2: Generating audio from prompts using checkpoint: $CHECKPOINT_PATH"
rm -rf "$GEN_AUDIO_DIR"
mkdir -p "$GEN_AUDIO_DIR"

i=0
while IFS= read -r prompt; do
  filename=$(printf "generated_%03d.wav" "$i")
  echo "Generating for prompt $i: \"$prompt\""
  python infer.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --prompt "$prompt" \
    --output_path "$GEN_AUDIO_DIR/$filename"
  i=$((i+1))
done < "$PROMPTS_FILE"

# --- 3. Run Evaluation Metrics ---
echo "STEP 3: Installing evaluation dependencies and calculating scores..."
pip install -q laion-clap frechet-audio-distance torch-audiomentations "git+https://github.com/qiuqiangkong/torch_panns.git"

python eval.py \
    --generated_dir "$GEN_AUDIO_DIR" \
    --real_dir "$REAL_AUDIO_DIR" \
    --prompts_file "$PROMPTS_FILE"

echo "================================================="
echo "           EVALUATION COMPLETE"
echo "================================================="
