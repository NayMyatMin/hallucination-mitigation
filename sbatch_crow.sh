#!/bin/bash

#################################################
## UNIFIED SLURM BATCH SCRIPT FOR LORA TRAINING ##
## Supports both Simple LoRA and CROW training using crow_trainer.py ##
#################################################

#SBATCH --nodes=1                   # Use 1 node
#SBATCH --cpus-per-task=8           # 8 CPUs for processing
#SBATCH --mem=64GB                  # 64GB memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --constraint=a100           # Target A100 GPUs
#SBATCH --time=01-00:00:00          # Maximum run time of 1 day
#SBATCH --output=%u.lora_train      # Log file location
#SBATCH --partition=researchshort   # Partition assigned
#SBATCH --account=sunjunresearch    # Account assigned
#SBATCH --qos=research-1-qos        # QOS assigned
#SBATCH --job-name=lora_train       # Job name

#################################################
##            SCRIPT CONFIGURATION             ##
#################################################

# Function for error handling
handle_error() {
  echo "ERROR: An error occurred at line $1"
  echo "Script exited with status $2"
  exit $2
}

# Enable error handling
trap 'handle_error ${LINENO} $?' ERR

# Usage function
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_NAME        Model to use (llama2-7b, llama3-8b, mistral-7b)"
    echo "  -d, --dataset DATASET_PATH     Dataset path for fine-tuning"
    echo "  -o, --output OUTPUT_DIR        Output directory for trained model"
    echo "  --mode MODE                    Training mode: sft or crow (default: sft)"
    echo "  --epochs EPOCHS                Number of training epochs (default: 3)"
    echo "  --lr LEARNING_RATE             Learning rate (default: 2e-4)"
    echo "  --batch-size BATCH_SIZE        Per device batch size (default: 2)"
    echo "  --lora-rank RANK               LoRA rank (default: 16)"
    echo "  --lora-alpha ALPHA             LoRA alpha (default: 32.0)"
    echo "  --consistency-alpha ALPHA      Consistency loss weight for CROW (default: 1.0)"
    echo "  --epsilon EPSILON              Adversarial perturbation magnitude for CROW (default: 0.1)"
    echo "  --list-available               List available models"
    echo "  -h, --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m llama2-7b --mode sft"
    echo "  $0 -m llama2-7b --mode crow --consistency-alpha 1.5"
    echo "  $0 -m mistral-7b --mode sft --epochs 5 --lr 1e-4"
    echo "  $0 --list-available"
}

# Function to list available models
list_available() {
    echo "=== Available Models ==="
    echo "llama2-7b      -> meta-llama/Llama-2-7b-chat-hf"
    echo "llama3-8b      -> meta-llama/Llama-3.1-8B-Instruct"  
    echo "mistral-7b     -> mistralai/Mistral-7B-Instruct-v0.3"
    echo ""
    echo "=== Available Training Modes ==="
    echo "sft    -> Standard LoRA fine-tuning without consistency regularization"
    echo "crow           -> CROW consistency training with layer regularization"
}

# Default values
MODEL_NAME="llama2-7b"
CUSTOM_DATASET_PATH=""
CUSTOM_OUTPUT_DIR=""
TRAINING_MODE="sft"
EPOCHS="3"
LEARNING_RATE="2e-4"
BATCH_SIZE="2"
LORA_RANK="8"
LORA_ALPHA="16.0"
CONSISTENCY_ALPHA="1.0"
EPSILON="0.1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -d|--dataset)
            CUSTOM_DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output)
            CUSTOM_OUTPUT_DIR="$2"
            shift 2
            ;;
        --mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --consistency-alpha)
            CONSISTENCY_ALPHA="$2"
            shift 2
            ;;
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        --list-available)
            list_available
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate training mode
if [[ "$TRAINING_MODE" != "sft" && "$TRAINING_MODE" != "crow" ]]; then
    echo "Error: Invalid training mode '$TRAINING_MODE'. Must be 'sft' or 'crow'"
    show_usage
    exit 1
fi

# Check if running in SLURM or directly
if [ -n "$SLURM_JOB_ID" ]; then
    echo "===== SLURM Job Information ====="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node: $SLURMD_NODENAME"
    echo "Allocated GPUs: $SLURM_JOB_GPUS"
    echo "Start time: $(date)"
    echo "================================="
    
    # Load modules for SLURM environment
    echo "Loading necessary modules..."
    module purge
    module load Python/3.10.16-GCCcore-13.3.0 || echo "Warning: Python module failed to load"
    module load CUDA/12.6.0 || echo "Warning: CUDA module failed to load"
    
    # Activate virtual environment
    source ~/myenv/bin/activate || echo "Warning: Failed to activate virtual environment"
    
    # Verify environment
    echo "Python version: $(python --version)"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || echo "Warning: PyTorch check failed"
else
    echo "===== Direct Execution ====="
    echo "Not running in SLURM environment"
    echo "Start time: $(date)"
    echo "============================"
fi

#################################################
##            MODEL CONFIGURATION              ##
#################################################

# Function to get model path from model name
get_model_path() {
    case $1 in
        "llama2-7b")
            echo "meta-llama/Llama-2-7b-chat-hf"
            ;;
        "llama3-8b")
            echo "meta-llama/Llama-3.1-8B-Instruct"
            ;;
        "mistral-7b")
            echo "mistralai/Mistral-7B-Instruct-v0.3"
            ;;
        *)
            echo "Unknown model: $1"
            return 1
            ;;
    esac
}

# Function to get model directory name
get_model_dir() {
    local suffix=""
    if [ "$TRAINING_MODE" = "crow" ]; then
        suffix="-CROW"
    else
        suffix="-LoRA-simple"
    fi
    
    case $1 in
        "llama2-7b")
            echo "Llama-2-7b${suffix}"
            ;;
        "llama3-8b")
            echo "Llama-3.1-8B${suffix}"
            ;;
        "mistral-7b")
            echo "Mistral-7B${suffix}"
            ;;
        *)
            echo "Unknown model: $1"
            return 1
            ;;
    esac
}

#################################################
##            CONFIGURATION LOGIC              ##
#################################################

# Validate required parameters
if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model name is required"
    show_usage
    exit 1
fi

# Get model path
MODEL_PATH=$(get_model_path "$MODEL_NAME")
if [ $? -ne 0 ]; then
    echo "Error: $MODEL_PATH"
    list_available
    exit 1
fi

# Get model directory name
MODEL_DIR=$(get_model_dir "$MODEL_NAME")
if [ $? -ne 0 ]; then
    echo "Error: $MODEL_DIR"
    exit 1
fi

# Set default paths if not provided
if [ -z "$CUSTOM_DATASET_PATH" ]; then
    DATASET_PATH="data/datatrain/distilled_alpaca_all.json"
else
    DATASET_PATH="$CUSTOM_DATASET_PATH"
fi

if [ -z "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="lora_weight/$MODEL_DIR"
else
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
fi

# Validate dataset path
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set consistency regularization flag
if [ "$TRAINING_MODE" = "crow" ]; then
    ENABLE_CONSISTENCY="True"
else
    ENABLE_CONSISTENCY="False"
fi

#################################################
##              TRAINING EXECUTION             ##
#################################################

echo "===== LoRA Training Configuration ====="
echo "Training Mode: $TRAINING_MODE"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
if [ "$TRAINING_MODE" = "crow" ]; then
    echo "Consistency Alpha: $CONSISTENCY_ALPHA"
    echo "Adversarial Epsilon: $EPSILON"
fi
echo "=============================================="

# Build the command
CMD="python crow_trainer.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir True \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataloader_pin_memory False \
    --load_best_model_at_end False \
    --ddp_find_unused_parameters False \
    --group_by_length True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --enable_consistency_regularization $ENABLE_CONSISTENCY \
    --print_loss True"

# Add CROW-specific parameters if needed
if [ "$TRAINING_MODE" = "crow" ]; then
    CMD="$CMD --alpha $CONSISTENCY_ALPHA --epsilon $EPSILON"
fi

# Run the training
echo "Starting $TRAINING_MODE training..."
echo "Command: $CMD"
echo ""
eval $CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "===== Training Completed Successfully ====="
    echo "Model saved to: $OUTPUT_DIR"
    echo "End time: $(date)"
    
    # List the contents of the output directory
    echo "Output directory contents:"
    ls -la "$OUTPUT_DIR"
else
    echo "===== Training Failed ====="
    echo "Check the logs above for error details"
    exit 1
fi

echo "===== $TRAINING_MODE Training Complete =====" 