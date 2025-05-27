# Hallucination Mitigation via CROW Training

A research codebase implementing **CROW (Consistency RegularizatiOn)** training for reducing hallucinations in large language models through layer consistency regularization and adversarial training.

## 🎯 Overview

This project addresses the critical problem of hallucinations in large language models by introducing a novel consistency-based training methodology. CROW training enforces consistency between adjacent transformer layers using adversarial perturbations, leading to more reliable and factually accurate model outputs.

### Key Features

- **🔬 CROW Training**: Novel consistency regularization with adversarial training
- **📊 Multi-Dataset Evaluation**: Comprehensive assessment on CoQA, TriviaQA, SQuAD, and Natural Questions
- **🤖 Automated Hallucination Detection**: GPT-4o-mini based evaluation with detailed categorization
- **⚡ Parameter-Efficient**: LoRA integration for efficient fine-tuning
- **🔧 Production Ready**: SLURM integration for cluster computing
- **📈 Layer Analysis**: Cosine similarity analysis between transformer layers

## 🏗️ Methodology

### CROW Training Algorithm

1. **Forward Pass**: Extract hidden states from all transformer layers
2. **Consistency Loss**: Compute cosine similarity between adjacent layers:
   ```python
   consistency_loss = (1 - F.cosine_similarity(h_states, next_h_states, dim=-1)).mean()
   ```
3. **Adversarial Perturbation**: Apply FGSM to input embeddings
4. **Perturbed Consistency**: Recompute consistency loss with perturbed inputs
5. **Combined Loss**: `L_total = L_LM + α * L_consistency_perturbed`

### Hallucination Detection

- **INTRINSIC**: Contradicts provided ground truth
- **EXTRINSIC**: Adds external information not in ground truth
- **FACTUAL_ERROR**: Contains incorrect facts
- **NONE**: No hallucinations detected

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 12.6+ (for GPU training)
- 64GB+ RAM recommended
- A100 GPU recommended for training

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd hallucination-mitigation
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Linux/Mac
   # or
   myenv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers datasets peft accelerate
   pip install pandas numpy scikit-learn tqdm
   pip install requests openai  # For GPT evaluation
   ```

4. **Set up data directories:**
   ```bash
   mkdir -p data/datasets data/datatrain
   mkdir -p lora_weight base_results
   ```

## 📚 Data Preparation

### Training Data

Place your instruction fine-tuning dataset in `data/datatrain/`. The expected format is JSON with the following structure:

```json
[
  {
    "instruction": "Question or task description",
    "input": "Optional context or input (can be empty)",
    "output": "Expected response or answer"
  }
]
```

### Evaluation Datasets

Download the following datasets to `data/datasets/`:

- **CoQA**: `coqa-dev-v1.0.json` from [Stanford CoQA](https://stanfordnlp.github.io/coqa/)
- **SQuAD 2.0**: `dev-v2.0.json` from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- **TriviaQA** and **Natural Questions**: Automatically downloaded via HuggingFace datasets

## 🚀 Usage

### Training

#### CROW Training (Recommended)

```bash
./sbatch_crow.sh -m llama2-7b --mode crow --epochs 3 --consistency-alpha 1.5
```

#### Standard LoRA Fine-tuning

```bash
./sbatch_crow.sh -m llama2-7b --mode sft --epochs 3 --lr 2e-4
```

#### Available Models

- `llama2-7b` → meta-llama/Llama-2-7b-chat-hf
- `llama3-8b` → meta-llama/Llama-3.1-8B-Instruct
- `mistral-7b` → mistralai/Mistral-7B-Instruct-v0.3

#### Training Parameters

```bash
./sbatch_crow.sh \
  --model llama2-7b \
  --mode crow \
  --epochs 3 \
  --lr 2e-4 \
  --batch-size 2 \
  --lora-rank 8 \
  --lora-alpha 16.0 \
  --consistency-alpha 1.5 \
  --epsilon 0.1 \
  --dataset path/to/custom/dataset.json \
  --output path/to/output/directory
```

### Evaluation

#### Basic Evaluation

```bash
python evaluate.py \
  --dataset coqa triviaqa SQuAD nq_open \
  --model Llama-2-7b-chat-hf \
  --lora_weights ./lora_weight/Llama-2-7b-CROW \
  --batch_size 4
```

#### With GPT-4o-mini Hallucination Analysis

```bash
export OPENAI_API_KEY="your-api-key"

python evaluate.py \
  --dataset coqa \
  --model Llama-2-7b-chat-hf \
  --lora_weights ./lora_weight/Llama-2-7b-CROW \
  --evaluate_with_gpt \
  --log_cosine_similarity \
  --batch_size 4
```

#### Parallel Dataset Evaluation

```bash
python evaluate.py \
  --dataset coqa triviaqa SQuAD \
  --model Llama-2-7b-chat-hf \
  --lora_weights ./lora_weight/Llama-2-7b-CROW \
  --parallel_datasets \
  --num_processes 4
```

### Result Analysis

Filter and analyze results:

```bash
python filter_results.py \
  input_results.txt \
  filtered_results.txt
```

## 📁 Project Structure

```
hallucination-mitigation/
├── 📄 crow_trainer.py           # Main CROW training implementation
├── 📄 evaluate.py               # Evaluation framework
├── 📄 gpt_evaluation.py         # GPT-based hallucination assessment
├── 📄 model_utils.py            # Model loading and configuration utilities
├── 📄 filter_results.py         # Result filtering and analysis
├── 📄 _settings.py              # Configuration and path management
├── 📁 dataeval/                 # Dataset-specific evaluation modules
│   ├── 📄 coqa.py
│   ├── 📄 triviaqa.py
│   ├── 📄 SQuAD.py
│   └── 📄 nq_open.py
├── 📁 models/                   # Model loading utilities
├── 📁 utils/                    # Utility functions
│   ├── 📄 __init__.py          # Seed management, logging
│   └── 📄 parallel.py          # Parallel processing utilities
├── 📁 data/
│   ├── 📁 datasets/            # Evaluation datasets
│   └── 📁 datatrain/           # Training datasets
├── 📁 lora_weight/             # Trained LoRA weights
├── 📁 base_results/            # Evaluation results
├── 📄 sbatch_crow.sh           # SLURM training script
├── 📄 sbatch_evaluate_*.sh     # SLURM evaluation scripts
└── 📄 README.md
```

## ⚙️ Configuration

### CROW Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 0.1 | Adversarial perturbation magnitude |
| `alpha` | 5.5 | Consistency loss weight |
| `lora_rank` | 16 | LoRA adaptation rank |
| `lora_alpha` | 32.0 | LoRA scaling factor |
| `enable_consistency_regularization` | True | Enable CROW training |

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Evaluation batch size |
| `fraction_of_data_to_use` | 1.0 | Fraction of dataset to evaluate |
| `log_cosine_similarity` | False | Log layer-wise similarities |
| `parallel_datasets` | False | Process datasets in parallel |

## 📊 Expected Results

### Hallucination Metrics

- **Hallucination Severity**: 1-10 scale (lower is better)
- **Factual Accuracy**: 1-10 scale (higher is better)
- **Overconfidence**: 1-10 scale (lower is better)
- **Overall Reliability**: 1-10 scale (higher is better)

### Layer Consistency

- **Cosine Similarity**: Between adjacent transformer layers
- **Consistency Correlation**: Relationship between consistency and hallucinations

## 🔬 Research Context

This work contributes to the field of AI safety and reliability by:

1. **Novel Training Methodology**: CROW represents a new approach to hallucination mitigation
2. **Comprehensive Evaluation**: Multi-dataset, multi-metric assessment framework
3. **Practical Implementation**: Production-ready code for reproducible research
4. **Open Science**: Transparent methodology and reproducible experiments

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Tokenizer Warnings**: Set `TOKENIZERS_PARALLELISM=false`
3. **SLURM Issues**: Check module loading and environment activation
4. **Dataset Not Found**: Ensure datasets are in correct `data/datasets/` location

### Debug Mode

Enable detailed logging:

```bash
python evaluate.py --log_cosine_similarity --print_loss True
```

## 📄 Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{crow-hallucination-mitigation,
  title={Hallucination Mitigation for Large Language Models},
  author={[Nay Myat Min]},
  year={2025},
  url={[https://github.com/NayMyatMin/hallucination-mitigation]}
}
```

