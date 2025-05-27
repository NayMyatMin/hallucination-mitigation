import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import models

# Mapping from simplified model names to full HuggingFace paths
MODEL_PATH_MAPPING = {
    'Llama-2-7b-chat-hf': 'meta-llama/Llama-2-7b-chat-hf',
    'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3'
}


def get_dataset_fn(data_name):
    """Return the appropriate dataset loading function"""
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset
    if data_name == 'SQuAD':
        return SQuAD.get_dataset
    raise ValueError(f"Unsupported dataset: {data_name}")


def get_generation_config(input_ids, tokenizer, data_name):
    """Configure generation parameters based on dataset type"""
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    elif data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    elif data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    elif data_name == 'SQuAD':
        generation_config = SQuAD._generate_config(tokenizer)
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    
    # Add common settings
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = False
    
    # Make sure pad_token_id is set properly - this can cause blank outputs if not configured correctly
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        generation_config['pad_token_id'] = tokenizer.pad_token_id
    else:
        # If pad_token_id is not set in tokenizer, use eos_token_id as fallback
        generation_config['pad_token_id'] = tokenizer.eos_token_id
    
    # Check for other potential issues
    if 'do_sample' not in generation_config:
        generation_config['do_sample'] = False
    
    # Make sure we're not skipping special tokens that might be needed for proper generation
    if 'use_cache' not in generation_config:
        generation_config['use_cache'] = True
        
    return generation_config


def load_model_from_hub(model_name, device):
    """Load model and tokenizer directly from HuggingFace Hub"""
    try:
        print(f"Attempting to load {model_name} directly from HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        print(f"Successfully loaded {model_name} from HuggingFace Hub")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading from HuggingFace Hub: {e}")
        return None, None


def load_model(model_name, device, use_hf_directly=True):
    """Load model and tokenizer with fallback options"""
    short_model_name = model_name
    full_model_name = MODEL_PATH_MAPPING.get(model_name, model_name)
    
    if use_hf_directly:
        # Try loading directly from HuggingFace first
        model, tokenizer = load_model_from_hub(full_model_name, device)
        if model is not None:
            return model, tokenizer
        
        print(f"Falling back to local model loading with name: {short_model_name}")
    
    # Use the regular loading method from models module with the local name
    print(f"Loading model from local path using name: {short_model_name}")
    return models.load_model_and_tokenizer(short_model_name, device)


def extract_ground_truth(batch, dataset_name):
    """Extract ground truth answer from batch based on dataset format"""
    if dataset_name == 'SQuAD' and 'answers' in batch:
        # SQuAD dataset has answers in a different format
        try:
            # Check if the answers field is empty first
            if len(batch['answers']) > 0:
                answers = batch['answers'][0]  # Get the first element (list of answer dictionaries)
                
                # Print debug information (only for first few examples)
                if 'id' in batch and batch['id'][0].startswith('5726'):  # Just sample a few for debugging
                    print(f"SQuAD answers format: {type(answers)}")
                    print(f"SQuAD answers content: {answers}")
                
                # Handle the SQuAD format where answers is a list of dictionaries
                if isinstance(answers, list) and len(answers) > 0:
                    # Each answer is a dict with 'text' and 'answer_start'
                    if isinstance(answers[0], dict) and 'text' in answers[0]:
                        return answers[0]['text']  # Return the text of the first answer
                
                # Old logic for backwards compatibility
                elif isinstance(answers, dict) and 'text' in answers:
                    if isinstance(answers['text'], (list, tuple)) and len(answers['text']) > 0:
                        return answers['text'][0]  # Take the first answer text
                    elif isinstance(answers['text'], str):
                        return answers['text']  # Direct string
                
                # If no valid format was found
                if 'id' in batch and batch['id'][0].startswith('5726'):  # Just sample a few for debugging
                    print("Warning: SQuAD answers format is unexpected, couldn't extract ground truth")
            else:
                print("Warning: SQuAD example has empty answers field")
        except Exception as e:
            print(f"Error extracting SQuAD answer: {e}")
    elif 'answer' in batch:
        return batch['answer'][0]
    
    return None


def setup_results_directory(model_name):
    """Set up the results directory structure for the model"""
    # Check for environment variable override first
    env_results_dir = os.environ.get('RESULTS_DIR')
    if env_results_dir:
        print(f"Using results directory from environment: {env_results_dir}")
        results_dir = env_results_dir
        os.makedirs(results_dir, exist_ok=True)
        model_results_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        return model_results_dir
        
    # Default behavior if no environment variable is set
    # Initialize results directory
    results_dir = "sft_results"
    os.makedirs(results_dir, exist_ok=True)

    # Check for training type in model_name and create appropriate folder
    model_folder_name = model_name
    
    # Update results directory name based on model type
    if "-LoRA-consistency" in model_name:
        results_dir = "consistency_results"
    elif "-LoRA-finetuning" in model_name:
        results_dir = "finetuning_results"
    
    # Create the specific model results directory
    os.makedirs(results_dir, exist_ok=True)
    model_results_dir = os.path.join(results_dir, model_folder_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    return model_results_dir


def get_output_path(model_results_dir, dataset_name, custom_output_file=None):
    """Get the output file path for the results"""
    if custom_output_file:
        # User provided custom output file base name - use it inside model_results_dir
        base_filename = f"{custom_output_file}_{dataset_name}"
        # Ensure the directory exists
        os.makedirs(model_results_dir, exist_ok=True)
        return os.path.join(model_results_dir, base_filename)
    else:
        # Extract training type for the file name
        training_type = "base"
        # Infer training type more robustly from the directory path components
        path_parts = model_results_dir.lower().split(os.sep)
        if "consistency_results" in path_parts or "consistency" in path_parts:
             training_type = "consistency"
        elif "finetuning_results" in path_parts or "finetuning" in path_parts:
             training_type = "finetuning"
        elif "lora" in path_parts: # Check for "lora" if specific type not found
             training_type = "lora"
            
        # Use standard format with training type included
        filename = f"result_{dataset_name}_{training_type}.txt"
        # Ensure the directory exists
        os.makedirs(model_results_dir, exist_ok=True)
        return os.path.join(model_results_dir, filename) 