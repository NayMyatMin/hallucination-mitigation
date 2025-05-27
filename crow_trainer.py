#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CROW: Consistency Regularization for Consistency Fine-tuning

This script provides a standalone implementation of the CROW training method,
which uses layer consistency regularization with adversarial training to improve
model consistency and reduce hallucinations through consistency fine-tuning.
"""

import os
import json
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
from transformers.utils import is_peft_available

if is_peft_available():
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training
    )

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# Constants
IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    token: Optional[str] = field(
        default=None, 
        metadata={"help": "The token to use as HTTP bearer authorization for remote files"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading a model from HF Hub"}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2 implementation"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    train_file: str = field(
        metadata={"help": "The input training data file (a json file)."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to fine-tuning methods.
    """
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank of LoRA matrices"}
    )
    lora_alpha: float = field(
        default=32.0,
        metadata={"help": "Scaling factor for LoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout for LoRA layers"}
    )

@dataclass
class CROWArguments:
    """
    Arguments for CROW (Consistency RegularizatiOn) training.
    """
    epsilon: float = field(
        default=0.1,
        metadata={"help": "Magnitude of adversarial perturbations"}
    )
    alpha: float = field(
        default=5.5,
        metadata={"help": "Weight for consistency loss"}
    )
    print_loss: bool = field(
        default=False,
        metadata={"help": "Whether to print loss values during training"}
    )
    enable_consistency_regularization: bool = field(
        default=True,
        metadata={"help": "Whether to enable CROW consistency regularization. If False, performs simple LoRA fine-tuning."}
    )

class CROWTrainer(Seq2SeqTrainer):
    """
    CROW: Consistency RegularizatiOn Trainer for consistency fine-tuning.
    
    This trainer can operate in two modes:
    1. CROW mode (enable_consistency_regularization=True): Implements layer consistency 
       regularization with adversarial training to improve model consistency and reduce hallucinations.
    2. Simple LoRA mode (enable_consistency_regularization=False): Performs standard 
       LoRA fine-tuning without consistency regularization.
    """
    
    def __init__(self, crow_args: CROWArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crow_args = crow_args
        
        # Only configure for consistency training if enabled
        if self.crow_args.enable_consistency_regularization:
            # Ensure model outputs hidden states
            if hasattr(self.model.config, 'output_hidden_states'):
                self.model.config.output_hidden_states = True
            
            # Verify model has enough layers for consistency loss
            # We need at least 3 layers since we use layers [1:-2] and [2:-1]
            if hasattr(self.model.config, 'num_hidden_layers'):
                if self.model.config.num_hidden_layers < 3:
                    raise ValueError(f"Model must have at least 3 layers for consistency loss, got {self.model.config.num_hidden_layers}")
            else:
                logger.warning("Could not verify number of model layers - proceeding anyway")
        else:
            logger.info("Consistency regularization disabled - using simple LoRA fine-tuning mode")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes loss based on the training mode:
        - If consistency regularization is enabled: standard LM loss + consistency loss with adversarial training
        - If disabled: standard language modeling loss only (simple LoRA fine-tuning)
        """
        # If consistency regularization is disabled, use standard loss computation
        if not self.crow_args.enable_consistency_regularization:
            return super().compute_loss(model, inputs, return_outputs)
        
        # CROW consistency regularization mode
        inputs = inputs.copy()

        # Unwrap model for proper accelerator handling (critical for compatibility)
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model

        # Get input embeddings and enable gradient computation (using unwrapped model)
        inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
        
        # Forward pass with original embeddings to get hidden states
        outputs = unwrapped_model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

        # Calculate consistency loss between adjacent layers
        if len(hidden_states) < 3:
            raise RuntimeError(f"Need at least 3 hidden states for consistency loss, got {len(hidden_states)}")
            
        h_states = torch.stack(hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
        next_h_states = torch.stack(hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

        cos_sims_vec = F.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-6)  # Shape: [num_layers, batch_size, seq_len]
        consistency_loss = (1 - cos_sims_vec).mean()

        # Zero gradients
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        # Backward pass for consistency_loss to get gradients
        consistency_loss.backward(retain_graph=True)

        # Extract gradients w.r.t. inputs_embeds
        gradients = inputs_embeds.grad.detach()

        # Zero gradients in model parameters to prevent updates from consistency_loss
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        # Generate adversarial perturbations using FGSM
        epsilon = self.crow_args.epsilon
        perturbation = epsilon * gradients.sign()
        perturbed_embeds = inputs_embeds + perturbation

        # Forward pass with perturbed inputs for consistency regularization
        perturbed_outputs = unwrapped_model(inputs_embeds=perturbed_embeds, output_hidden_states=True, use_cache=False)
        perturbed_hidden_states = perturbed_outputs.hidden_states

        # Compute perturbed consistency loss using the same vectorized method
        perturbed_h_states = torch.stack(perturbed_hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
        perturbed_next_h_states = torch.stack(perturbed_hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

        perturbed_cos_sims_vec = F.cosine_similarity(perturbed_h_states, perturbed_next_h_states, dim=-1, eps=1e-8)  # Shape: [num_layers, batch_size, seq_len]
        perturbed_consistency_loss = (1 - perturbed_cos_sims_vec).mean()
        
        if self.crow_args.print_loss:
            logger.info(f"Perturbed Consistency Loss: {perturbed_consistency_loss.item():.6f}")

        # Standard language modeling loss
        standard_outputs = model(**inputs)
        standard_loss = standard_outputs["loss"] if isinstance(standard_outputs, dict) else standard_outputs[0]

        # Combined Loss
        alpha = self.crow_args.alpha
        total_loss = standard_loss + alpha * perturbed_consistency_loss

        if self.crow_args.print_loss:
            logger.info(f"Standard Loss: {standard_loss.item():.6f}")
            logger.info(f"Total Loss: {total_loss.item():.6f}")

        return (total_loss, standard_outputs) if return_outputs else total_loss


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    """
    
    def __init__(
        self, 
        data_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
    ):
        super(InstructionDataset, self).__init__()
        
        logger.info(f"Loading dataset from {data_path}")
        
        # Validate dataset file exists
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset file not found: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dataset file {data_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading dataset from {data_path}: {e}")
        
        # Validate dataset format
        if not isinstance(self.dataset, list) or len(self.dataset) == 0:
            raise ValueError(f"Dataset must be a non-empty list, got {type(self.dataset)}")
        
        # Check first example has required fields
        required_fields = ["instruction", "output"]
        example = self.dataset[0]
        missing_fields = [field for field in required_fields if field not in example]
        if missing_fields:
            raise ValueError(f"Dataset examples missing required fields: {missing_fields}")
            
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        logger.info(f"Loaded {len(self.dataset)} examples from {data_path}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        instruction = example["instruction"]
        input_text = example.get("input", "")
        response = example["output"]
        
        # Alpaca template format
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize the full conversation as one sequence to avoid duplicate special tokens
        full_text = prompt + response
        tokenized_full = self.tokenizer(full_text, truncation=False, return_tensors=None, add_special_tokens=True)
        
        # Get prompt length to create labels (mask prompt tokens)
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors=None, add_special_tokens=False)
        prompt_len = len(tokenized_prompt["input_ids"])
        
        # Build the input_ids and labels for causal LM training
        input_ids = tokenized_full["input_ids"]
        labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
        
        # Truncate sequences to max_seq_length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            
        # Create attention_mask
        attention_mask = [1] * len(input_ids)
        
        # Return as lists (not tensors) for more efficient collation
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class OptimizedDataCollator:
    """
    Optimized data collator that efficiently handles padding and tensor conversion.
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of=8, label_pad_token_id=IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, features):
        # Find the maximum length in the batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Pad to multiple of pad_to_multiple_of for efficient computation
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        batch_size = len(features)
        
        # Efficiently create batch tensors using torch.full for better performance
        batch_input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        batch_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        batch_labels = torch.full((batch_size, max_length), self.label_pad_token_id, dtype=torch.long)
        
        # Fill in the actual values
        for i, feature in enumerate(features):
            seq_len = len(feature["input_ids"])
            batch_input_ids[i, :seq_len] = torch.tensor(feature["input_ids"], dtype=torch.long)
            batch_attention_mask[i, :seq_len] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            batch_labels[i, :seq_len] = torch.tensor(feature["labels"], dtype=torch.long)
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, CROWArguments))
    model_args, data_args, training_args, finetuning_args, crow_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
        use_fast=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        device_map="auto",
    )
    
    # Configure model for training
    model.config.use_cache = False
    model.config.output_hidden_states = True  # Essential for CROW training
    model.gradient_checkpointing_enable()
    
    # Apply LoRA for parameter-efficient fine-tuning
    if is_peft_available():
        # Create new LoRA adapter for consistency training
        logger.info("Creating new LoRA adapter for CROW consistency training")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=finetuning_args.lora_rank,
            lora_alpha=finetuning_args.lora_alpha,
            lora_dropout=finetuning_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        
        model.print_trainable_parameters()
    else:
        logger.warning("PEFT not available - training full model parameters")
    
    # Create dataset
    train_dataset = InstructionDataset(
        data_path=data_args.train_file,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )
    
    # Create data collator
    data_collator = OptimizedDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )
    
    # Create CROW trainer
    trainer = CROWTrainer(
        crow_args=crow_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting CROW consistency training...")
    train_result = trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("CROW consistency training completed successfully!")


if __name__ == "__main__":
    main() 