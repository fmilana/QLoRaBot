import torch
import json
import os
import gc
import argparse  # Import argparse for command-line arguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset


# Set PyTorch memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set max sequence length
MAX_LENGTH = 2048


def format_prompts_func(example_or_batch):
    def format_single(example):
        conversation = ""
        for entry in example["context"]:
            conversation += f"[{entry['speaker']}]: {entry['message']}\n"
        conversation += f"[RESPONSE]: {example['target_response']}"
        return conversation

    # Detect batch
    if isinstance(example_or_batch["context"][0], dict):
        # Single example
        return [format_single(example_or_batch)]
    else:
        # Batch
        return [format_single({"context": ctx, "target_response": resp}) 
                for ctx, resp in zip(example_or_batch["context"], example_or_batch["target_response"])]


def filter_by_length(examples, tokenizer, max_length=MAX_LENGTH):
    """
    Filter examples that would exceed the maximum length when tokenized.
    Returns a boolean mask of which examples to keep.
    """
    formatted_examples = format_prompts_func(examples)
    tokenized_lengths = [len(tokenizer.encode(example)) for example in formatted_examples]
    return [length <= max_length for length in tokenized_lengths]


def load_raw_dataset(file_path, tokenizer, max_length=MAX_LENGTH):
    """
    Load raw dataset from JSON file and filter by tokenized length.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Create a temporary dataset to use with the filter function
    temp_dataset = Dataset.from_list(raw_data)
    
    # Apply length filtering
    print(f"Filtering examples longer than {max_length} tokens...")
    filtered_dataset = temp_dataset.filter(
        lambda examples: filter_by_length(examples, tokenizer, max_length),
        batched=True,
        batch_size=100  # Process in batches for efficiency
    )
    
    print(f"Original dataset size: {len(temp_dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    return filtered_dataset


def train_with_qlora(train_dataset, val_dataset, model_name, output_dir, epochs=2, batch_size=1):
    """
    Fine-tune a model using QLoRA with memory optimizations.
    """
    # Free up CUDA memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # Double quantization to save more memory
    )
    
    # Load the model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        offload_folder="offload",  # Enable offloading to CPU/disk
        offload_state_dict=True,  # Offload state dict when not in use
        low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add speaker tags as special tokens
    speaker_tokens = ['[Federico]:', '[Paolo]:', '[Riccardo Santini]:', '[Guglielmone]:']
    tokenizer.add_special_tokens({'additional_special_tokens': speaker_tokens})

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Determine target modules based on model architecture
    if "deepseek" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "llama" in model_name.lower() or "mistral" in model_name.lower() or "mixtral" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "bloom" in model_name.lower():
        target_modules = ["query_key_value"]
    elif "falcon" in model_name.lower():
        target_modules = ["query_key_value"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=1e-4,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=500,  # Save less frequently than evaluation
        eval_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        logging_dir="./logs",
        fp16=False,
        report_to="none",
        packing=False,
        max_seq_length=MAX_LENGTH  # Set maximum sequence length
    )
    
    # Set up the response template for completion-only training
    response_template = "[RESPONSE]:"

    # Create a data collator that masks loss for tokens before the response template
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    
    # Create SFTTrainer with completion-only data collator and formatting_func
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=format_prompts_func,  # Use the formatting function
        data_collator=data_collator,         # Use completion-only data collator
    )
    
    # Train the model
    trainer.train()
    
    # Calculate perplexity explicitly on validation set
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    
    print(f"Final validation loss: {eval_loss}")
    print(f"Final validation perplexity: {perplexity}")
    
    # Save the model
    trainer.save_model(output_dir)
    
    return model, tokenizer, perplexity


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fine-tune a language model with QLoRA')
    parser.add_argument('--model-name', type=str, default="meta-llama/Meta-Llama-3-8B", 
                        help='Base model to use (default: meta-llama/Meta-Llama-3-8B)')
    parser.add_argument('--output-name', type=str, required=True, 
                        help='Custom name for the output model directory')
    parser.add_argument('--user', type=str, default=None,
                        help='User name to load specific train/val files (e.g., "paolo_v1")')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training (default: 1)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Free up CUDA memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Define model name and output directory with custom name
    model_name = args.model_name
    output_dir = f"model/fine-tuned/{args.output_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer first for length filtering
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add speaker tags as special tokens
    speaker_tokens = ['[Federico]:', '[Paolo]:', '[Riccardo Santini]:', '[Guglielmone]:']
    tokenizer.add_special_tokens({'additional_special_tokens': speaker_tokens})
    
    # Determine file paths based on user parameter
    if args.user:
        train_dataset_path = f"data/processed/train_conversations_{args.user}.json"
        val_dataset_path = f"data/processed/val_conversations_{args.user}.json"
    else:
        train_dataset_path = "data/processed/train_conversations.json"
        val_dataset_path = "data/processed/val_conversations.json"
    
    print(f"Loading training dataset from {train_dataset_path}")
    train_dataset = load_raw_dataset(train_dataset_path, tokenizer, MAX_LENGTH)
    
    print(f"Loading validation dataset from {val_dataset_path}")
    val_dataset = load_raw_dataset(val_dataset_path, tokenizer, MAX_LENGTH)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Train the model with QLoRA and completion-only loss
    model, tokenizer, val_perplexity = train_with_qlora(
        train_dataset, 
        val_dataset, 
        model_name, 
        output_dir, 
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print(f"Model fine-tuning complete. Model saved to {output_dir}")
    print(f"Final validation perplexity: {val_perplexity}")
    print("To test the model, use test_model.py")