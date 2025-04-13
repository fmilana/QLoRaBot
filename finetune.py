import os
import argparse
import warnings
import json
import torch
import gc
import transformers
import bitsandbytes as bnb
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback


chosen_user = None


def load_raw_dataset(file_path, min_words):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    examples = [
        conversation for conversation in raw_data 
        if len(conversation['target_response'].split()) >= min_words
    ]

    filtered_dataset = Dataset.from_list(examples)

    print(f"Original dataset size: {len(raw_data)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    return filtered_dataset


def format_prompts_func(example_or_batch):
    def format_single(example):
        conversation = f"<|system|>\nYou are {chosen_user} in a WhatsApp group chat with his friends that switch between English and Italian. Respond naturally in {chosen_user}'s style to the conversation below. The message should take the context into account so that it is coherent and flows naturally with the conversation. When {chosen_user} is mentioned in any of the messages in the conversation, you should pay more attention to that message when replying.</|system|>\n\n"

        for entry in example["context"]:
            conversation += f"[{entry['speaker']}]: {entry['message']}\n"

        conversation += f"[RESPONSE]: {example['target_response']}"

        return conversation

    if isinstance(example_or_batch["context"][0], dict):
        return [format_single(example_or_batch)]
    else:
        return [format_single({"context": ctx, "target_response": resp}) 
                for ctx, resp in zip(example_or_batch["context"], example_or_batch["target_response"])]


def find_all_linear_names(model, bits):
    cls = (
        bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    return list(lora_module_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a language model with QLoRA')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B',
                        help='Model name to load (default: "meta-llama/Meta-Llama-3-8B")')
    parser.add_argument('--user', type=str, default=None,
                        help='User name to load specific train/val files (e.g., "Paolo")')
    parser.add_argument('--min-msg-length', type=int, default=0,
                        help='Minimum number of words in messages (default: 0)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience. Number of evaluation calls with no improvement after which training will stop (default: 3)')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping based on validation loss')

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs("model/training", exist_ok=True)
    os.makedirs("model/fine-tuned", exist_ok=True)

    torch.cuda.empty_cache()
    gc.collect()

    chosen_user = args.user

    warnings.filterwarnings('ignore')

    rank = 16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = ['[Federico]:', '[Paolo]:', '[Riccardo Santini]:', '[Guglielmone]:', '<|system|>', '</|system|>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    target_modules = [
        'gate_proj',
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'down_proj',
        'up_proj',
        'lm_head'
    ]

    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, config)

    train_dataset_path = f'data/processed/train_conversations_{args.user.lower()}.json'
    val_dataset_path = f'data/processed/val_conversations_{args.user.lower()}.json'
        
    print(f'Loading training dataset from {train_dataset_path}')
    train_dataset = load_raw_dataset(train_dataset_path, args.min_msg_length)

    print(f'Loading validation dataset from {val_dataset_path}')
    val_dataset = load_raw_dataset(val_dataset_path, args.min_msg_length)

    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')

    response_template = "[RESPONSE]:"

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    
    training_args = transformers.TrainingArguments(
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        warmup_steps=4,
        learning_rate=2e-4,
        fp16=True,
        evaluation_strategy='steps',
        eval_steps=50,  # Evaluate every 50 steps
        save_safetensors=True,
        save_steps=50,  # Save checkpoints every 50 steps
        save_strategy='steps',
        save_total_limit=2,  # Keep only the 2 best models
        logging_steps=50,
        output_dir=f'model/training',
        optim='paged_adamw_8bit',
        run_name=f'finetune-{args.user}-{args.model_name.split("/")[-1]}',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = []
    if args.early_stopping:
        print(f"Early stopping enabled with patience {args.patience}")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=format_prompts_func,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    model.config.use_cache = False

    trainer.train()

    # Save the best model
    print("Saving the best model...")
    model.save_pretrained('model/fine-tuned')
    tokenizer.save_pretrained('model/fine-tuned')

    print('Model fine-tuning complete.')
    print('Model saved to model/fine-tuned')
    print('Tokenizer saved to model/fine-tuned')