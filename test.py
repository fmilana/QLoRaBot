import torch
import gc
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel


def load_fine_tuned_model(model_path, base_model_name):
    """
    Load a fine-tuned model for inference with memory optimizations.
    
    Args:
        model_path: Path to the fine-tuned LoRA adapter
        base_model_name: Name of the base model
    
    Returns:
        model, tokenizer: The loaded model and tokenizer
    """
    # Free up CUDA memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Configure quantization for inference
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load the base model with quantization
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add the same special tokens that were used during training
    speaker_tokens = ['[Federico]:', '[Paolo]:', '[Riccardo Santini]:', '[Guglielmone]:']
    tokenizer.add_special_tokens({'additional_special_tokens': speaker_tokens})

    # Resize model embeddings to match tokenizer size
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load the fine-tuned LoRA adapter
    print(f"Loading adapter from: {model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map="auto",
    )
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    # Ensure the prompt ends with the speaker tag
    if not prompt.strip().endswith("[Paolo]:"):
        prompt = prompt.strip() + "\n[Paolo]:"
    
    # Create input tokens with attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # Set appropriate max context length
        return_attention_mask=True  # Explicitly request attention mask
    ).to(model.device)
    
    # Now pass both input_ids and attention_mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Pass the attention mask
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return generated_text.strip()


def format_conversation(messages):
    """
    Format a list of messages into the conversation format expected by the model.
    
    Args:
        messages: List of dictionaries with 'speaker' and 'message' keys
    
    Returns:
        Formatted conversation string
    """
    conversation = ""
    for message in messages:
        conversation += f"[{message['speaker']}]: {message['message']}\n"
    return conversation.strip()


def parse_conversation_input(user_input):
    """
    Parse user input in a simple format to create a conversation.
    Format: "Speaker1: Message1 | Speaker2: Message2 | ..."
    
    Returns:
        List of message dictionaries
    """
    messages = []
    parts = user_input.split("|")
    
    for part in parts:
        part = part.strip()
        if ":" in part:
            speaker, message = part.split(":", 1)
            messages.append({
                "speaker": speaker.strip(),
                "message": message.strip()
            })
    
    return messages


def interactive_mode(model, tokenizer):
    """
    Run an interactive session with the model.
    """
    print("\n===== Interactive Mode =====")
    print("Type 'exit' to quit")
    print("Format your input as: 'Guglielmone: Message1 | Paolo: Message2 | ...'")
    
    while True:
        user_input = input("\nEnter conversation: ")
        if user_input.lower() == 'exit':
            break
        
        # Check if user input is in the expected format
        if "|" in user_input:
            messages = parse_conversation_input(user_input)
            conversation = format_conversation(messages)
        else:
            # If simple format not detected, use as-is (for advanced users)
            conversation = user_input
        
        # Ensure it ends with the speaker tag
        if not conversation.strip().endswith("[Paolo]:"):
            conversation = conversation.strip() + "\n[Paolo]:"
        
        print("\nFormatted prompt:")
        print(conversation)
        print("\nGenerating response...")
        
        completion = generate_response(
            model, 
            tokenizer, 
            conversation
        )
        
        print(f"\nPaolo's response: {completion}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a fine-tuned model")
    parser.add_argument("--model_path", type=str, default="model/fine-tuned/qlora_model", 
                        help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B", 
                        help="Name of the base model")
    parser.add_argument("--prompt", type=str, 
                        help="Single prompt to test (if not provided, interactive mode will start)")
    parser.add_argument("--max_length", type=int, default=100, 
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, 
                        help="Repetition penalty")
    
    args = parser.parse_args()
    
    # Load the model
    model, tokenizer = load_fine_tuned_model(args.model_path, args.base_model)
    
    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")
        
        # Format the prompt if it doesn't already have the expected format
        if not args.prompt.strip().endswith("[Paolo]:"):
            if "|" in args.prompt:
                messages = parse_conversation_input(args.prompt)
                formatted_prompt = format_conversation(messages) + "\n[Paolo]:"
            else:
                formatted_prompt = args.prompt.strip() + "\n[Paolo]:"
            print(f"\nFormatted prompt: {formatted_prompt}")
        else:
            formatted_prompt = args.prompt
        
        completion = generate_response(
            model, 
            tokenizer, 
            formatted_prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print(f"\nPaolo's response: {completion}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)