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

chosen_user = None

def load_fine_tuned_model(model_path, base_model_name):
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
    speaker_tokens = ['[Federico]:', '[Paolo]:', '[Riccardo Santini]:', '[Guglielmone]:', '<|system|>', '</|system|>']
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


def parse_conversation_input(user_input):
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


def format_conversation(messages):
    # Add the instruction at the start
    conversation = f"<|system|>\nYou are {chosen_user} in a WhatsApp group chat with his friends that switch between English and Italian. Respond naturally in {chosen_user}'s style to the conversation below. The message should take the context into account so that it is coherent and flows naturally with the conversation. When {chosen_user} is mentioned in any of the messages in the conversation, you should pay more attention to that message when replying.</|system|>\n\n"
    
    # Then add the messages
    for message in messages:
        conversation += f"[{message['speaker']}]: {message['message']}\n"
    
    # Add the response marker
    conversation += f"[RESPONSE]:"
    
    return conversation.strip()


def interactive_mode(model, tokenizer):
    print("\n===== Interactive Mode =====")
    print("Type 'exit' to quit")
    print("Format your input as: 'Guglielmone: Message1 | Paolo: Message2 | ...'")
    
    while True:
        user_input = input("\nEnter conversation: ")
        if user_input.lower() == 'exit':
            break
        
        messages = parse_conversation_input(user_input)
        conversation = format_conversation(messages)
        
        print("\nFormatted prompt:")
        print(conversation)
        print("\nGenerating response...")
        
        completion = generate_response(
            model, 
            tokenizer, 
            conversation
        )
        
        print(f"\n[{chosen_user}]: {completion}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a fine-tuned model")
    parser.add_argument('--user', type=str, default=None,
                        help='User name to be used in the prompt.')
    parser.add_argument("--model_path", type=str, default="model/fine-tuned/", 
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

    chosen_user = args.user if args.user else "Paolo"
    
    # Load the model
    model, tokenizer = load_fine_tuned_model(args.model_path, args.base_model)
    
    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")

        messages = parse_conversation_input(args.prompt)
        formatted_prompt = format_conversation(messages)

        print(f"\nFormatted prompt: {formatted_prompt}")
        
        completion = generate_response(
            model, 
            tokenizer, 
            formatted_prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print(f"[{chosen_user}]: {completion}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)