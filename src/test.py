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

def load_fine_tuned_model(model_path, base_model_name, adapter_scale):
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
    
    # Add the same special tokens that were used during training
    special_tokens = ['[Federico]:', '[Paolo]:', '[Riccardo Santini]:', '[Guglielmone]:']
    pad_token = '<|pad|>'
    tokenizer.add_special_tokens({'pad_token': pad_token, 'additional_special_tokens': special_tokens})
    
    # Set padding side to match training
    tokenizer.padding_side = 'right'

    # Resize model embeddings to match tokenizer size
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    
    # Load the fine-tuned LoRA adapter
    print(f"Loading adapter from: {model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map="auto",
    )

    model.set_adapter_scale('default', adapter_scale)
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    # Create input tokens with attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,  # Match max_seq_length from training
        return_attention_mask=True
    ).to(model.device)
    
    # Generate with the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
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


def format_conversation(messages, chosen_user):
    # Format using the chat template to match training
    system_message = {
        'role': 'system', 
        'content': f'You are an assistant that mimics {chosen_user} in a WhatsApp group chat with his friends that switch between English and Italian. Respond naturally in {chosen_user}\'s style to the conversation below. The message should take the context into account so that it is coherent and flows naturally with the conversation. When {chosen_user} is mentioned in any of the messages in the conversation, you should pay more attention to that message when replying.'
    }
    
    user_messages = [
        {'role': 'user', 'content': f"[{message['speaker']}]: {message['message']}"} 
        for message in messages
    ]
    
    # Apply the chat template without the assistant's response
    formatted_conversation = tokenizer.apply_chat_template(
        [system_message] + user_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_conversation


def interactive_mode(model, tokenizer, chosen_user):
    print("\n===== Interactive Mode =====")
    print("Type 'exit' to quit")
    print("Format your input as: 'Guglielmone: Message1 | Paolo: Message2 | ...'")
    
    while True:
        user_input = input("\nEnter conversation: ")
        if user_input.lower() == 'exit':
            break
        
        messages = parse_conversation_input(user_input)
        conversation = format_conversation(messages, chosen_user)
        
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
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                        help="Name of the base model")
    parser.add_argument("--adapter_scale", type=float, default=1.0,
                        help="Scale for the LoRA adapter")
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
    model, tokenizer = load_fine_tuned_model(args.model_path, args.base_model, args.adapter_scale)
    
    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")

        messages = parse_conversation_input(args.prompt)
        formatted_prompt = format_conversation(messages, chosen_user)

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
        interactive_mode(model, tokenizer, chosen_user)