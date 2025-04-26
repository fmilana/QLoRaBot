# from transformers import AutoTokenizer


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# chosen_user = "Paolo"
# example = {
#     "context": [
#         {"role": "user", "message": "[Federico]: Hey, how are you?"},
#         {"role": "user", "message": "[Guglielmone]: We should meet later!"},
#         {"role": "user", "message": "[Riccardo Santini]: We should meet later!"},
#         {"role": "user", "message": "[Guglielmone]: We should meet later!"}
#     ],
#     "target_response": "Sure, let me check my schedule."
# }

# messages = [
#     {'role': 'system', 'content': f'You are an assistant that mimics {chosen_user} in a WhatsApp group chat with his friends that switch between English and Italian. Respond naturally in {chosen_user}\'s style to the conversation below. The message should take the context into account so that it is coherent and flows naturally with the conversation. When {chosen_user} is mentioned in any of the messages in the conversation, you should pay more attention to that message when replying.'}
# ]

# for entry in example["context"]:
#     messages.append({'role': 'user', 'content': f"{entry['message']}"})

# messages.append({'role': 'assistant', 'content': example['target_response']})

# formatted = tokenizer.apply_chat_template(messages, tokenize=False)

# print(repr(formatted))



import argparse
from transformers import AutoTokenizer

def load_tokenizer(base_model_name):
    """
    Load only the tokenizer without the model.
    """
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Add the same special tokens that were used during training
    special_tokens = ['[Federico]:', '[Paolo]:', '[Riccardo Santini]:', '[Guglielmone]:']
    pad_token = '<|pad|>'
    tokenizer.add_special_tokens({'pad_token': pad_token, 'additional_special_tokens': special_tokens})
    
    # Set padding side to match training
    tokenizer.padding_side = 'right'

    return tokenizer

def parse_conversation_input(user_input):
    """
    Parse the input conversation format.
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

def format_conversation(messages, chosen_user, tokenizer):
    """
    Format the conversation using the chat template.
    """
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
        tokenize=False 
    )
    
    return formatted_conversation

def interactive_mode(tokenizer, chosen_user):
    """
    Run in interactive mode to test prompt formatting.
    """
    print("\n===== Prompt Format Tester =====")
    print("Type 'exit' to quit")
    print("Format your input as: 'Guglielmone: Message1 | Paolo: Message2 | ...'")
    
    while True:
        user_input = input("\nEnter conversation: ")
        if user_input.lower() == 'exit':
            break
        
        messages = parse_conversation_input(user_input)
        conversation = format_conversation(messages, chosen_user, tokenizer)
        
        print("\nFormatted prompt:")
        print("-" * 50)
        print(conversation)
        print("-" * 50)
        
        # Also show the structured messages for verification
        print("\nParsed messages:")
        for msg in messages:
            print(f"  Speaker: '{msg['speaker']}', Message: '{msg['message']}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test prompt formatting without running the model")
    parser.add_argument('--user', type=str, default="Paolo",
                        help='User name to be used in the prompt (default: Paolo)')
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                        help="Name of the base model for tokenizer")
    parser.add_argument("--prompt", type=str, 
                        help="Single prompt to test (if not provided, interactive mode will start)")
    
    args = parser.parse_args()

    chosen_user = args.user
    
    # Load only the tokenizer
    tokenizer = load_tokenizer(args.base_model)
    
    if args.prompt:
        # Single prompt mode
        print(f"\nInput: {args.prompt}")

        messages = parse_conversation_input(args.prompt)
        formatted_prompt = format_conversation(messages, chosen_user, tokenizer)

        print("\nFormatted prompt:")
        print("-" * 50)
        print(formatted_prompt)
        print("-" * 50)
        
        # Also show the structured messages for verification
        print("\nParsed messages:")
        for msg in messages:
            print(f"  Speaker: '{msg['speaker']}', Message: '{msg['message']}'")
    else:
        # Interactive mode
        interactive_mode(tokenizer, chosen_user)