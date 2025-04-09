import os
import re
import json
import random
import pandas as pd
import argparse  # Import argparse for command-line arguments
from datasets import Dataset


def parse_whatsapp_chat(file_path):
    '''
    Parses a WhatsApp chat file and returns a dictionary of messages.
    '''
    regex = r'(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - ([^:]+): (.*)'
    event_regex = r'(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.+)'  # For system messages/events
    filtered = ['<Media omitted>', 'This message was deleted', 'You deleted this message', 'POLL:']
    messages = []

    last_timestamp = None
    last_sender = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(regex, line)
            
            if match:
                timestamp, sender, message = match.groups()

                to_skip = False

                for filtered_message in filtered:
                    if filtered_message in message:
                        to_skip = True
                        break
                if to_skip:
                    continue

                message = message.replace('<This message was edited>', '')

                messages.append({
                    'timestamp': timestamp,
                    'sender': sender,
                    'message': message.strip()
                })
                last_timestamp = timestamp
                last_sender = sender
            else:
                event_match = re.match(event_regex, line)
                if not event_match:
                    messages.append({
                        'timestamp': last_timestamp,
                        'sender': last_sender,
                        'message': line.strip()
                    })
    
    return messages


def dump_messages_to_json(messages, output_path):
    '''
    Dumps the parsed messages to a JSON file.
    '''
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(messages, json_file, ensure_ascii=False, indent=4)


def prepare_training_data(messages, chosen_user, context_size=5):
    '''
    Format WhatsApp messages into structured conversation pairs suitable for fine-tuning.
    '''
    # Initialize the list to store conversation pairs
    conversations = []
    
    # Create conversation pairs
    for i in range(len(messages)):
        # Skip system messages
        if messages[i]['sender'] == 'SYSTEM':
            continue
            
        # If we found a message from the chosen person
        if messages[i]['sender'] == chosen_user:
            # Get context messages
            context = []
            
            # Get the most recent messages before the current response
            count = 0
            for j in range(i-1, -1, -1):
                # Add each previous message separately to the context
                context.insert(0, {
                    'speaker': messages[j]['sender'],
                    'message': messages[j]['message']
                })
                count += 1
                if count >= context_size:
                    break
                    
            # Only create a pair if we have context messages
            if context:
                conversations.append({
                    'context': context,
                    'target_response': messages[i]['message']
                })
    
    return conversations
    

def dump_conversations_to_json(conversations, output_path, randomize=True):
    '''
    Dumps the formatted conversations to a JSON file.
    '''
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Randomize the order of conversations if specified
    if randomize:
        random.shuffle(conversations)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(conversations, json_file, ensure_ascii=False, indent=4)


# Function to split data into train/validation sets
def split_train_val(conversations, train_ratio=0.9, randomize=True):
    '''
    Splits conversations into training and validation sets.
    
    Args:
        conversations: List of conversation pairs
        train_ratio: Ratio of training data (default 0.9)
        randomize: Whether to randomize before splitting (default True)
    
    Returns:
        train_conversations, val_conversations
    '''
    # Create a copy to avoid modifying the original
    conversations_copy = conversations.copy()
    
    # Randomize if specified
    if randomize:
        random.shuffle(conversations_copy)
    
    # Calculate split index
    split_idx = int(len(conversations_copy) * train_ratio)
    
    # Split the data
    train_conversations = conversations_copy[:split_idx]
    val_conversations = conversations_copy[split_idx:]
    
    return train_conversations, val_conversations


# Main function to run the script
if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process WhatsApp chat data with custom naming')
    parser.add_argument('--user', type=str, default='Paolo', help='User to use for training data (default: Paolo)')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Ratio of training data (default: 0.9)')
    
    # Parse arguments
    args = parser.parse_args()
    chosen_user = args.user
    custom_name = chosen_user.replace(' ', '_').lower()
    train_ratio = args.train_ratio
    
    # Set a random seed for reproducibility
    random.seed(42)

    # Check if chat has already been parsed
    if not os.path.exists('data/processed/parsed_chat.json'):
        # Create output directory structure
        os.makedirs('data/processed', exist_ok=True)
        
        all_messages = []
        # Get all files in the data/raw directory
        raw_files_dir = 'data/raw'
        files = os.listdir(raw_files_dir)
        
        # Process each file
        for file in files:
            # Create the full file path
            file_path = os.path.join(raw_files_dir, file)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Parse the WhatsApp chat and append the messages to the list
                file_messages = parse_whatsapp_chat(file_path)
                all_messages.extend(file_messages)
                print(f'Parsed {len(file_messages)} messages from {file}')
        
        # Save all combined messages to JSON with custom name
        parsed_chat_path = f'data/processed/parsed_chat.json'
        dump_messages_to_json(all_messages, parsed_chat_path)
        print(f'Parsed {len(all_messages)} total messages and saved to {parsed_chat_path}')
    else:
        # Load already parsed messages
        with open('data/processed/parsed_chat.json', 'r', encoding='utf-8') as json_file:
            all_messages = json.load(json_file)
    
    conversations = prepare_training_data(all_messages, chosen_user)
    
    # Split into train/validation sets
    train_conversations, val_conversations = split_train_val(conversations, train_ratio=train_ratio, randomize=True)
    
    # Save train and validation sets with custom name
    train_path = f'data/processed/train_conversations_{custom_name}.json'
    val_path = f'data/processed/val_conversations_{custom_name}.json'
    
    dump_conversations_to_json(train_conversations, train_path, randomize=False)
    dump_conversations_to_json(val_conversations, val_path, randomize=False)
    
    print(f'Split data into {len(train_conversations)} training samples and {len(val_conversations)} validation samples')
    print(f'Files saved with "{custom_name}" appended to filenames')