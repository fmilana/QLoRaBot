import os
import re
import json
import random
import pandas as pd
import argparse
from datasets import Dataset


def parse_whatsapp_chat(file_path):
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(messages, json_file, ensure_ascii=False, indent=4)


def prepare_training_data(messages, chosen_user, context_size):
    conversations = []
    
    for i in range(len(messages)):
        if messages[i]['sender'] == 'SYSTEM':
            continue
            
        if messages[i]['sender'] == chosen_user:
            context = []
            
            count = 0
            for j in range(i-1, -1, -1):
                context.insert(0, {
                    'speaker': messages[j]['sender'],
                    'message': messages[j]['message']
                })
                count += 1
                if count >= context_size:
                    break
                    
            if context:
                conversations.append({
                    'context': context,
                    'target_response': messages[i]['message']
                })
    
    return conversations
    

def dump_conversations_to_json(conversations, output_path, randomize=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if randomize:
        random.shuffle(conversations)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(conversations, json_file, ensure_ascii=False, indent=4)


def split_train_val(conversations, train_ratio=0.9, randomize=True):
    conversations_copy = conversations.copy()
    
    if randomize:
        random.shuffle(conversations_copy)
    
    split_idx = int(len(conversations_copy) * train_ratio)
    
    train_conversations = conversations_copy[:split_idx]
    val_conversations = conversations_copy[split_idx:]
    
    return train_conversations, val_conversations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process WhatsApp chat data with custom naming')
    parser.add_argument('--user', type=str, default='Paolo', help='User to use for training data (default: Paolo)')
    parser.add_argument('--context-size', type=int, default=5, help='Number of context messages to include (default: 5)')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Ratio of training data (default: 0.9)')
    
    args = parser.parse_args()
    chosen_user = args.user
    custom_name = chosen_user.replace(' ', '_').lower()
    context_size = args.context_size
    train_ratio = args.train_ratio
    
    random.seed(42)

    if not os.path.exists('data/processed/parsed_chat.json'):
        os.makedirs('data/processed', exist_ok=True)
        
        all_messages = []
        raw_files_dir = 'data/raw'
        files = os.listdir(raw_files_dir)
        
        for file in files:
            file_path = os.path.join(raw_files_dir, file)
            
            if os.path.isfile(file_path):
                file_messages = parse_whatsapp_chat(file_path)
                all_messages.extend(file_messages)
                print(f'Parsed {len(file_messages)} messages from {file}')
        
        parsed_chat_path = f'data/processed/parsed_chat.json'
        dump_messages_to_json(all_messages, parsed_chat_path)
        print(f'Parsed {len(all_messages)} total messages and saved to {parsed_chat_path}')
    else:
        with open('data/processed/parsed_chat.json', 'r', encoding='utf-8') as json_file:
            all_messages = json.load(json_file)
    
    conversations = prepare_training_data(all_messages, chosen_user, context_size=context_size)
    
    train_conversations, val_conversations = split_train_val(conversations, train_ratio=train_ratio, randomize=True)
    
    train_path = f'data/processed/train_conversations_{custom_name}.json'
    val_path = f'data/processed/val_conversations_{custom_name}.json'
    
    dump_conversations_to_json(train_conversations, train_path, randomize=False)
    dump_conversations_to_json(val_conversations, val_path, randomize=False)
    
    print(f'Split data into {len(train_conversations)} training samples and {len(val_conversations)} validation samples')
    print(f'Files saved with "{custom_name}" appended to filenames')