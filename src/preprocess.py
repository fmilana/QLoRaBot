import os
import re
import json
import random
import pandas as pd
import argparse
from datasets import Dataset
from datetime import datetime


def parse_whatsapp_chat(file_path, is_private):
    regex = r'(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - ([^:]+): (.*)'
    event_regex = r'(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.+)'  # For system messages/events
    filtered_contents = ['<Media omitted>', '<Video note omitted>', 'This message was deleted', 'You deleted this message', 'Shared route', 'POLL:', 'EVENT:', 'location:', 'Missed voice call']
    link_regex = r'^(https?://[^\s]+)' 
    messages = []

    last_sender = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(regex, line)
            
            if match:
                timestamp, sender, message = match.groups()

                message = message.replace('<This message was edited>', '')
                message = message.replace('This message was edited', '')
                message = message.strip()

                # If the message is empty after cleaning, skip it
                if not message:
                    continue

                to_skip = False

                if re.match(link_regex, message):
                    to_skip = True
                else:
                    for filtered_message in filtered_contents:
                        if filtered_message in message:
                            to_skip = True
                            break

                if to_skip:
                    messages.append({
                        'chat_id': os.path.basename(file_path).replace('.txt', ''),
                        'chat_type': 'private' if is_private else 'group',
                        'sender': sender,
                        'message': message.strip(),
                        'filtered': True
                    })
                    continue

                # Check if this message is from the same sender as the previous one (and previous is not filtered)
                if messages and sender == messages[-1]['sender'] and not messages[-1]['filtered']:
                    # Concatenate with the previous message using \n
                    messages[-1]['message'] += '\n' + message.strip()
                else:
                    # Add as a new message
                    messages.append({
                        'chat_id': os.path.basename(file_path).replace('.txt', ''),
                        'chat_type': 'private' if is_private else 'group',
                        'sender': sender,
                        'message': message.strip(),
                        'filtered': False
                    })
                
                last_sender = sender
            else:
                event_match = re.match(event_regex, line)
                if not event_match and messages:  # Make sure we have at least one message
                    # This is a continuation of the previous message
                    line = line.replace('<This message was edited>', '')
                    line = line.replace('This message was edited', '')
                    line = line.strip()
                    messages[-1]['message'] += '\n' + line.strip()
    
    return messages


def dump_messages_to_json(messages, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(messages, json_file, ensure_ascii=False, indent=4)


def prepare_data(messages, chosen_user, context_window=10):
    conversations = []
    
    for i in range(len(messages)):
        if messages[i]['filtered']:
            continue  # Skip filtered target messages
            
        if messages[i]['sender'] == chosen_user:
            context = []
            added_messages = 0
            j = i - 1
            
            while j >= 0 and added_messages < context_window:
                if messages[j]['filtered']:
                    break  # Stop if we hit a filtered message (breaks consecutiveness)

                context.insert(0, {
                    'speaker': messages[j]['sender'],
                    'message': messages[j]['message']
                })
                added_messages += 1
                j -= 1
            
            if len(context) == context_window:  # Only include full-length contexts
                conversations.append({
                    'chat_type': messages[i]['chat_type'],
                    'context': context,
                    'target_response': messages[i]['message']
                })
    
    return conversations


def dump_conversations_to_json(conversations, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(conversations, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process WhatsApp chat data with custom naming')
    parser.add_argument('--user', type=str, default='Paolo', help='User to use for training data (default: Paolo)')
    parser.add_argument('--context-window', type=int, default=10, help='Number of context messages to include (default: 10)')
    parser.add_argument('--val-chat', type=str, default='bombardiro.txt', help='Chat file to use for validation (default: bombardiro.txt)')
    parser.add_argument('--private', type=str, default='no', help='Include private chats in training data or not, or only use private chats (yes/no/only)', choices=['yes', 'no', 'only'])
    
    args = parser.parse_args()
    chosen_user = args.user
    custom_name = chosen_user.replace(' ', '_').lower()
    context_window = args.context_window
    val_chat = args.val_chat
    private = args.private
    
    random.seed(42)

    os.makedirs('data/processed', exist_ok=True)
    
    train_messages = []
    val_messages = []
    raw_files_dir = 'data/raw'
    files = os.listdir(raw_files_dir)
    
    # Process each chat file
    for file in files:
        if not file.endswith('uncleaned.txt') and not file.endswith('.gitignore'):
            file_path = os.path.join(raw_files_dir, file)
            
            if os.path.isfile(file_path):
                is_private = file == 'private_paolo.txt'
                
                # Skip private chats if not included
                if private == 'no' and is_private:
                    print(f'Skipping private chat {file} (use --private yes or only to include)')
                    continue
                elif private == 'only' and not is_private:
                    print(f'Skipping group chat {file} (use --private yes to also include group chats)')
                    continue
                    
                file_messages = parse_whatsapp_chat(file_path, is_private)
                
                # If this is our validation chat, add to validation set
                # only runs if private is not 'only'
                if file == val_chat:
                    val_messages.extend(file_messages)
                    print(f'Parsed {len(file_messages)} messages from {file} for VALIDATION')
                else:
                    train_messages.extend(file_messages)
                    print(f'Parsed {len(file_messages)} messages from {file} for TRAINING')

    if private == 'only':
        # use the last 10% of the private chat for validation
        num_samples = len(train_messages)
        val_split_index = int(num_samples * 0.9)
        val_messages = train_messages[val_split_index + context_window:]
        train_messages = train_messages[:val_split_index]
    
    # Save all parsed messages to a single file
    parsed_chats_path = f'data/processed/parsed_chats.json'
    all_messages = train_messages + val_messages
    dump_messages_to_json(all_messages, parsed_chats_path)
    print(f'Parsed {len(all_messages)} total messages and saved to {parsed_chats_path}')
    
    # Prepare training and validation conversations
    train_conversations = prepare_data(train_messages, chosen_user, context_window=context_window)
    val_conversations = prepare_data(val_messages, chosen_user, context_window=context_window)

    # Final save paths
    train_path = f'data/processed/train_conversations_{custom_name}.json'
    val_path = f'data/processed/val_conversations_{custom_name}.json'

    # Save to JSON
    dump_conversations_to_json(train_conversations, train_path)
    dump_conversations_to_json(val_conversations, val_path)

    # Calculate private and group message counts for reporting
    private_count = sum(1 for msg in train_messages if msg['chat_type'] == 'private')
    group_count = len(train_messages) - private_count
    
    print(f"TRAINING SET: {len(train_conversations)} samples ({group_count} from group chats, {private_count} from private chats).")
    print(f"VALIDATION SET: {len(val_conversations)} samples from {val_chat}.")