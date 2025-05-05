'''
Run the GEC system and produce GEC CTM

Copyright: S&I Challenge 2024
'''
# coding: utf-8
import argparse
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    BartForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import pandas as pd
from ast import literal_eval
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path
from path import makeDir, checkDirExists, checkFileExists, makeCmdPath
from tqdm import tqdm


# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="GEC and CTM generation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input tsv with ids, data_flt, and sentence_flt")
    parser.add_argument("--gec_model", type=str, required=True, help="Path to the GEC model.")
    parser.add_argument("--gec_ctm", type=str, required=True, help="Output path for the GEC CTM file.")
    return parser.parse_args()


def process_and_split_sentences(df, tokenizer):
    # Function to split sentences while keeping the text intact
    def split_sentences(row):
        # Tokenize the sentence and return the token offsets as well
        encoding = tokenizer(row['sentence_flt'], return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding['input_ids']
        token_spans = encoding['offset_mapping']
    
        # Split tokens into chunks of 126 tokens or fewer (128 - 2 special tokens)
        chunk_texts = []
        for i in range(0, len(tokens), 126):
            # Get the current chunk of tokens
            chunk_tokens = tokens[i:i+126]
            chunk_spans = token_spans[i:i+126]
        
            # Get the start and end position of the chunk in the original sentence
            start_idx = chunk_spans[0][0]  # Start character index of the first token in the chunk
            end_idx = chunk_spans[-1][1]  # End character index of the last token in the chunk
        
            # Extract the chunk text from the original sentence
            chunk_texts.append(row['sentence_flt'][start_idx:end_idx])
    
        # Return the resulting chunks with their original ID
        return [(row['ids'], text, row['data_flt']) for text in chunk_texts]
    
    # Apply the function to the DataFrame
    split_data = df.apply(split_sentences, axis=1)

    # Flatten the resulting list of tuples
    split_data_flat = [item for sublist in split_data for item in sublist]

    # Create a new DataFrame with split sentences
    return(pd.DataFrame(split_data_flat, columns=["ids", "sentence_flt", "data_flt"]))

def correct_grammar(input_texts, tokenizer, model, device, batch_size=32, num_return_sequences=1):
    all_tgt_texts = []
    for i in tqdm(range(0, len(input_texts), batch_size), desc="Running GEC"):
        batch_texts = input_texts[i:i+batch_size]
        batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        translated = model.generate(**batch, max_new_tokens=256, num_beams=5, num_return_sequences=num_return_sequences)
        tgt_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        all_tgt_texts.extend(tgt_texts)
    return all_tgt_texts

def align_sentences(row):
    """
    Aligns tokens in the list with the corresponding altered sentence.
    Handles insertions to include modified timing, adjusting the start time
    and duration for insertions before the first word in the original.
    """
    original_tokens = [t[0] for t in row['data_flt']]
    altered_tokens = row['sentence_gec'].split()
    original_data = row['data_flt']
    
    # Match tokens using SequenceMatcher
    matcher = SequenceMatcher(None, original_tokens, altered_tokens)
    aligned_data = []
    used_indices = set()  # Keep track of used indices in original_data
    last_valid_tuple = None  # Track the last valid tuple for calculating insertion timestamps

    # Keep track of whether we are inserting before the first original word
    first_word_start_time = original_data[0][1] if original_data else 0.0
    pre_first_word_insertions = []  # Temporary list for insertions before the first word

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Direct match
            for orig, new in zip(original_tokens[i1:i2], altered_tokens[j1:j2]):
                idx = next(idx for idx in range(i1, i2) if idx not in used_indices)
                used_indices.add(idx)
                tuple_data = original_data[idx]
                aligned_data.append((*tuple_data, new))
                last_valid_tuple = tuple_data  # Update last valid tuple
        elif tag == 'replace':
            # Handle mismatches
            for orig, new in zip(original_tokens[i1:i2], altered_tokens[j1:j2]):
                idx = next(idx for idx in range(i1, i2) if idx not in used_indices)
                used_indices.add(idx)
                tuple_data = original_data[idx]
                aligned_data.append((*tuple_data, new))
                last_valid_tuple = tuple_data  # Update last valid tuple
        elif tag == 'insert':
            # Handle inserted tokens
            for new in altered_tokens[j1:j2]:
                if i1 == 0:  # Check if the insertion is before the first word
                    pre_first_word_insertions.append((None, first_word_start_time, 0.0, None, new))
                else:
                    if last_valid_tuple:
                        new_second = last_valid_tuple[1] + last_valid_tuple[2]
                    else:
                        new_second = 0.0  # If no last valid tuple, start at 0.0
                    aligned_data.append((None, new_second, 0.0, None, new))
        elif tag == 'delete':
            # Handle deleted tokens
            for orig in original_tokens[i1:i2]:
                idx = next(idx for idx in range(i1, i2) if idx not in used_indices)
                used_indices.add(idx)
                tuple_data = original_data[idx]
                aligned_data.append((*tuple_data, None))
                last_valid_tuple = tuple_data  # Update last valid tuple

    # Prepend insertions before the first word to aligned_data
    aligned_data = pre_first_word_insertions + aligned_data

    return aligned_data

def align_dataset(df, list_col='data_flt', sentence_col='sentence_gec'):
    """
    Aligns tokens in each row of the dataset.
    """
    df['data_gec'] = df.apply(align_sentences, axis=1)
    return df

# Main function
def main(input_file, gec_model_path, gec_ctm_path):
    # torch.cuda.set_device(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    df = pd.read_csv(input_file, sep='\t')
    df['data_flt'] = df['data_flt'].apply(literal_eval)
    
    # Load tokenizer and model
    model_name = gec_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Process and split sentences
    df = process_and_split_sentences(df, tokenizer)
    
    #Run GEC
    input_texts = df['sentence_flt'].tolist()
    df['sentence_gec'] = correct_grammar(input_texts, tokenizer=tokenizer, model=model, device=device, batch_size=32)
    
    # Combine contiguous sentences
    
    result = []
    prev_row = None
    for _, row in df.iterrows():
        if prev_row is not None and prev_row['ids'] == row['ids']:
            prev_row['sentence_flt'] += ' ' + row['sentence_flt']
            prev_row['sentence_gec'] += ' ' + row['sentence_gec']
        else:
            if prev_row is not None:
                result.append(prev_row)
            prev_row = row.copy()
    if prev_row is not None:
            result.append(prev_row)
    df = pd.DataFrame(result)
    
    #Align GEC with FLT timestamps and info
    df = align_dataset(df)
    #Remove flt word from gec tuples
    df['data_gec'] = df['data_gec'].apply(lambda x: [t[1:] for t in x])
    # Save GEC CTM
    gec_tuple = pd.DataFrame(df['data_gec'].explode())
    pre_ctm = pd.merge(left=df[['ids']], right=gec_tuple, right_index=True, left_index=True).reset_index(drop=True)
    ctm = pd.concat([pre_ctm[['ids']], pd.DataFrame(pre_ctm['data_gec'].to_list(), \
                                                    columns=['start', 'duration', 'prob', 'word'])], axis=1)
    ctm['channel'] = 1
    ctm = ctm[['ids', 'channel', 'start', 'duration', 'word', 'prob']]
    #Drop 'deletions'
    ctm = ctm.loc[ctm['word'].notna()].reset_index(drop=True)
    #Arbitrarily set 0.5 as prob for 'insertions'
    ctm['prob'] = ctm['prob'].fillna(0.5)
    ctm.to_csv(gec_ctm_path, sep='\t', header=None, index=None)
    print("GEC complete.")

if __name__ == "__main__":
    args = parse_arguments()

    # file/dir checks and set-up
    checkFileExists(args.input_file)
    checkFileExists(args.gec_model)
    
    output_dir = str(Path(args.gec_ctm).parent)
    makeDir ( output_dir, False )            

    makeCmdPath ( output_dir )
    
    # run GEC
    main(args.input_file, args.gec_model, args.gec_ctm)






