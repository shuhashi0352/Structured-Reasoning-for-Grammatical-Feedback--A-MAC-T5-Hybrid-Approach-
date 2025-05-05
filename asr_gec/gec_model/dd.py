'''
Runs BERT-based disfluency detection and removes disfluences from the transcription

Copyright: S&I Challenge 2024
'''
# coding: utf-8
import pandas as pd
import transformers
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification, AutoConfig
import argparse
import os

from torch import cuda
from pathlib import Path
from path import makeDir, checkDirExists, checkFileExists, makeCmdPath
from tqdm import tqdm


MAX_LEN = 128 # Set a max token length for BERT input.

# Label mapping
labels_to_ids = {'corr': 0, 'disf': 1}
ids_to_labels = {0: 'corr', 1: 'disf'}


# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Disfluency Detection and Fluent CTM Generation")
    parser.add_argument("--input_ctm", type=str, required=True, help="Path to the input CTM file.")
    parser.add_argument("--disfluency_model", type=str, required=True, help="Path to the disfluency detection model.")
    parser.add_argument("--fluent_ctm", type=str, required=True, help="Output path for the fluent CTM file.")
    parser.add_argument("--gec_tsv", type=str, required=True, help="Output path for the pre-GEC TSV file.")
    return parser.parse_args()

# Load data and preprocess
def load_and_group_data(input_file):
    ### ARGs 
    ### => take .ctm as input / separate columns by whitespace / the first row is not a header / allow using regex for "sep=" ("\s" is a regex representation)
    df = pd.read_csv(input_file, sep='\s', header=None, engine='python')
    # Group by 'ids_specific' to aggregate data for each unique value
    grouped_df = df.groupby(0).agg({
        4: lambda words: ' '.join(words),  # Join words to form a sentence
        2: list,                          # Collect start times
        3: list, # Collect durations
        5: list, # Collect probabilities
    }).reset_index()

    print(grouped_df)

    # Create the 'word_data' column by zipping (word, start, duration) for each group
    grouped_df[6] = grouped_df.apply(lambda row: list(zip(row[4].split(), row[2], row[3], row[5])), axis=1)

    # Drop the separate 'start' and 'duration' columns, as they're now in 'word_data'
    grouped_df = grouped_df[[0, 4, 6]]

    grouped_df.columns = ['ids', 'sentence', 'data']
    
    #Make sure there are no "empty sentences"

    grouped_df = grouped_df.loc[grouped_df['data'].apply(len) != 0].reset_index(drop=True)
    return grouped_df


def process_and_split_sentences(grouped_df, tokenizer):
    # Function to split sentences while keeping the text intact
    def split_sentences(row):
        # Tokenize the sentence and return the token offsets as well
        encoding = tokenizer(row['sentence'], return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding['input_ids']
        token_spans = encoding['offset_mapping']
    
        # Split tokens into chunks of 126 tokens or fewer (128 - 2 special tokens)
        chunk_texts = []
        # for i in range(0, len(tokens), 126):
        for i in tqdm(range(0, len(tokens), 126), desc="Token windows", leave=False):
            # Get the current chunk of tokens
            chunk_tokens = tokens[i:i+126]
            chunk_spans = token_spans[i:i+126]
        
            # Get the start and end position of the chunk in the original sentence
            start_idx = chunk_spans[0][0]  # Start character index of the first token in the chunk
            end_idx = chunk_spans[-1][1]  # End character index of the last token in the chunk
        
            # Extract the chunk text from the original sentence
            chunk_texts.append(row['sentence'][start_idx:end_idx])
    
        # Return the resulting chunks with their original ID
        return [(row['ids'], text, row['data']) for text in chunk_texts]
    
    # Apply the function to the DataFrame
    split_data = grouped_df.apply(split_sentences, axis=1)

    # Flatten the resulting list of tuples
    split_data_flat = [item for sublist in split_data for item in sublist]

    # Create a new DataFrame with split sentences
    return(pd.DataFrame(split_data_flat, columns=["ids", "sentence", "data"]))

def disf_detect(sentence, model, tokenizer, max_len, device):
    inputs = tokenizer(str(sentence).split(), is_pretokenized=True, return_offsets_mapping=True, \
                       padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

    import torch.nn.functional as F
    probs = F.softmax(outputs[0], dim=1).view(-1, model.num_labels)

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions, probs)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    prob_list = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
            prob_list.append(token_pred[2])
        else:
            continue
    return(prediction)


# Main function
def main(input_file, disfluency_model_path, fluent_ctm_path, gec_tsv_path):    
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and preprocess data
    grouped_df = load_and_group_data(input_file)
    
    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', is_pretokenized=True)
    model = torch.load(disfluency_model_path, map_location=torch.device("cpu")).to(device)
    
    # Process and split sentences
    split_df = process_and_split_sentences(grouped_df, tokenizer)

    tqdm.pandas(desc="Running disfluency detection")
    split_df['disf_labels'] = split_df['sentence'].progress_apply(
    lambda x: disf_detect(x, model, tokenizer, MAX_LEN, device)
    )
    # split_df['disf_labels'] = split_df['sentence'].apply(lambda x: disf_detect(x, model, tokenizer, MAX_LEN, device))
    
    # Combine contiguous sentences and labels
    result = []
    prev_row = None
    for _, row in split_df.iterrows():
        if prev_row is not None and prev_row['ids'] == row['ids']:
            prev_row['sentence'] += ' ' + row['sentence']
            prev_row['disf_labels'].extend(row['disf_labels'])
        else:
            if prev_row is not None:
                result.append(prev_row)
            prev_row = row.copy()
    if prev_row is not None:
        result.append(prev_row)
    result_df = pd.DataFrame(result)

    # Filter and save results
    result_df['data+disf'] = result_df.apply(lambda x: [tup + (num,) for tup, num in zip(x.data, x.disf_labels)], axis=1)
    result_df['data_flt'] = result_df['data+disf'].apply(lambda x: [y for y in x if y[4] == 'corr'])
    result_df['sentence_flt'] = result_df['data_flt'].apply(lambda x: ' '.join([y[0] for y in x]))
    result_df.drop(columns=['data+disf', 'disf_labels'], inplace=True)
    result_df['data_flt'] = result_df['data_flt'].apply(lambda x: [t[:-1] for t in x])
    result_df.reset_index(drop=True, inplace=True)

    # Save Fluent CTM
    flt_tuple = pd.DataFrame(result_df['data_flt'].explode())
    pre_ctm = pd.merge(left=result_df[['ids']], right=flt_tuple, right_index=True, left_index=True).reset_index(drop=True)
    ctm = pd.concat([pre_ctm[['ids']], pd.DataFrame(pre_ctm['data_flt'].to_list(), columns=['word', 'start', 'duration', 'prob'])], axis=1)
    ctm['channel'] = 1
    ctm = ctm[['ids', 'channel', 'start', 'duration', 'word', 'prob']]
    ctm.to_csv(fluent_ctm_path, sep='\t', header=None, index=None)

    # Save pre-GEC TSV
    result_df[['ids', 'data_flt', 'sentence_flt']].to_csv(gec_tsv_path, sep='\t', header=True, index=None)
    print("DD complete.")

if __name__ == "__main__":
    args = parse_arguments()

    # file/dir checks and set-up
    checkFileExists(args.input_ctm)
    checkFileExists(args.disfluency_model)
    
    output_dir = str(Path(args.fluent_ctm).parent)
    makeDir ( output_dir, False )            
    output_dir = str(Path(args.gec_tsv).parent)
    makeDir ( output_dir, False )            

    makeCmdPath ( output_dir )
    
    # run disfluency detection
    main(args.input_ctm, args.disfluency_model, args.fluent_ctm, args.gec_tsv)
