'''
Create files for ERRANT 

Copyright: S&I Challenge 2024
'''
# coding: utf-8
import pandas as pd
import numpy as np
import argparse
import spacy
nlp = spacy.load('en_core_web_sm')

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Files generator for ERRANT")
    parser.add_argument("--flt_stm", type=str, required=True, help="Path to the fluent input STM file (reference).")
    parser.add_argument("--flt_ctm", type=str, required=True, help="Path to the fluent input CTM file (hypothesis).")
    parser.add_argument("--gec_stm", type=str, required=True, help="Path to the GEC input STM file (reference).")
    parser.add_argument("--gec_ctm", type=str, required=True, help="Path to the GEC input CTM file (hypothesis).")
    parser.add_argument("--ref_src", type=str, required=True, help="Output path for the source reference file.")
    parser.add_argument("--hyp_src", type=str, required=True, help="Output path for the source hypothesis file.")
    parser.add_argument("--ref_tgt", type=str, required=True, help="Output path for the target reference file.")
    parser.add_argument("--hyp_tgt", type=str, required=True, help="Output path for the target hypothesis file.")
    return parser.parse_args()

def load_stm(file_path):

    # Initialize a list to store valid rows
    data = []

    # Open and process the file
    with open(file_path, 'r') as f:
        for line in f:
            # Skip metadata lines and empty lines
            if line.startswith(';;') or line.strip() == '':
                continue
            # Split the line into fixed columns and the remaining transcription
            parts = line.split(maxsplit=5)
            data.append(parts)

    # Create a DataFrame from the processed data
    # Define column names based on the data structure
    df = pd.DataFrame(data, columns=["ID", "Category", "Code", "Start", "End", "Transcription_Info"])

    # Further split the Transcription_Info into `Info` and `Transcription`
    df[["Info", "Transcription"]] = df["Transcription_Info"].str.extract(r"<(.*?)>\s*(.*)")

    # Drop the original Transcription_Info column
    df.drop(columns=["Transcription_Info"], inplace=True)
    return df


def get_non_scorable_words(ignored_segments, ctm):
    to_eliminates = []
    # Exclude rows in df2 that overlap with ignored segments in df1
    for _, row in ignored_segments.iterrows():
        base_id = row['ID']
        start, end = float(row['Start']), float(row['End'])
    
    
        to_eliminate = ctm.loc[(ctm['Base_ID'] == base_id) & (ctm[2] >= start) & (ctm[2] < end) & (ctm[2]+ctm[3] <= end)]
    
        to_eliminates.append(to_eliminate)
        
    ignored_words = pd.concat(to_eliminates)
    return(ignored_words)

# Main function
def main(flt_stm, flt_ctm, gec_stm, gec_ctm, ref_src, hyp_src, ref_tgt, hyp_tgt):
    # Load and preprocess data
    stm_flt = load_stm(flt_stm)
    # Filter segments with "IGNORE_TIME_SEGMENT_IN_SCORING"
    ignored_segments_flt = stm_flt[stm_flt['Transcription'] == "IGNORE_TIME_SEGMENT_IN_SCORING"]
    
    #Load ctm
    ctm_flt = pd.read_csv(flt_ctm, sep='\t', header=None)
    # Create a helper column in df2 for base ID (strip "_PHxx")
    ctm_flt['Base_ID'] = ctm_flt[0].str.rsplit('_', n=1).str[0]
    
    ignored_words_flt = get_non_scorable_words(ignored_segments_flt, ctm_flt)
    
    filtered_ctm_flt = ctm_flt.loc[~ctm_flt.index.isin(ignored_words_flt.index)]
    filtered_stm_flt = stm_flt.loc[~stm_flt.index.isin(ignored_segments_flt.index)]
    
    # Group by 'ID' and concatenate the words
    flt_hyp = filtered_ctm_flt.groupby('Base_ID')[4].apply(' '.join).reset_index()
    # Rename columns (optional)
    flt_hyp.columns = ['ID', 'flt_hyp']
    
    # Group by 'ID' and concatenate the words
    flt_ref = filtered_stm_flt[['ID', 'Transcription']].groupby('ID')['Transcription'].apply(' '.join).reset_index()

    # Rename columns (optional)
    flt_ref.columns = ['ID', 'flt_ref']
    
    flt = pd.merge(left=flt_ref, right=flt_hyp, on='ID', how='inner')
    
    #DO THE SAME WITH GEC
    
    # Load and preprocess data
    stm_gec = load_stm(gec_stm)
    # Filter segments with "IGNORE_TIME_SEGMENT_IN_SCORING"
    ignored_segments_gec = stm_gec[stm_gec['Transcription'] == "IGNORE_TIME_SEGMENT_IN_SCORING"]
    
    #Load ctm
    ctm_gec = pd.read_csv(gec_ctm, sep='\t', header=None)
    # Create a helper column in df2 for base ID (strip "_PHxx")
    ctm_gec['Base_ID'] = ctm_gec[0].str.rsplit('_', n=1).str[0]
    
    ignored_words_gec = get_non_scorable_words(ignored_segments_gec, ctm_gec)
    
    filtered_ctm_gec = ctm_gec.loc[~ctm_gec.index.isin(ignored_words_gec.index)]
    filtered_stm_gec = stm_gec.loc[~stm_gec.index.isin(ignored_segments_gec.index)]
    
    # Group by 'ID' and concatenate the words
    gec_hyp = filtered_ctm_gec.groupby('Base_ID')[4].apply(' '.join).reset_index()
    # Rename columns (optional)
    gec_hyp.columns = ['ID', 'gec_hyp']
    
    # Group by 'ID' and concatenate the words
    gec_ref = filtered_stm_gec[['ID', 'Transcription']].groupby('ID')['Transcription'].apply(' '.join).reset_index()

    # Rename columns (optional)
    gec_ref.columns = ['ID', 'gec_ref']
    
    gec = pd.merge(left=gec_ref, right=gec_hyp, on='ID', how='left')
    
    #FINAL MERGE
    final = pd.merge(left=flt, right=gec, on='ID')
    
    #TOKENIZATION FOR ERRANT
    final['flt_ref'] = final['flt_ref'].apply(lambda x: ' '.join([y.text for y in nlp(str(x))]))
    final['flt_hyp'] = final['flt_hyp'].apply(lambda x: ' '.join([y.text for y in nlp(str(x))]))
    
    final['gec_ref'] = final['gec_ref'].apply(lambda x: ' '.join([y.text for y in nlp(str(x))]))
    final['gec_hyp'] = final['gec_hyp'].apply(lambda x: ' '.join([y.text for y in nlp(str(x))]))
   

    # Save files
    final[['flt_ref']].to_csv(ref_src, sep='\t', header=None, index=None)
    final[['flt_hyp']].to_csv(hyp_src, sep='\t', header=None, index=None)
    final[['gec_ref']].to_csv(ref_tgt, sep='\t', header=None, index=None)
    final[['gec_hyp']].to_csv(hyp_tgt, sep='\t', header=None, index=None)
    print("Process completed.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.flt_stm, args.flt_ctm, args.gec_stm, args.gec_ctm, args.ref_src, args.hyp_src, args.ref_tgt, args.hyp_tgt)

