'''
Normalise a CTM for one of input to: ASR WER scoring; spoken language assessment (SLA); disfluency detection (DD)/grammatical error correction (GEC)

Copyright: S&I Challenge 2024
'''
import os
import argparse
import sys
import re
from text_normalisation import lower_case, remove_punctuation, convert_to_ascii, remove_hesitation, remove_partial_words, ordinal_numbers, num2word, special_symbols, remove_hyphens
from path import checkFileExists, checkDirExists, makeDir, makeCmdPath

def convert_ctm( ctm_file, task='asr' ):
    modified_ctm_lines = []

    with open(ctm_file, 'r') as ctm_file:
        for line in ctm_file:
            ctm_data = line.strip().split()
            # CTM format: <file_id> <channel> <start_time> <duration> <word> <confidence>
            word_text = ctm_data[4] 
            word_text = convert_to_ascii(word_text) 
            word_text = lower_case(word_text)
            word_text = remove_punctuation(word_text)
            word_text = remove_hyphens(word_text)
            word_text = special_symbols(word_text)
            word_text = num2word(word_text)
            word_text = remove_hyphens(word_text)
            word_text = ordinal_numbers(word_text)
            if task == 'asr' or task == 'dd' or task == 'gec':
                word_text = remove_hesitation(word_text)
                word_text = remove_partial_words(word_text)

            
            # Update the word text in the CTM line, skip empty lines
            # check if word_text contains more than one word, if so, split it and add to the list in different lines
            if ' ' in word_text:
                words = word_text.split()
                duration = float(ctm_data[3])/len(words)
                formatted_duration = "{:.2f}".format(duration) 
                for word_index, word in enumerate(words):
                    # start_time = str(float(ctm_data[2]) +  word_index * duration)
                    if word:
                        start_time = str(float(ctm_data[2]) +  word_index * duration)
                        # keep start time in 2 decimal places
                        start_time = "{:.2f}".format(float(start_time))
                        ctm_data[2] = start_time
                        # ctm_data[3] = str(duration)
                        ctm_data[3] = formatted_duration
                        ctm_data[4] = word
                        modified_ctm_line = ' '.join(ctm_data) + '\n'
                        modified_ctm_lines.append(modified_ctm_line)
                # continue
            else:
                ctm_data[4] = word_text
                if word_text:
                    modified_ctm_line = ' '.join(ctm_data) + '\n'
                    modified_ctm_lines.append(modified_ctm_line)
            
    return modified_ctm_lines

def modify_ctm(ctm_files, output_file, task='asr'):
    modified_ctm_lines = []

    for ctm_file in ctm_files:
        rev_ctm_lines = convert_ctm ( ctm_file, task )
        modified_ctm_lines.extend(rev_ctm_lines)
        
    # Sort ctm_lines based on the filename
    modified_ctm_lines.sort(key=lambda x: x.split()[0])

    # Write the modified CTM lines to the output CTM file for all tasks
    with open(output_file, 'w') as ctm_out:
        ctm_out.writelines(modified_ctm_lines)

def main():
    parser = argparse.ArgumentParser(description="Modify CTM files based on norm type")
    parser.add_argument("--ctm_file", required=True, help="Path to the input CTM file")
    parser.add_argument("--output_file", required=True, help="Path to the output modified CTM file")
    parser.add_argument("--task", default='asr', help="Task type (sla or asr or gec or dd)")
    args = parser.parse_args()

    # Ensure the input files exists
    ctm_files = args.ctm_file.split()
    for ctm_file in ctm_files:
        checkFileExists ( ctm_file )

    out_dir = os.path.dirname ( args.output_file )
    if len(out_dir) > 0:
        makeDir ( out_dir, False )
        
    # Save the command line input
    makeCmdPath ( out_dir )
    
    # Process the CTM file(s)
    modify_ctm(ctm_files, args.output_file, args.task)

if __name__ == "__main__":
    main()
