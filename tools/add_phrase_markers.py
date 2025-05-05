'''
Add phrase markers to files in the CTM into phrases - use Whisper punctuation to do so

Copyright: S&I Challenge 2024
'''
import os
import argparse
import sys
from pathlib import Path
from path import makeDir, checkDirExists, checkFileExists, makeCmd, makeCmdPath

def add_phrase_markers ( ctm_fname ):
    phrase_punct = '!?.'
    
    ctm_info = {}
    fileids=[]
    for line in open (ctm_fname):
        if line.strip():
            lineitems = line.strip().split()
            fileid = lineitems[0]
            if fileid not in ctm_info:
                ctm_info[fileid] = []
                fileids.append(fileid)
                phrase_num = 1
            token = lineitems[4]
            phrase_id = phrase_id=('%s_PH%02d' % (fileid, phrase_num))
            phrase_stg = phrase_id + ' ' + ' '.join(lineitems[1:])
            ctm_info[fileid].append(phrase_stg)
            if token[-1] in phrase_punct:
                # end of a phrase
                phrase_num = phrase_num + 1
    return ctm_info

def main (args):
    #------------------------------------------------------------------------------
    # read in command line arguments
    #------------------------------------------------------------------------------
    input_ctm = args.input_ctm
    output_ctm = args.output_ctm

    checkFileExists ( input_ctm )
    output_dir = str(Path(output_ctm).parent)
    makeDir ( output_dir, False )        
     
    #------------------------------------------------------------------------------
    # save command line arguments to file
    #------------------------------------------------------------------------------
    makeCmdPath (output_dir)

    #------------------------------------------------------------------------------
    # read in CTM and write out phrase marked CTM
    #------------------------------------------------------------------------------
    ctm_info = add_phrase_markers (input_ctm)
    fp = open(output_ctm, 'w')
    for fileid in ctm_info:
        for line in ctm_info[fileid]:
            print(line,file=fp)
    fp.close()


if __name__ == "__main__":
    #------------------------------------------------------------------------------
    # arguments
    #------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Split file entries in CTM into phrases")
    parser.add_argument('--input_ctm', required=True, help="Path to the input CTM file")
    parser.add_argument('--output_ctm', required=True, help="Path to the output CTM file")

    args = parser.parse_args()
    main(args)

