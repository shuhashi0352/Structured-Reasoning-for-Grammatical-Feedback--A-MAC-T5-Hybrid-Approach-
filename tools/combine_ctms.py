'''
Combine CTMs, optional select only subset
'''
import os
import argparse
import sys
from pathlib import Path
from path import makeDir, checkDirExists, checkFileExists, makeCmd, makeCmdPath

def read_ctm ( ctm_fname ):
    ctm_info = {}
    fileids = []
    for line in open (ctm_fname):
        if line.strip():
            lineitems = line.strip().split()
            fileid = lineitems[0]
            if fileid not in ctm_info:
                ctm_info[fileid] = []
                fileids.append(fileid)
            ctm_info[fileid].append(line.strip())
    return fileids, ctm_info

def main (args):
    #------------------------------------------------------------------------------
    # read in command line arguments
    #------------------------------------------------------------------------------
    output_ctm = args.output_ctm

    input_ctms = args.input_ctm.split()
    for input_ctm in input_ctms:
        checkFileExists ( input_ctm )
    if args.wav_list:
        wav_list = args.wav_list
        checkFileExists(wav_list)
        sel_wavs = True
    else:
        sel_wavs = False
    output_dir = str(Path(output_ctm).parent)
    makeDir ( output_dir, False )        
     
    #------------------------------------------------------------------------------
    # save command line arguments to file
    #------------------------------------------------------------------------------
    makeCmdPath (output_dir)

    #------------------------------------------------------------------------------
    # read in CTM and write out sorted info
    #------------------------------------------------------------------------------
    wavids=[]
    if sel_wavs is True:        
        fp = open(wav_list,'r')
        for line in fp:
            if line.strip():
                wavid, wav = line.strip().split()
                wavids.append(wavid)
        fp.close()

    all_fileids=[]
    all_ctm_info={}
    for input_ctm in input_ctms:
        fileids, ctm_info = read_ctm (input_ctm)
        all_fileids.extend(fileids)
        all_ctm_info.update(ctm_info)
        
    all_fileids = sorted(all_fileids)
    fp = open(output_ctm, 'w')
    for fileid in all_fileids:
        if sel_wavs is True and fileid not in wavids:
            continue
        for line in all_ctm_info[fileid]:
            print(line, file=fp)
    fp.close()

if __name__ == "__main__":
    #------------------------------------------------------------------------------
    # arguments
    #------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Combine CTMs, for optional subset")
    parser.add_argument('--input_ctm', required=True, help="Path to the input CTM files")
    parser.add_argument('--wav_list', required=False, help="Wavlist for files to keep")
    parser.add_argument('--output_ctm', required=True, help="Path to the output CTM")

    args = parser.parse_args()
    main(args)
                
    
