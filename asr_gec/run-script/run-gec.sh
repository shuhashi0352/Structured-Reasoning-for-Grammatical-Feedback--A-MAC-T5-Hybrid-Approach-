#!/bin/bash

# run grammatical error correction (GEC)
# modular pipeline i.e. disfluency detection (DD) followed by GEC

ALLARGS="$0 $@"
# EDIT SCRIPT HERE: set up sandi directory to your location
sandi="."

# set up reference and tools directories
tools_dir=$sandi/baseline-pipeline
ref_dir=$sandi/reference-materials

# set up environment
DD_ENV=$sandi/envs/sandi-dd-env.sh
GEC_ENV=$sandi/envs/sandi-all-env.sh

# Function to print usage information
print_usage() {
    echo "Usage: $0 --ctm <ctm_filepath> --dd_model <model_path> --gec_model <model_path> --out_dir <out_dir>"
    echo "  --ctm          : Normalised input CTM file with phrase markers"
    echo "  --dd_model     : Disfluency detection BERT model"
    echo "  --gec_model    : GEC BART model"
    echo "  --out_dir      : Top directory name for output."
    echo "  --cpu          : (Optional) run on CPU."
    echo "  --num_device   : (Optional) number of GPU device to run on."
    echo "  --help         : Display this help message."
    exit 100
}

# Default values for optional arguments
num_device=0

# Function to parse long options
parse_long_options() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ctm)
                shift
                ctm=$1
		shift
                ;;
            --dd_model)
                shift
                dd_model=$1
		shift
                ;;
            --gec_model)
                shift
                gec_model=$1
		shift
                ;;
            --out_dir)
                shift
                out_dir=$1
		shift
                ;;
            --cpu)
                shift
                cpu=1
                ;;
            --num_device)
                shift
                num_device=$1
		shift
                ;;
            --help)
                print_usage
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                ;;
        esac
    done
}

# Call the function to parse long options
parse_long_options "$@"

# Check if required arguments are provided
if [[ -z "$ctm" ]] || [[ -z "$dd_model" ]] || [[ -z "$gec_model" ]] || [[ -z "$out_dir" ]] ; then
    print_usage
fi

if [ -f $out_dir/gec.ctm ]; then
    echo "ERROR: final output $out_dir/gec.ctm ... delete before running"
    exit 100
fi

# Create CMDs/ directory if it doesn't exist
mkdir -p CMDs/$out_dir

# Cache the command-line arguments
cmdfile=CMDs/$out_dir/run-gec.sh.cmds
echo "$ALLARGS" >> "$cmdfile"
echo "------------------------------------------------------------------------" >> "$cmdfile"

# Print the arguments to verify (optional)
echo "Input CTM:        ${ctm}"
echo "DD model:         ${dd_model}"
echo "GEC model:        ${gec_model}"
echo "Output directory: ${out_dir}"

mkdir -p $out_dir
log_dir=$out_dir/LOGs
mkdir -p $log_dir

echo "Log directory:    ${log_dir}"

# run disfluency detection
eval "$(/Users/hashimotoamaneten/miniconda3/bin/conda shell.bash hook)"
echo "Running disfluency detection"
# source $DD_ENV
conda activate sandi-dd-intel

dd_dir=$out_dir/dd
mkdir -p $dd_dir

python $tools_dir/gec-tools/dd.py \
       --input_ctm $ctm \
       --disfluency_model $dd_model \
       --fluent_ctm $out_dir/fluent.ctm \
       --gec_tsv $dd_dir/pre-gec.tsv \
  2>&1 | tee $log_dir/dd.LOG

# python $tools_dir/gec-tools/dd.py \
#        --input_ctm $ctm \
#        --disfluency_model $dd_model \
#        --fluent_ctm $out_dir/fluent.ctm \
#        --gec_tsv $dd_dir/pre-gec.tsv 2>&1 tee $log_dir/dd.LOG
    #    --gec_tsv $dd_dir/pre-gec.tsv |& tee $log_dir/dd.LOG

if [[ ! -f $out_dir/fluent.ctm ]] || [[ ! -f $dd_dir/pre-gec.tsv ]]; then
    echo "ERROR: didn't find expected outputs of disfluency detection stage"
    exit 100
fi

# run GEC
echo "Running grammatical error correction"
# source $GEC_ENV
conda activate sandi-all-intel

python $tools_dir/gec-tools/gec-plus-align.py \
       --input_file $dd_dir/pre-gec.tsv \
       --gec_model $gec_model \
       --gec_ctm $out_dir/gec.ctm \
  2>&1 | tee $log_dir/gec.LOG

# python $tools_dir/gec-tools/gec-plus-align.py \
#        --input_file $dd_dir/pre-gec.tsv \
#        --gec_model $gec_model \
#        --gec_ctm $out_dir/gec.ctm 2>&1 tee $log_dir/gec.LOG
    #    --gec_ctm $out_dir/gec.ctm |& tee $log_dir/gec.LOG

if [ ! -f $out_dir/gec.ctm ]; then
    echo "ERROR: didn't find expected output of GEC plus align stage"
    exit 100
fi





