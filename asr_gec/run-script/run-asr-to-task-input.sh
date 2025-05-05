#!/bin/bash

# Convert CTM output from one or more ASR runs into normalised CTM file for use in GEC pipeline

ALLARGS="$0 $@"

# EDIT SCRIPT HERE: set up sandi directory to your location
sandi="."

# set up reference and tools directories
tools_dir=$sandi/baseline-pipeline/tools
ref_dir=$sandi/reference-materials

# set up environment
ENV=$sandi/envs/sandi-all-env.sh

# Function to print usage information
print_usage() {
    echo "Usage: $0 --input_ctms <asr_ctm1> [asr_ctm2 ...] --flist <file_list> --task <task> --out_dir <output_dir>"
    echo "  --input_ctms   : Path to the input CTMs from ASR."
    echo "  --flist        : Path to the file list."
    echo "  --task         : Target task: asr/gec/sla-P{1,3,4,5}/sla-overall."
    echo "  --out_dir      : Experimental output directory."
    echo "  --help         : Display this help message."
    exit 100
}

# Function to parse long options
parse_long_options() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input_ctms)
                shift
                input_ctms=$1
		shift
                ;;
            --flist)
                shift
                flist=$1
		shift
                ;;
            --out_dir)
                shift
                out_dir=$1
		shift
                ;;
            --task)
                shift
                task=$1
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
if [[ -z "$flist" ]] || [[ -z "$input_ctms" ]] || [[ -z "$out_dir" ]] || [[ -z "$task" ]] ; then
    print_usage
fi

if [ ! -f $flist ]; then
    echo "ERROR: file list $flist not found."
    exit 100
fi


if [[ $task == "asr" ]] && [[ -f $out_dir/$task.ctm ]] ; then
    echo "ERROR: output CTM already exists ... delete before running: $out_dir/$task.ctm"
    exit 100
fi

norm_dir=$out_dir/norm
if [ -f $norm_dir/$task.ctm ]; then
    echo "ERROR: normalised CTM already exists ... delete before running: $norm_dir/$task.ctm."
    exit 100
fi

# Create CMDs/ directory if it doesn't exist
mkdir -p CMDs/$out_dir

# Cache the command-line arguments
cmdfile=CMDs/$out_dir/run-asr-to-task-input.sh
echo "$ALLARGS" >> "$cmdfile"
echo "------------------------------------------------------------------------" >> "$cmdfile"

# source the environment
source $ENV

# Print the arguments to visually verify
echo "Input CTMs:       ${input_ctms}"
echo "File list:        ${flist}"
echo "Task:             ${task}"
echo "Output directory: ${out_dir}"

mkdir -p $out_dir
log_dir=$out_dir/LOGs
mkdir -p $log_dir

echo "Log directory: $log_dir"

### Step1: select files prior to normalisation
pre_norm_ctm=${norm_dir}/${task}.pre-norm.ctm
echo "Step1: Select files prior to normalisation: output to ${pre_norm_ctm}"
python $tools_dir/select_files.py \
       --input_ctm "${input_ctms}" \
       --wav_list $flist \
       --output_ctm $pre_norm_ctm 2>&1 | tee $log_dir/select_files.$task.log
    #    --output_ctm $pre_norm_ctm |& tee $log_dir/select_files.$task.log

if [ ! -f $pre_norm_ctm ]; then
    echo "ERROR: CTM not created: $pre_norm_ctm"
    exit 100
fi

### Step1a: add phrase markers for GEC
if [ $task == "gec" ]; then
    pre_norm_phrase_ctm=${norm_dir}/${task}.pre-norm.phrase.ctm
    echo "Step 1a: Add phrase markers for GEC: output to ${pre_norm_phrase_ctm}"
    python $tools_dir/add_phrase_markers.py \
           --input_ctm $pre_norm_ctm \
           --output_ctm $pre_norm_phrase_ctm 2>&1 | tee $log_dir/add_phrase_markers.$task.log
	#    --output_ctm $pre_norm_phrase_ctm |& tee $log_dir/add_phrase_markers.$task.log

    if [ ! -f ${pre_norm_phrase_ctm} ]; then
	echo "ERROR: CTM not created: ${pre_norm_phrase_ctm}"
	exit 100
    fi
fi

### Step2: normalise files
echo "Step 2: Normalise CTM: output to ${norm_dir}/$task.ctm"
if [ $task == "gec" ]; then
    input_ctm=$pre_norm_phrase_ctm
else
    input_ctm=$pre_norm_ctm
fi
		
python $tools_dir/ctm_norm.py \
       --ctm_file $input_ctm \
       --output_file ${norm_dir}/$task.ctm \
       --task $task 2>&1 | tee $log_dir/ctm_norm.$task.log
    #    --task $task |& tee $log_dir/ctm_norm.$task.log

if [ ! -f ${norm_dir}/$task.ctm ]; then
    echo "ERROR: CTM not created: ${norm_dir}/$task.ctm"
    exit 100
fi

### Step2a: copy ctm to level above norm directory
if [ $task == "asr" ]; then
    echo "Step 2: copying CTM to final location"
    \cp ${norm_dir}/asr.ctm ${out_dir}/asr.ctm
    if [ ! -f ${out_dir}/asr.ctm ]; then
	echo "Unable to copy ${norm_dir}/asr.ctm to ${out_dir}/asr.ctm"
	exit 100
    fi
fi

echo "Complete: CTM can be found in ${norm_dir}/$task.ctm"

