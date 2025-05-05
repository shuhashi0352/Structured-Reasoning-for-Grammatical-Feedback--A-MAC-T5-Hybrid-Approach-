Implementation Instruction for Structured Reasoning for Grammatical Feedback -A MAC-T5 Hybrid Approach-

Shusuke Hashimoto

##########
NOTE1: Different models are required for running whole scripts. Specifically, the environment of the disfluency detection has to be different from all the other phases.
See the "envs" folder for specifics, you will see two different settings such as:

・sandi-dd
・sandi-all

You will need to change the environment at 2. GEC.


NOTE2: I took the disfluency detection and the gec model away from this repository (FOR_715) because large capacity kept me from compressing the folder to create a zip file.

My apology for the incomplete resources.
##########

----------

1. ASR

Run the Script (FOR_715/asr_gec/run-scriptrun-asr-to-task-input.sh) to get fluent.ctm

FOR_715/asr_gec/run-script/run-asr-to-task-input.sh \
  --input_ctms FOR_715/asr_gec/asr_gec_dataset/asr/decode/dev-subset-asr/asr.ctm \
  --flist FOR_715/asr_gec/asr_gec_dataset/flist/dev-gec.tsv \
  --task gec \
  --out_dir FOR_715/asr_gec/asr_gec_dataset/gec

-----

2. GEC

Run the Script (run-gec.sh) to get gec.ctm

NOTE: Make sure to change the environment into "sandi-dd"

FOR_715/asr_gec/run-script/run-gec.sh \
  --ctm FOR_715/asr_gec/asr_gec_dataset/gec/fluent.ctm \
  --dd_model FOR_715/sandi-models/dd/dd_model \
  --gec_model FOR_715/sandi-models/gec/gec_model \
  --out_dir FOR_715/asr_gec/asr_gec_dataset/gec


=> It will be stuck halfway through because of the environment compatibility.
・The disfluency detection model employs the older version of transformer, which is not compatible with the one used for the gec model)
・However, you will successfully get the output (FOR_715/asr_gec/asr_gec_dataset/gec/dd/pre-gec.tsv) from the disfluency detection model, and it will stop implementing before running gec-plus-align.py.


=> So, you have to manually run gec-plus-align.py to complete the whole script.

python FOR_715/asr_gec/gec_model/gec-plus-align.py \
  --input_file FOR_715/asr_gec/asr_gec_dataset/gec/dd/pre-gec.tsv \
  --gec_model FOR_715/sandi-models/gec/gec_model \
  --gec_ctm FOR_715/asr_gec/asr_gec_dataset/gec/gec.ctm

To get the output (gec.ctm)

-----

3. GECF Pre-processing

=> Implement create_files_for_errant.py to get four text files

python FOR_715/preprocess_for_fb/create_files_for_errant.py \
  --flt_stm FOR_715/preprocess_for_fb/gec_reference/dev-fluent.stm \
  --flt_ctm FOR_715/asr_gec/asr_gec_dataset/gec/fluent.ctm \
  --gec_stm FOR_715/preprocess_for_fb/gec_reference/dev-gec.stm \
  --gec_ctm FOR_715/asr_gec/asr_gec_dataset/gec/gec.ctm \
  --ref_src FOR_715/preprocess_for_fb/raw_txt/r-fluent.txt \
  --hyp_src FOR_715/preprocess_for_fb/raw_txt/p-fluent.txt \
  --ref_tgt FOR_715/preprocess_for_fb/raw_txt/r-gec.txt \
  --hyp_tgt FOR_715/preprocess_for_fb/raw_txt/p-gec.txt


=> Implement spoken_errant.py to get FOR_715/fb/fb_dataset/rest/spoken-gec-feedback.json

python FOR_715/preprocess_for_fb/spoken_errant.py \
  --ref_asr FOR_715/preprocess_for_fb/raw_txt/r-fluent.txt \
  --hyp_asr FOR_715/preprocess_for_fb/raw_txt/p-fluent.txt \
  --ref_gec FOR_715/preprocess_for_fb/raw_txt/r-gec.txt \
  --hyp_gec FOR_715/preprocess_for_fb/raw_txt/p-gec.txt


=> Run FOR_715/fb/fb_models/filter_json.py to get filtered json files (for train/dev/test)

・The design of this implementation depends on which dataset you need to receive.
(One-shot or two-shot makes the implementation methods different)

・Make sure to remove the filtered instances from the original data and save it (rest_)
(Otherwise, for example, you may get the same instances in both train and dev data)

See the actual python file for further details.

-----

4. Feedback Creation through Open AI API

=> Once you get all the train/dev/test datasets, run the API python files.
(The API key is required)

FOR_715/api/chain-of-thought-api.py -> for train data
FOR_715/api/ref-fb-api.py           -> for train/dev/test data

-----

5. GECF

=> After gaining all references and CoT feedback, run the two models.

FOR_715/fb/fb_models/t5.py             -> The transformer-based model
FOR_715/fb/fb_models/mac_t5_hybrid.py  -> The reasoning-based model

-----

6. Evaluation

=> After getting all the json outputs with generated feedback, evaluate the feedback.

FOR_715/eval/llm_eval/eval-api.py    -> LLM_based Evaluation (The API key is required)
FOR_715/eval/bertscore/bertscore.py  -> BERTScore Evaluation