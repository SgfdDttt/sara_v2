# prepare data for and run argument identification with BERT-based CRF models
source global_vars.sh

# check/make necessary data and folders
mkdir -p $PROCESSED_DATA
mkdir -p $EXP_DIR
for dir in $SARA $SARA2 $STATUTES $SPANS $BOUNDARIES $LEGAL_BERT; do
    if [[ ! -d "$dir" ]]; then
       echo "missing $dir"
       exit 0
    fi
done

# prepare data for BERT-based CRF
python code/argument_identification_prepare_data.py \
    --statutes $STATUTES \
    --spans $SPANS \
    --boundaries $BOUNDARIES \
    --savefile $PROCESSED_DATA/bert_crf_data.json

# run exps with {Legal BERT, ordinary BERT} and {thaw top layer, freeze top layer}
for bert_model in "bert-base-cased" $LEGAL_BERT; do
    for thaw_top_layer in 0 1; do
        exp_dir=$EXP_DIR"/"$bert_model"_"$thaw_top_layer
        bash code/run_argument_identification.sh $exp_dir \
            $bert_model $thaw_top_layer
    done
done
