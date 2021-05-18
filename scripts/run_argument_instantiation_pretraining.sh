source global_vars.sh

# check/make necessary data and folders
mkdir -p $PROCESSED_DATA
mkdir -p $EXP_DIR
for dir in $SARA $SARA2 $STATUTES $SPANS $BOUNDARIES $SPLITS $STRUCTURE $LEGAL_BERT; do
    if [[ ! -d "$dir" ]]; then
       echo "missing $dir"
       exit 0
    fi
done

for filename in $ARGUMENT_INSTANTIATION $SILVER_ARGUMENT_INSTANTIATION; do
    if [[ ! -f $filename ]]; then
        echo "missing $filename"
        exit 0
    fi
done


# CONSTANTS AND DATA
python code/argument_instantiation_prepare_data.py \
    --output_file $PREPROCESSED_DATA/argument_instantiation_data.json \
    --statutes $STATUTES --cases $CASES --splits $SPLITS --spans $SPANS \
    --boundaries $BOUNDARIES --structure $STRUCTURE \
    --gold_argument_instantiation $ARGUMENT_INSTANTIATION \
    --silver_argument_instantiation $SILVER_ARGUMENT_INSTANTIATION

exp_dir=$EXP_DIR/argument_instantiation_pretraining
mkdir -p $exp_dir
# PRETRAINING
echo STAGE 0 training
mkdir -p $exp_dir/stage_0
CUDA_VISIBLE_DEVICES=`GPU` python code/train_argument_instantiation.py \
    --datafile $PREPROCESSED_DATA/argument_instantiation_data.json --expdir $exp_dir/stage_0 \
    --training_stage 0 --max_epochs 10 --patience 5 --epoch_size 50000 \
    --gmm_model $exp_dir/cluster_model.pkl --thaw_top_layer --weight_decay 0 \
    --update_period 128 --bert_model $LEGAL_BERT --learning_rate 1e-05
echo STAGE 1 training
mkdir -p $exp_dir/stage_1
cp $exp_dir/stage_0/best_model.pt $exp_dir/stage_1/checkpoint.pt
CUDA_VISIBLE_DEVICES=`GPU` python code/train_argument_instantiation.py \
    --datafile $PREPROCESSED_DATA/argument_instantiation_data.json --expdir $exp_dir/stage_1 \
    --training_stage 1 --max_epochs 10 --patience 5 --epoch_size 50000 \
    --gmm_model $exp_dir/cluster_model.pkl --thaw_top_layer --weight_decay 0 \
    --update_period 128 --bert_model $LEGAL_BERT --learning_rate 1e-05
echo STAGE 2 training
mkdir -p $exp_dir/stage_2
cp $exp_dir/stage_1/best_model.pt $exp_dir/stage_2/checkpoint.pt
CUDA_VISIBLE_DEVICES=`GPU` python code/train_argument_instantiation.py \
    --datafile $PREPROCESSED_DATA/argument_instantiation_data.json --expdir $exp_dir/stage_2 \
    --training_stage 2 --max_epochs 8 --patience 4 --epoch_size 10000 \
    --gmm_model $exp_dir/cluster_model.pkl --weight_decay 0 --update_period 128 \
    --bert_model $LEGAL_BERT --learning_rate 1e-05 --max_depth 3
