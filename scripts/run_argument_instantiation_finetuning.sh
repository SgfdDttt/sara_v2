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

PRETRAINED_MODEL_FOLDER=/export/c12/nholzen/sara_v2/exp/argument_instantiation_pretraining # TODO provide a pointer to the folder containing the files of your pretrained model
PRETRAINED_MODEL=$PRETRAINED_MODEL_FOLDER/stage_2/best_model.pt # TODO provide a pointer to the folder containing the files of your pretrained model
PRETRAINED_CLUSTER_MODEL=$PRETRAINED_MODEL_FOLDER/cluster_model.pkl # TODO provide a pointer to the folder containing the files of your pretrained model
if [[ ! -d $PRETRAINED_MODEL_FOLDER ]]; then
    echo "missing a pretrained model folder or does not point to a directory"
    exit 0
fi
if [[ ! -f $PRETRAINED_MODEL ]]; then
    echo "missing a pretrained model or does not point to a file"
    exit 0
fi
if [[ ! -f $PRETRAINED_CLUSTER_MODEL ]]; then
    echo "missing a pretrained cluster model or does not point to a file"
    exit 0
fi

# CONSTANTS AND DATA
arg_inst_datafile=$PROCESSED_DATA/argument_instantiation_data.json
[ -f $arg_inst_datafile ] || python code/argument_instantiation_prepare_data.py \
    --savefile $PROCESSED_DATA/argument_instantiation_data.json \
    --statutes $STATUTES --cases $CASES --splits $SPLITS --spans $SPANS \
    --boundaries $BOUNDARIES --structure $STRUCTURE \
    --gold_argument_instantiation $ARGUMENT_INSTANTIATION \
    --silver_argument_instantiation $SILVER_ARGUMENT_INSTANTIATION

arg_inst_splits_datafile=$PROCESSED_DATA/argument_instantiation_data_splits.json
[ -f $arg_inst_splits_datafile ] || python code/argument_instantiation_split_data.py \
    $arg_inst_datafile $arg_inst_splits_datafile

# In the experiments below, we're only providing a single experiment, with a single set of hyperparameters. The paper
# specifies the full range of parameters tried. It is up to you to modify this script to run the full grid search.
# In addition, you will probably want to parallelize the for loop going over all the splits.

# FINETUNING
exp_dir=$EXP_DIR/argument_instantiation_finetuning
mkdir -p $exp_dir
for spliti in 0 1 2 3 4 5 6 7 8 9; do
    exp_dir2=$exp_dir/split_$spliti
    mkdir -p $exp_dir2
    cp $PRETRAINED_MODEL $exp_dir2/checkpoint.pt || exit 0
    CUDA_VISIBLE_DEVICES=`free-gpu` python code/train_argument_instantiation.py \
        --datafile $arg_inst_splits_datafile --gmm_model_file $PRETRAINED_MODEL/cluster_model.pkl \
        --expdir $exp_dir2 --training_stage 2 --max_epochs 2 --patience 10 --max_depth 3 \
        --weight_decay 0 --learning_rate 0.01 --update_period 128 --batch 4 \
        --split_index $spliti || exit 0
done

# FINETUNING - ABLATIONS
# ablation experiments

# -STRUCTURE
exp_dir=$EXP_DIR/argument_instantiation_finetuning_no_structure
mkdir -p $exp_dir
for spliti in 0 1 2 3 4 5 6 7 8 9; do
    exp_dir2=$exp_dir/split_$spliti
    mkdir -p $exp_dir2
    cp $PRETRAINED_MODEL $exp_dir2/checkpoint.pt || exit 0
    CUDA_VISIBLE_DEVICES=`free-gpu` python code/train_argument_instantiation.py \
        --datafile $arg_inst_splits_datafile --gmm_model_file $PRETRAINED_MODEL/cluster_model.pkl \
        --expdir $exp_dir2 --training_stage 1 --max_epochs 2 --patience 20 \
        --weight_decay 0 --split_index $spliti --update_period 256 \
        --learning_rate 0.01 --batch 4 || exit 0
done

# -PRETRAIN
exp_dir=$EXP_DIR/argument_instantiation_finetuning_no_pretrain
mkdir -p $exp_dir
for spliti in 0 1 2 3 4 5 6 7 8 9; do
    exp_dir2=$exp_dir/split_$spliti
    mkdir -p $exp_dir2
    CUDA_VISIBLE_DEVICES=`free-gpu` python code/train_argument_instantiation.py \
        --datafile $arg_inst_splits_datafile --gmm_model_file $exp_dir2/cluster_model.pkl \
        --expdir $exp_dir2 --training_stage 2 --max_epochs 2 --patience 10 \
        --weight_decay 0 --split_index $spliti --batch 4 --max_depth 3 \
        --update_period 64 --learning_rate 0.01 || exit 0
done

# -STRUCTURE, -PRETRAIN
exp_dir=$EXP_DIR/argument_instantiation_finetuning_no_structure_no_pretrain
mkdir -p $exp_dir
for spliti in 0 1 2 3 4 5 6 7 8 9; do
    exp_dir2=$exp_dir/split_$spliti
    mkdir -p $exp_dir2
    CUDA_VISIBLE_DEVICES=`free-gpu` python code/train_argument_instantiation.py \
        --datafile $arg_inst_splits_datafile --gmm_model_file $exp_dir2/cluster_model.pkl \
        --expdir $exp_dir2 --training_stage 1 --max_epochs 2 --patience 20 --weight_decay 0 \
        --split_index $spliti --batch 4 --learning_rate 0.01 --update_period 128 || exit 0
done
