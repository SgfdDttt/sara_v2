source global_vars.sh

exp_dir=$1
bert_model=$2
thaw_top_layer=$3

for split in 1 2 3 4 5 6 7
do
    echo "SPLIT "$split
    if [ $split -eq 1 ]
    then
        dev_sections="152" 
        test_sections="3306" 
    elif [ $split -eq 2 ]
    then
        dev_sections="3306" 
        test_sections="1" 
    elif [ $split -eq 3 ]
    then
        dev_sections="1" 
        test_sections="2" 
    elif [ $split -eq 4 ]
    then
        dev_sections="2" 
        test_sections="63" 
    elif [ $split -eq 5 ]
    then
        dev_sections="63" 
        test_sections="68 3301 7703" 
    elif [ $split -eq 6 ]
    then
        dev_sections="68 3301 7703" 
        test_sections="151" 
    elif [ $split -eq 7 ]
    then
        dev_sections="151" 
        test_sections="152" 
    fi
    split_dir=$exp_dir/split_$split
    mkdir -p $split_dir
    if [ $thaw_top_layer -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=`free-gpu` python code/argument_identification_train_bert_crf.py \
            --datafile $PROCESSED_DATA/argument_identification_bert_data.json --dev $dev_sections \
            --test $test_sections --expdir $split_dir --bert_model $bert_model --thaw_top_layer || exit 0
    else
        CUDA_VISIBLE_DEVICES=`free-gpu` python code/argument_identification_train_bert_crf.py \
            --datafile $PROCESSED_DATA/argument_identification_bert_data.json --dev $dev_sections \
            --test $test_sections --expdir $split_dir --bert_model $bert_model || exit 0
    fi
done
python code/aggregate_argument_identification_bert_crf_results.py $exp_dir
