source global_vars.sh

# check/make necessary data and folders
mkdir -p $PROCESSED_DATA
mkdir -p $EXP_DIR
for dir in $SARA $SARA2 $STATUTES $SPANS $BOUNDARIES $SPLITS $STRUCTURE; do
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

# BASELINE
code/argument_instantiation_baseline.py \
    $PREPROCESSED_DATA/argument_instantiation_data.json
