# prepare data for and run argument identification with BERT-based CRF models
source global_vars.sh

# check/make necessary data and folders
mkdir -p $PROCESSED_DATA
mkdir -p $EXP_DIR
for dir in $SARA $SARA2 $STATUTES $SPANS $BOUNDARIES $STANFORD_PARSER; do
    if [[ ! -d "$dir" ]]; then
       echo "missing $dir"
       exit 0
    fi
done

exp_dir=$EXP_DIR/argument_identification_parser
mkdir -p $exp_dir
# run stanford parser on statutes
parser_output=$exp_dir/parser_output.txt
bash code/lexparser.sh $STATUTES/*  > $parser_output
# find spans
rspan_output=$exp_dir/rspan_output.txt
python code/rspan.py $parser_output $rspan_output
# postprocess
python code/postprocess_rspans.py --statutes $STATUTES \
    --input $rspan_output --output_dir $exp_dir/rspans
# score
python code/score_spans.py --spans $SPANS --boundaries $BOUNDARIES \
    --candidates $exp_dir/rspans
