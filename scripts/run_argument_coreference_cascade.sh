source global_vars.sh

# check/make necessary data and folders
mkdir -p $PROCESSED_DATA
mkdir -p $EXP_DIR

# you have to have run scripts/run_argument_identification_crf.sh first
argument_identification_exp_dir=$EXP_DIR"/"$LEGAL_BERT"_1"

for dir in $SARA $SARA2 $STATUTES $SPANS $BOUNDARIES $argument_identification_exp_dir; do
    if [[ ! -d "$dir" ]]; then
       echo "missing $dir"
       exit 0
    fi
done
if [[ ! -f $COREF_SCORER ]]; then
	echo "missing $COREF_SCORER"
	exit 0
fi

# CONSTANTS AND DATA
gold_coref_conll=$PROCESSED_DATA/coref.conll
# make sure the data is up to date
python code/spans2conll_coref.py --boundaries $BOUNDARIES \
	--spans $SPANS --savefile $gold_coref_conll
cp $gold_coref_conll coref.tmp
python code/renumber_singleton_coref_clusters_conll.py coref.tmp $gold_coref_conll
rm coref.tmp

# SINGLE MENTION COREFERENCE BASELINE
output_file=$EXP_DIR/cascade_coref.json
python code/coreference_cascade.py $argument_identification_exp_dir $output_file

# SCORE BASELINE
python code/score_coref_json.py $output_file
