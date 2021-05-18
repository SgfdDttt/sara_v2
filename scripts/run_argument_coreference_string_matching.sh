source global_vars.sh

# check/make necessary data and folders
mkdir -p $PROCESSED_DATA
mkdir -p $EXP_DIR
for dir in $SARA $SARA2 $STATUTES $SPANS $BOUNDARIES; do
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

# STRING MATCHING COREFERENCE BASELINE
output_dir=$EXP_DIR/string_matching_coref
mkdir -p $output_dir
output_file=$EXP_DIR/string_matching_coref.conll
python code/coreference_baseline_string_matching.py --spans $SPANS \
    --statutes $STATUTES --output_dir $output_dir
python $code_dir/spans2conll_coref.py --boundaries $BOUNDARIES \
    --spans $output_dir --savefile $output_file
cp $output_file coref.tmp
python code/renumber_singleton_coref_clusters_conll.py coref.tmp $output_file
rm coref.tmp 

# SCORE BASELINE
python code/score_coref_conll.py $gold_coref_conll $output_file
for x in muc ceafm ceafe blanc
do
    echo $x
    perl $COREF_SCORER $x $gold_coref_conll $output_file | tail -3
done
