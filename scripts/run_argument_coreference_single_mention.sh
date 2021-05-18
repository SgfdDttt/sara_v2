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
root_dir=/export/c12/nholzen/tax_law
gold_coref_conll=$PROCESSED_DATA/coref.conll
# make sure the data is up to date
python code/spans2conll_coref.py --boundaries $BOUNDARIES \
	--spans $SPANS --savefile $gold_coref_conll
cp $gold_coref_conll coref.tmp
python code/renumber_singleton_coref_clusters_conll.py coref.tmp $gold_coref_conll
rm coref.tmp

# SINGLE MENTION COREFERENCE BASELINE
output_dir=$EXP_DIR/single_mention_coref
mkdir -p $output_dir
output_file=$EXP_DIR/single_mention_coref.conll
python code/coreference_baseline_single_mention.py --spans $SPANS \
	--statutes $STATUTES --output_dir $output_dir
python code/spans2conll_coref.py --boundaries $BOUNDARIES \
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
