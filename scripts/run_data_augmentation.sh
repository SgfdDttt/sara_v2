#!/bin/bash
source global_vars.sh

# check/make necessary data and folders
mkdir -p $PROCESSED_DATA
mkdir -p $EXP_DIR
for dir in $SARA $PROLOG $CASES; do
    if [[ ! -d "$dir" ]]; then
       echo "missing $dir"
       exit 0
    fi
done

# each call to code/run_predicate_and_case.py can be very slow
# you will want to parallelize these nested loops
num_cases=`ls -lh $CASES/ | wc -l`
num_cases=$((num_cases-1))
start_case=0
end_case=$num_cases
num_predicates=`grep '^s' $PROLOG/section* | wc -l`
num_predicates=$((num_predicates+1)) # account for special 'tax' predicate
for ((casei=$start_case; casei<$end_case; casei++))
do
    for ((predi=0; predi<$num_predicates; predi++))
    do
        timeout 10h python code/run_predicate_and_case.py \
            --cases $CASES --prolog $PROLOG \
            --case_id $casei --predicate_id $predi \
            --tmp_file "$casei-$predi" || exit 0
    done
done | grep '^\(A\|B\|C\)' | cut -f2- > $SILVER_ARGUMENT_INSTANTIATION
