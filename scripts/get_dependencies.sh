pip install -r requirements.txt || exit 0
bash scripts/get_data.sh || exit 0
git clone https://github.com/conll/reference-coreference-scorers.git || exit 0
wget https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip || exit 0
unzip stanford-parser-full-2018-10-17.zip || exit 0
echo "Download Legal Bert and unzip the file into this repository (see README.md)."
