# SARA v2
This repository contains the code associated with the [SARA v2 paper](https://aclanthology.org/2021.acl-long.213.pdf).

# Data
The data is hosted on [NLLP@JHU](https://nlp.jhu.edu/law/). Look for "SARA v2".

# Contents
The folder `scripts` contains all you need to reproduce the experiments from the paper. The scripts assume that you have completed the requirements below, and will both prepare the data and run the experiments.

# Requirements
To run the scripts in this repository, you will need a number of other pieces of code. Either go through the instructions below, or run `bash scripts/get_dependencies.sh`.

## General
The code in this repository relies on bash and Python 3.5.3. Install all the needed packages into a dedicated virtual environment with `pip install -r requirements.txt`

## Data
Download SARA and SARAv2 and place them in this repository under `sara` and `sara_v2` respectively; or run `bash scripts/get_data.sh`.

## Coreference scorer
The scorer used for conventional coreference metrics can be found in [this Github repository](https://github.com/conll/reference-coreference-scorers). Clone the latter repository from within your clone of the SARA v2 repository.

## LegalBert
Download Legal Bert from the website of [Tax Law NLP Resources](https://archive.data.jhu.edu/dataset.xhtml?persistentId=doi:10.7281/T1/N1X6I4) or directly from [here](https://archive.data.jhu.edu/file.xhtml?persistentId=doi:10.7281/T1/N1X6I4/8NZ3AD&version=2.0), and unzip the file into this repository, into a folder named `LegalBert`, eg with the command `unzip LegalBert.zip -d LegalBert`.

## Stanford parser
This repository uses [Version 3.9.2](https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip) of the parser, timestamped 2018-10-17. Download the parser using the preceding link and unzip it into this repository.

## Configuration

Some of these scripts rely on GPUs. Since there is no universal configuration for these devices, it is up to you to modify the scripts in the right places. For that, grep for 'GPU' in `scripts` and in `code`.

Some scripts will need to download models from [huggingface](https://huggingface.co/), which requires an internet connection.

## Argument identification with BERT-based CRF

Specifically for the scripts running argument identification with a BERT-based CRF, you need Python 3.7.9. and Allennlp and Huggingface. Those requireents are captured in `requirements_crf.txt`. The requirements for the rest of the codebase are not compatible with `requirements_crf.txt`, so you need a separate environment.
