#!/usr/bin/env bash
#
# Runs the English PCFG parser on one or more files, printing trees only

if [ ! $# -ge 1 ]; then
  echo Usage: `basename $0` 'file(s)'
  echo
  exit
fi

scriptdir=`dirname $0`
scriptdir="$scriptdir/../stanford-parser-full-2018-10-17"

java -mx32g -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
 -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $*
