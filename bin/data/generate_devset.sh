#!/usr/bin/env bash
set -e

if [ $# -ne 1 ] ; then
    echo 'usage : generate_devset train_prefix'
fi
TRAIN_PREFIX=$1
BASE_DIR=$(dirname $TRAIN_PREFIX)

OUTPUT_DIR=$BASE_DIR/data_nnready/
mkdir -p $OUTPUT_DIR
echo "generating output directory for train test split: $OUTPUT_DIR"

echo "train-dev split:: 10 : 1"
for f in $TRAIN_PREFIX.*; do
    outfile_train=train.${f##*.}
    outfile_dev=dev.${f##*.}
    let "a=$(wc -l $f  | awk '{print $1}') / 10"
    let "b = $(wc -l $f  | awk '{print $1}') - a"
    echo "train vs dev:    $b  :  $a"
    head -n $a $f > $OUTPUT_DIR/$outfile_dev
    tail -n $b $f > $OUTPUT_DIR/$outfile_train

done