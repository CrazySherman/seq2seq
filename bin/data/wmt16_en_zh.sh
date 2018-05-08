#! /usr/bin/env bash

# Author: Sherman Wong
# Description: bashcript to preprocess training data from chinse and english
: ${SRC_DIR?"Need to set SRC_DIR"}
set -e

#export SRC_DIR=
#export SRC_LANG=
#export TGT_LANG=
SRC_LANG=${SRC_LANG:-en}
TGT_LANG=${TGT_LANG:-zh}
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

OUTPUT_DIR=${OUTPUT_DIR:-/tmp/wmt16_en_zh}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."
echo "cleaning up all the existing corpus in the output directory..."
rm $OUTPUT_DIR/*.$SRC_LANG $OUTPUT_DIR/*.$TGT_LANG  || true

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

mkdir -p $OUTPUT_DIR_DATA

echo 'copying source corpus...'
cp $SRC_DIR/*  $OUTPUT_DIR/

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

########################## sort of standardized corpus processing pipeline ####################
# Tokenize data and cat all of them into a single train file
for f in ${OUTPUT_DIR}/*.$SRC_LANG; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l $SRC_LANG -threads 8 < $f >> $OUTPUT_DIR/train.tok.$SRC_LANG
done

for f in ${OUTPUT_DIR}/*.$TGT_LANG; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l $TGT_LANG -threads 8 < $f >> $OUTPUT_DIR/train.tok.$TGT_LANG
done

# Clean all corpora

echo "Cleaning train data..."
${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl "$OUTPUT_DIR/train.tok" $SRC_LANG $TGT_LANG "$OUTPUT_DIR/train.tok.clean" 1 80

# Create character vocabulary
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.tok.clean.$SRC_LANG \
  > ${OUTPUT_DIR}/vocab.tok.char.$SRC_LANG
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.tok.clean.$TGT_LANG \
  > ${OUTPUT_DIR}/vocab.tok.char.$TGT_LANG


# Create vocabulary data
$BASE_DIR/bin/tools/generate_vocab.py \
   --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.tok.clean.$SRC_LANG \
  > ${OUTPUT_DIR}/vocab.50k.$SRC_LANG \

$BASE_DIR/bin/tools/generate_vocab.py \
  --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.tok.clean.$TGT_LANG \
  > ${OUTPUT_DIR}/vocab.50k.$TGT_LANG \

# Generate Subword Units (BPE)
# Clone Subword NMT
if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
fi

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.$SRC_LANG" "${OUTPUT_DIR}/train.tok.clean.$TGT_LANG" | \
    ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in $SRC_LANG $TGT_LANG; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.$SRC_LANG" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.$TGT_LANG" | \
    ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

  # Generate dev set by splitting the train file
  $BASE_DIR/bin/data/generate_devset.sh ${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}
done

echo "All done."
