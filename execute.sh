#!/bin/bash

WET_PATH=$1
NODE_ID=$2
SHARD_NUM=$3
OUT_PATH=$4
TMP_PATH=$5

mkdir -p ${OUT_PATH}

for i in $(seq 0 ${SHARD_NUM}); do
    mkdir -p ${TMP_PATH}
    zcat ${WET_PATH} | python3 sharding.py ${NODE_ID} ${i} | awk '{print "https://commoncrawl.s3.amazonaws.com/"$0}' | head -n 50 | parallel -j 50 wget {} -P ${TMP_PATH}
    ls -1d ${TMP_PATH}/* | head -n 50 | python3 main.py ./bin ${OUT_PATH}/lm_scores_${NODE_ID}_${i}.txt ${OUT_PATH}/langstats_${NODE_ID}_${i}.txt
    rm -r ${TMP_PATH}
    