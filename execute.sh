#!/bin/bash

WET_PATH=$1
PY_PATH=$2
NODE_ID=$3
SHARD_ID=$4
OUT_PATH=$5
TMP_PATH=$6


zcat ${WET_PATH} | head -n 1120 | ${PY_PATH} sharding.py ${NODE_ID} ${SHARD_ID} | awk '{print "https://commoncrawl.s3.amazonaws.com/"$0}' | head -n 50 | xargs -n1 -P 50 wget {} -P ${TMP_PATH}; ls -1d ${TMP_PATH}/* | head -n 50 | ${PY_PATH} main.py ./bin ${OUT_PATH}/lm_scores_${NODE_ID}_${SHARD_ID}.txt ${OUT_PATH}/langstats_${NODE_ID}_${SHARD_ID}.txt ${TMP_PATH}

