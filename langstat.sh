#!/bin/bash
TMP_PATH=$1
PY_PATH=$2
OUT_PATH=$3
NODE_ID=$4
SHARD_ID=$5

ls -1d ${TMP_PATH}/* | head -n 50 | ${PY_PATH} main.py ./bin ${OUT_PATH}/lm_scores_${NODE_ID}_${SHARD_ID}.txt ${OUT_PATH}/langstats_${NODE_ID}_${i}.txt ${TMP_PATH}
