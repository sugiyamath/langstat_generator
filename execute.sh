#!/bin/bash

WET_PATH=$1
PY_PATH=$2
NODE_ID=$3
SHARD_ID=$4
OUT_PATH=$5
TMP_PATH=$6

bash download.sh ${WET_PATH} ${PY_PATH} ${NODE_ID} ${SHARD_ID} ${TMP_PATH}
bash langstat.sh ${TMP_PATH} ${PY_PATH} ${OUT_PATH} ${NODE_ID} ${SHARD_ID}


