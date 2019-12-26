#!/bin/bash
WET_PATH=$1
PY_PATH=$2
NODE_ID=$3
SHARD_ID=$4
TMP_PATH=$5

zcat ${WET_PATH} | head -n 1120 | ${PY_PATH} sharding.py ${NODE_ID} ${SHARD_ID} | awk '{print "https://commoncrawl.s3.amazonaws.com/"$0}' | head -n 50 | ./parallel -j 50 wget {} -P ${TMP_PATH}
