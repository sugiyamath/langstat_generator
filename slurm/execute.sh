#!/bin/bash

WET_PATH=$1
PY_PATH=$2
NODE_ID=$3
MAX_SHARD_NUM=$4
OUT_DIR=$5
BIN_DIR=$6

sudo timedatectl set-timezone UTC
sudo timedatectl set-ntp true
sudo timedatectl set-local-rtc 0

rm -r /tmp/tmp*
sudo umount -l ./cc_data
sudo umount -l ./langstat
./goofys commoncrawl ./cc_data
./goofys langstat ./langstat
#zcat ${WET_PATH} | head -n 1120 | awk '{print "https://commoncrawl.s3.amazonaws.com/"$0}' | ${PY_PATH} main.py ${NODE_ID} ${MAX_SHARD_NUM} ${BIN_DIR} ${OUT_DIR}
zcat ${WET_PATH} | head -n 1120 | awk '{print "./cc_data/"$0}' | ${PY_PATH} main.py ${NODE_ID} ${MAX_SHARD_NUM} ${BIN_DIR} ${OUT_DIR} | ${PY_PATH} scripts/timer_out.py

