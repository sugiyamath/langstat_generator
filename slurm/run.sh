#!/bin/bash

#SBATCH -N 7
#SBATCH -n 7
#SBATCH -c 16
#SBATCH -t 08:00:00


sudo timedatectl set-timezone UTC
sudo timedatectl set-ntp true
sudo timedatectl set-local-rtc 0

PY_PATH=./venv/bin/python
OUT_DIR=./out
WET_PATH=./wet.paths.gz
BIN_DIR=./bin
MAX_SHARD_NUM=4
MAX_NODE_NUM=6

mkdir -p ${OUT_DIR}
mkdir -p ./cc_data

for i in $(seq 0 ${MAX_NODE_NUM}); do
    srun -N 1 -n 1 bash execute.sh ${WET_PATH} ${PY_PATH} ${i} ${MAX_SHARD_NUM} ${OUT_DIR} ${BIN_DIR} &
done
wait

