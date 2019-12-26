#!/bin/bash

#SBATCH -N 7
#SBATCH -n 28
#SBATCH -c 16
#SBATCH -t 08:00:00

TMP_PREFIX=./tmp
PY_PATH=./venv/bin/python
OUT_PATH=./out
WET_PATH=./wet.paths.gz

for j in $(seq 0 3); do
    for i in $(seq 0 6); do
	TMP_PATH=${TMP_PREFIX}_${i}_${j}
	mkdir -p ${TMP_PATH}
	srun -N 1 -n 1 bash download.sh ${WET_PATH} ${PY_PATH} ${i} ${j} ${TMP_PATH} &
    done
    wait
    for i in $(seq 0 6); do
	TMP_PATH=${TMP_PREFIX}_${i}_${j}
	srun -N 1 -n 1 bash langstat.sh ${TMP_PATH} ${PY_PATH} ${OUT_PATH} ${i} ${j} &
    done
    wait
    rm ${TMP_PREFIX}* -r
done

wait
