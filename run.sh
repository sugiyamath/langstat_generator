#!/bin/bash

#SBATCH -N 7
#SBATCH -n 7
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
	srun -N 1 -n 1 bash execute.sh ${WET_PATH} ${PY_PATH} ${i} ${j} ${OUT_PATH} ${TMP_PATH} &
    done
    wait
    rm ${TMP_PREFIX}* -r
done
