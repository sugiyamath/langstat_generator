#!/bin/bash

#SBATCH -N 7
#SBATCH -n 7
#SBATCH -c 16
#SBATCH -t 08:00:00

for i in $(seq 0 6); do
    srun -N 1 -n 1 time bash execute.sh ./wet.paths.gz ${i} 3 ./out ./tmp &
done

wait
