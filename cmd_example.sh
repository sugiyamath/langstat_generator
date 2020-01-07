#!/bin/bash
zcat wet.paths.gz | head -n 1120 | python3 main.py 0 1 bin/ out > log
sort -t$'\t' -k1,1 -k2,2 -u out/langstat.0_0.txt | python3 fix_langstat.py > langstat.0_0.txt.fixed
