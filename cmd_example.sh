#!/bin/bash
ls -1d data_dl/* | python3 dedup.py > langstat.0.txt
sort -t$'\t' -k1,1 -k2,2 -u langstat.0.txt | python3 fix_langstat.py > langstat.0.txt.fixed
