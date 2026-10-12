#!/bin/bash
cd "$(dirname "$0")"
PY=~/cl_env/bin/python
OUT=../data/pilot
run() { $PY rev_run.py --out-dir $OUT --trials 2000 "$@" > logs/pilot_$1_$2.log 2>&1; }
run --name p_agn_purx --scheme shared --U 6 --agnostic --fading indep --epochs 60 --steps 100 --batch 64 --per-user-rx --eval-U 1 2 3 4 5 6 &
run --name p_agn      --scheme shared --U 6 --agnostic --fading indep --epochs 60 --steps 100 --batch 64 --eval-U 1 2 3 4 5 6 &
wait
echo PILOT4_DONE
