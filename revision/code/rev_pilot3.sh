#!/bin/bash
cd "$(dirname "$0")"
PY=~/cl_env/bin/python
OUT=../data/pilot
run() { $PY rev_run.py --out-dir $OUT --trials 2000 "$@" > logs/pilot_$1_$2.log 2>&1; }
run --name p_shared_U4_purx --scheme shared --U 4 --fading indep --epochs 60 --steps 100 --per-user-rx --robust &
run --name p_shared_U6_purx --scheme shared --U 6 --fading indep --epochs 60 --steps 100 --per-user-rx &
wait
echo PILOT3_DONE
