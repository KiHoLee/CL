#!/bin/bash
# Pilot wave 3: budget and optimizer sensitivity of the shared block.
cd "$(dirname "$0")"
PY=~/cl_env/bin/python
OUT=../data/pilot
run() { $PY rev_run.py --out-dir $OUT --trials 2000 "$@" > logs/pilot_$1_$2.log 2>&1; }
run --name p_shared_U4_long   --scheme shared --U 4 --fading indep --epochs 150 --steps 100 &
run --name p_shared_U4_lr2e3  --scheme shared --U 4 --fading indep --epochs 60 --steps 100 --beta 2e-3 &
run --name p_shared_U4_b64    --scheme shared --U 4 --fading indep --epochs 60 --steps 100 --batch 64 &
wait
echo PILOT2_DONE
