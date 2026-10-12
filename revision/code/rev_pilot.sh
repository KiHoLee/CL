#!/bin/bash
# Pilot runs (40 epochs x 100 steps) to fix the fading model and the budget.
cd "$(dirname "$0")"
PY=~/cl_env/bin/python
OUT=../data/pilot
mkdir -p $OUT logs
run() { $PY rev_run.py --out-dir $OUT --epochs 40 --steps 100 --trials 2000 "$@" > logs/pilot_$1_$2.log 2>&1; }
# wave 1
run --name p_shared_U4_indep  --scheme shared --U 4 --fading indep --robust --probe &
run --name p_ortho_U4_indep   --scheme ortho  --U 4 --fading indep --robust --probe &
run --name p_shared_U6_indep  --scheme shared --U 6 --fading indep &
wait
# wave 2
run --name p_shared_U4_common --scheme shared --U 4 --fading common --robust &
run --name p_ortho_U4_common  --scheme ortho  --U 4 --fading common --robust &
run --name p_ortho_U6_indep   --scheme ortho  --U 6 --fading indep &
wait
echo PILOT_DONE
