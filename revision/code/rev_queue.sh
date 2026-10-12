#!/bin/bash
# =========================================================
# rev_queue.sh — every training run of the revision, grouped by the
# figure or table it serves. Runs N_PAR jobs in parallel on this machine.
#   bash rev_queue.sh main      # Fig. 2/3, Table: shared U=1..6, ortho U=1..6, agnostic (3 seeds)
#   bash rev_queue.sh decomp    # gain decomposition at U=4 (and K=1, U=6)
#   bash rev_queue.sh ablate    # K sweep, MAML configuration, training SNR range
#   bash rev_queue.sh general   # SST-2, DistilBERT
# =========================================================
cd "$(dirname "$0")"
# local Mac: ~/cl_env; AINS: PY=~/tr_env/bin/python PYTHONPATH=~/sse_rev/pylib
PY=${PY:-~/cl_env/bin/python}
OUT=${OUT:-../data}
EP=${EP:-150}; ST=${ST:-100}; TR=${TR:-2000}; BS=${BS:-64}
EXTRA=${EXTRA:-}       # e.g. "--per-user-rx" once the receiver design is fixed
N_PAR=${N_PAR:-3}
mkdir -p $OUT logs
JOBS=()
add() { JOBS+=("$*"); }
runall() {
  for j in "${JOBS[@]}"; do
    name=$(echo "$j" | sed -E 's/.*--name ([^ ]+).*/\1/'); seed=$(echo "$j" | sed -E 's/.*--seed ([0-9]+).*/\1/')
    if [ -f "$OUT/${name}_s${seed}.json" ]; then echo "skip $name s$seed"; continue; fi
    # atomic lock so that several queue instances on one machine never run the same job
    mkdir -p "$OUT/locks"; if ! mkdir "$OUT/locks/${name}_s${seed}" 2>/dev/null; then echo "locked $name s$seed"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 20; done
    echo "start $name s$seed $(date +%H:%M)"
    ex=$EXTRA; case "$j" in *"--scheme ortho"*|*"--no-per-user-rx"*) ex="";; esac
    j=${j//--no-per-user-rx/}
    $PY rev_run.py --out-dir $OUT --epochs $EP --steps $ST --trials $TR --batch $BS $ex $j > "logs/${name}_s${seed}.log" 2>&1 &
    sleep 5
  done
  wait
}
add_decomp_a() {
  for s in 1 2 3; do add --name shared_U4_randmask --scheme shared --U 4 --K 4 --seed $s --freeze mask --probe; done
  add --name shared_U4_randtx   --scheme shared --U 4 --K 4 --seed 1 --freeze tx
  add --name shared_U4_randboth --scheme shared --U 4 --K 4 --seed 1 --freeze mask,tx
  add --name shared_U4_noattn   --scheme shared --U 4 --K 4 --seed 1 --no-attention
}
add_decomp_b() {
  add --name shared_U4_sharedrx --scheme shared --U 4 --K 4 --seed 1 --no-per-user-rx
  add --name shared_U6_sharedrx --scheme shared --U 6 --K 4 --seed 1 --no-per-user-rx
  add --name shared_U4_K1_randmask --scheme shared --U 4 --K 1 --seed 1 --freeze mask
  add --name shared_U4_K1_noattn   --scheme shared --U 4 --K 1 --seed 1 --no-attention
  add --name shared_U6_randmask --scheme shared --U 6 --K 4 --seed 1 --freeze mask
  add --name shared_U6_noattn   --scheme shared --U 6 --K 4 --seed 1 --no-attention
}
case "$1" in
main)      # proposal, per-U models and the one-model-for-all-U variant (3 seeds)
  for s in 1 2 3; do
    for U in 1 2 3 4 5 6; do
      extra=""; [ $U -eq 4 ] && extra="--probe --robust"; [ $U -eq 6 ] && extra="--probe --robust"
      add --name shared_U${U} --scheme shared --U $U --K 4 --seed $s $extra
    done
    add --name shared_agn --scheme shared --U 6 --K 4 --agnostic --seed $s --eval-U 1 2 3 4 5 6 --probe
  done ;;
main_ortho)   # orthogonal dimension sharing (3 seeds)
  for s in 1 2 3; do
    for U in 1 2 3 4 5 6; do
      extra=""; [ $U -eq 4 ] && extra="--probe --robust"; [ $U -eq 6 ] && extra="--probe --robust"
      add --name ortho_U${U}  --scheme ortho  --U $U --K 4 --seed $s $extra
    done
  done ;;
scale)        # scalability in U beyond the design point (seed 1)
  for U in 8 12 16; do
    add --name shared_U${U} --scheme shared --U $U --K 4 --seed 1
    add --name ortho_U${U}  --scheme ortho  --U $U --K 4 --seed 1
  done ;;
decomp)   add_decomp_a; add_decomp_b ;;
decomp_a) add_decomp_a ;;   # gain decomposition, part A (AINS)
decomp_b) add_decomp_b ;;   # gain decomposition, part B (Mac)
ortho_s3)     # orthogonal dimension sharing, seed 3 only (Mac)
  for U in 1 2 3 4 5 6; do
    extra=""; [ $U -eq 4 ] && extra="--probe --robust"; [ $U -eq 6 ] && extra="--probe --robust"
    add --name ortho_U${U}  --scheme ortho  --U $U --K 4 --seed 3 $extra
  done ;;
ablate_b)     # ablation remainder (Mac)
  add --name shared_U4_joint --scheme shared --U 4 --K 4 --seed 1 --mode joint --adapt-steps 0
  add --name shared_U4_trlow  --scheme shared --U 4 --K 4 --seed 1 --train-snrs 0 5 10
  add --name shared_U4_trhigh --scheme shared --U 4 --K 4 --seed 1 --train-snrs 15 20 25
  add --name shared_U4_common --scheme shared --U 4 --K 4 --seed 1 --fading common --robust
  add --name ortho_U4_common  --scheme ortho  --U 4 --K 4 --seed 1 --fading common --robust ;;
ablate)
  for K in 1 2 8; do add --name shared_U4_K${K} --scheme shared --U 4 --K $K --seed 1; done
  add --name shared_U4_alpha1e-4 --scheme shared --U 4 --K 4 --seed 1 --alpha 1e-4
  add --name shared_U4_alpha2e-3 --scheme shared --U 4 --K 4 --seed 1 --alpha 2e-3
  add --name shared_U4_mb2  --scheme shared --U 4 --K 4 --seed 1 --meta-batch 2
  add --name shared_U4_mb6  --scheme shared --U 4 --K 4 --seed 1 --meta-batch 6
  add --name shared_U4_inner3 --scheme shared --U 4 --K 4 --seed 1 --inner-steps 3
  add --name shared_U4_joint --scheme shared --U 4 --K 4 --seed 1 --mode joint --adapt-steps 0
  add --name shared_U4_trlow  --scheme shared --U 4 --K 4 --seed 1 --train-snrs 0 5 10
  add --name shared_U4_trhigh --scheme shared --U 4 --K 4 --seed 1 --train-snrs 15 20 25
  add --name shared_U4_common --scheme shared --U 4 --K 4 --seed 1 --fading common --robust
  add --name ortho_U4_common  --scheme ortho  --U 4 --K 4 --seed 1 --fading common --robust ;;
general)
  for U in 4 6; do
    add --name sst2_shared_U${U} --scheme shared --U $U --K 4 --seed 1 --dataset sst2 --probe
    add --name sst2_ortho_U${U}  --scheme ortho  --U $U --K 4 --seed 1 --dataset sst2 --probe
  done
  add --name distil_shared_U4 --scheme shared --U 4 --K 4 --seed 1 --model distilbert-base-uncased
  add --name distil_ortho_U4  --scheme ortho  --U 4 --K 4 --seed 1 --model distilbert-base-uncased ;;
*) echo "usage: rev_queue.sh main|main_ortho|ortho_s3|scale|decomp|decomp_a|decomp_b|ablate|ablate_b|general"; exit 1 ;;
esac
runall
echo "QUEUE $1 DONE $(date)"
