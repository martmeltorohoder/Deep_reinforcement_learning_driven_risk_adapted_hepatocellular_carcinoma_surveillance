#!/usr/bin/env bash
set -euo pipefail
for seed in $(seq 0 19)
do
  CUDA_VISIBLE_DEVICES=$seed scdsmdp-train --config settings/main.yaml --output "runs/seed_${seed}" &
done
wait

