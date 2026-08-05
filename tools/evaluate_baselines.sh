#!/usr/bin/env bash
set -euo pipefail
for policy in aasld easl apasl fib4 amap galad amri risk_adaptive
do
  scdsmdp-evaluate --config settings/main.yaml --policy "$policy" --episodes 50000
done
