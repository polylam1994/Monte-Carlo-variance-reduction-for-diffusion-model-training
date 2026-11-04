#!/usr/bin/env bash
# Reproduce the project.
#   bash scripts/reproduce.sh paper    # results/ -> runs.csv, figures, tables, numbers, paper/paper.pdf (minutes, CPU)
#   bash scripts/reproduce.sh fmnist   # re-run the Fashion-MNIST study on CPU (~3-4 h with 4 cores)
#   bash scripts/reproduce.sh pilot    # 1-minute end-to-end pilot on CPU
# CIFAR-10 runs: upload notebooks/cifar10_seed0.py, cifar10_seeds12.py and cifar10_sa_s2.py as Kaggle *script*
# kernels (GPU T4 x2, internet on) and decode their logs with scripts/fetch_kaggle_results.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
case "${1:-paper}" in
  paper)
    python scripts/collect_runs.py
    python scripts/make_tables.py
    python scripts/make_numbers.py
    python scripts/make_figures.py
    (cd paper && latexmk -pdf -interaction=nonstopmode paper.tex >/dev/null && echo "paper/paper.pdf built")
    ;;
  fmnist)
    python scripts/download_fmnist.py data/fmnist
    python scripts/run_queue.py --plan configs/plans/fmnist_main.json --cpu-workers 4
    ;;
  pilot)
    python scripts/download_fmnist.py data/fmnist
    python -m mcvr.train --config configs/pilot_fmnist.yaml --set out_dir=results/raw/pilot_rerun
    ;;
  *) echo "unknown target $1"; exit 1;;
esac
