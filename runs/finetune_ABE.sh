#!/usr/bin/env bash
# In the unZipro conda environment:
#   bash runs/finetune_ABE.sh --download-only
# Fine-tuning uses FOMAML to reduce GPU memory usage:
#   bash runs/finetune_ABE.sh
#   bash runs/finetune_ABE.sh --batchsize 1 --epochs 20
# requests uses the existing HTTP_PROXY / HTTPS_PROXY (or lowercase) variables.
# Biotite 0.40 requires NumPy < 2; requirements.txt pins NumPy 1.26.4.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON:-python}"
dataset_dir="data/finetuned_dataset/ABE"
pdb_dir="data/finetuned_dataset/PDB"
download_only=false
if [[ "${1:-}" == "--download-only" ]]; then
    download_only=true
    shift
fi

mkdir -p "$pdb_dir" "$dataset_dir"
# Combine the existing splits without changing train.csv or test.csv.
awk '{ sub(/\r$/, ""); if (NF && !seen[$0]++) print }' \
    "$dataset_dir/train.csv" "$dataset_dir/test.csv" > "$dataset_dir/PDB_IDs.txt"

"$python_bin" script/fetch_PDB_parallel.py \
    --pdblist "$dataset_dir/PDB_IDs.txt" \
    --outdir "$pdb_dir" \
    --cpu "${FETCH_WORKERS:-8}" \
    --report "$dataset_dir/download_report.json"

if "$download_only"; then
    exit 0
fi

"$python_bin" script/unZipro_finetuning.py \
    --train "$dataset_dir/train.csv" \
    --valid "$dataset_dir/test.csv" \
    --config_path config/unZipro_pretrain.json \
    --param Models/unZipro_params.pt \
    --project_name unZipro_ABE \
    --model Models/finetuned/ABE \
    --pdbdir "$pdb_dir" \
    --cache_dir data/finetuned_dataset/tmp \
    --epochs 20 \
    --first_order \
    --adapt_step 10 \
    --adapt_lr 1e-6 \
    --meta_lr 1e-6 \
    --batchsize 5 \
    --cpu "${CPU:-8}" \
    --gpu "${GPU:-0}" \
    "$@"
