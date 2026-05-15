#! /bin/bash
# ============================================================
#  Bash script: evaluate_pretrained_model.sh
#  Evaluate pre-trained model on multiple benchmark datasets
# ============================================================

# activate your environment
source ~/.bashrc
conda activate unZipro

# running project parameters
CONFIG_PATH="config/unZipro_pretrain.json"
PARAM="Models/unZipro_params.pt"
OUTDIR="outputs/seq_design"
GPU=0
STRATEGY="argmax"

datasets=("PDB50" "T500" "TS50" "CASP14" "CASP15" "CAMEO")

for dataset in "${datasets[@]}"; do
    echo "🚀 Running evaluation for ${dataset}..."
    python script/unZipro_evaluate.py \
        --project_name "unZipro_${dataset}_test" \
        --input "data/benchmark/seq_design/${dataset}.txt" \
        --pdbdir "data/benchmark/seq_design/${dataset}" \
        --outdir "$OUTDIR" \
        --config_path "$CONFIG_PATH" \
        --param "$PARAM" \
        --gpu "$GPU" \
        --sampling_strategy "$STRATEGY"

    echo "✅ Finished ${dataset}"
    echo "---------------------------------------------"
done

echo "🎉 All evaluations completed."