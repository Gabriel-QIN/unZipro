#!/bin/bash
# ============================================================
#  Bash script: run_luciferase.sh
#  Run unZipro for prioritization of high-fitness mutations in luciferase
# ============================================================

# Load conda environment
source ~/.bashrc
conda activate unZipro

# ------------------------------------------------------------
### luciferase (PDB ID: 1cli chain A, 550aa)
# ------------------------------------------------------------
python script/unZipro_mutation.py \
    --pdb 1cliA \
    --param Models/finetuned/unZipro_LUC.pt \
    --outdir outputs/mutation/enzyme \
    --name LUC \
    --rank_by_prob