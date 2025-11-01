#!/bin/bash
# ============================================================
#  Bash Script: run_R_protein.sh
#  Run unZipro for prioritization of high-fitness mutations in virus-resistance protein (HvMS1)
# ============================================================

# Load conda environment
source ~/.bashrc

# ------------------------------------------------------------
# plant virus-resistance protein (hvms1, 766aa, AlphaFold3 predicted, pLDDT=0.93)
# ------------------------------------------------------------
python script/unZipro_mutation.py \
    --pdb hvms1 \
    --param Models/finetuned/unZipro_HvMS1.pt \
    --outdir outputs/mutation/plant_resistance \
    --name hvms1 \
    --rank_by_prob