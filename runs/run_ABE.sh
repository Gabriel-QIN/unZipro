#!/bin/bash
# ============================================================
#  Bash script: run_ABE.sh
#  Run unZipro for prioritization of high-fitness mutations in ABE
# ============================================================

# Load conda environment
source ~/.bashrc
conda activate unZipro

# ------------------------------------------------------------
#  Adenine Base Editor (TadA8e, PDB ID: 6vpc, chain: E and F, 167aa)
# ------------------------------------------------------------
### pretrained model
python script/unZipro_mutation.py \
    --pdb 6vpcE,6vpcF \
    --pdbdir data/example/ \
    --gpu 0 \
    --param Models/unZipro_params.pt \
    --outdir outputs/mutation/genome_editing/ABE_pretrained \
    --name TadA8e \
    --rank_by_prob \
    --logits

### finetuned model
python script/unZipro_mutation.py \
    --pdb 6vpcE,6vpcF \
    --param Models/finetuned/unZipro_ABE.pt \
    --outdir outputs/mutation/genome_editing/ABE \
    --name TadA8e \
    --rank_by_prob