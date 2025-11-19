#!/bin/bash
# ============================================================
#  Bash Script: run_nuclease.sh
#  Run unZipro for prioritization of high-fitness mutations in three nucleases
#           (SpCas9 endonuclease, CasΦ2/Cas12j2 endonuclease, T5E exonuclease)
# ============================================================

# Load conda environment
source ~/.bashrc

# ------------------------------------------------------------
#  SpCas9 nuclease (PDB ID: 4oo8, chain D; 1338 aa)
# ------------------------------------------------------------
python script/unZipro_mutation.py \
    --pdb 4oo8D \
    --pdbdir data/example/ \
    --gpu 0 \
    --param Models/finetuned/unZipro_SpCas9.pt \
    --outdir outputs/mutation/genome_editing/Cas9 \
    --name SpCas9 \
    --rank_by_prob \
    --logits

# ------------------------------------------------------------
#  CasΦ2 / Cas12j2 nuclease (PDB ID: 7lys, chain A; 756 aa)
# ------------------------------------------------------------
python script/unZipro_mutation.py \
    --pdb 7lysA \
    --pdbdir data/example/ \
    --gpu 0 \
    --param Models/finetuned/unZipro_Cas12j2.pt \
    --outdir outputs/mutation/genome_editing/Cas12 \
    --name Cas12j2 \
    --rank_by_prob \
    --logits

# ------------------------------------------------------------
#  T5 endonuclease (AlphaFold3 predicted; pLDDT = 0.93; 291 aa)
# ------------------------------------------------------------
python script/unZipro_mutation.py \
    --pdb T5E \
    --pdbdir data/example/ \
    --gpu 0 \
    --param Models/finetuned/unZipro_T5E.pt \
    --outdir outputs/mutation/genome_editing/T5E \
    --name T5E \
    --rank_by_prob \
    --logits