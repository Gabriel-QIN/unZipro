#!/bin/bash
# ============================================================
#  Bash Script: run_R_protein.sh
#  Run unZipro for prioritization of high-fitness mutations in prime editor (PE)
# ============================================================

# ----------------------------------------------------------------------
# MMLV RT initiation state (PDB ID:8wut, chain E, 496aa)
# ----------------------------------------------------------------------

python script/unZipro_mutation.py \
    --pdb 8wutE \
    --param Models/finetuned/unZipro_MMLVRT.pt \
    --outdir outputs/mutation/plantTF \
    --name MMLVRT_initiation \
    --rank_by_prob

# ----------------------------------------------------------------------
# MMLV RT elongation state (PDB ID:8wuv, chain E, 496aa)
# ----------------------------------------------------------------------
python script/unZipro_mutation.py \
    --pdb 8wuvE \
    --param Models/finetuned/unZipro_MMLVRT.pt \
    --outdir outputs/mutation/plantTF \
    --name MMLVRT_elongation \
    --rank_by_prob