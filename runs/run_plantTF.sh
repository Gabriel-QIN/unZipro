#!/bin/bash
# ============================================================
#  Bash Script: run_plantTF.sh
#  Run unZipro for prioritization of high-fitness mutations in plant transcription factors DNA-binding domain (OsPHR2 and OsNAC3)
# ============================================================


### plant transcription factors
# ----------------------------------------------------------------------
# OsPHR2 (AlphaFold3 predicted, DNA binding domain 249-302, pLDDT=0.91))
# ----------------------------------------------------------------------

python script/unZipro_mutation.py \
    --pdb OsPHR2 \
    --param Models/finetuned/unZipro_OsPHR2.pt \
    --outdir outputs/mutation/plantTF \
    --name OsPHR2 \
    --rank_by_prob

# ----------------------------------------------------------------------
# OsNAC3 (AlphaFold3 predicted, DNA binding domain 24-143, pLDDT=0.90)
# ----------------------------------------------------------------------
python script/unZipro_mutation.py \
    --pdb OsNAC3 \
    --param Models/finetuned/unZipro_OsNAC3.pt \
    --outdir outputs/mutation/plantTF \
    --name OsNAC3 \
    --rank_by_prob