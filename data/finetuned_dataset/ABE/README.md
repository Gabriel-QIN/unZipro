# ABE download and fine-tuning example

`train.csv` contains 100 training structure IDs, and `test.csv` contains 20
validation IDs. `PDB_IDs.txt` combines these lists without changing either
split. All 120 structures were downloaded through the configured proxy and
passed the fine-tuning PDB check with 20 neighbors. See `download_report.json`
for per-file paths and validation counts.

The split groups structures by the first four characters of the PDB ID,
ignoring case, so different chains from one PDB cannot appear in both training
and validation. To remove the previous overlap while retaining all 120 IDs and
the 100/20 sizes, `7bv5C` and `7bv5D` were moved to validation, and `8aw32` and
`4lcpA` were moved to training. All four `7bv5` chains are now in validation;
both `8aw3` chains are in training. `PDB_IDs.txt` follows the updated training
list followed by the validation list.

Existing fine-tuned models may have used the previous split. Retrain on the
updated lists before reporting validation results for this split.

From the repository root, in the unZipro environment:

```bash
# Fetch missing structures only; existing files are reused.
bash runs/finetune_ABE.sh --download-only

# Fetch and run FOMAML fine-tuning to reduce GPU memory usage.
bash runs/finetune_ABE.sh

# Use smaller query batches if needed.
bash runs/finetune_ABE.sh --batchsize 1
```

The script uses `script/fetch_PDB_parallel.py`, inherits `http_proxy` and
`https_proxy`, and writes structures to `data/finetuned_dataset/PDB/`. Model
output is `Models/finetuned/ABE/unZipro_ABE.pt`.

Biotite 0.40 needs NumPy below 2. The current `unZipro` conda environment has
been repaired to use NumPy 1.26.4 with Biotite 0.40.0. For a fresh environment,
use the repository's pinned NumPy 1.26.4 dependency.
