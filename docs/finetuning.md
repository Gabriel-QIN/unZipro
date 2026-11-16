# unZipro finetuning strategy
There are three main steps for unZipro finetuning:

1. Run Foldseek search locally or in Foldseek web interface
2. Prepare dataset for meta-train (dissimilar structural tasks) and meta-test (homologous strcutural tasks) sets, respectively
3. Finetuning pre-trained inverse folding model

## Getting started
> Note: if you already have Foldseek environment and PDB100/AFDB50 database, skip this step!
### Install Foldseek
```sh
# install
conda install -c conda-forge -c bioconda foldseek
```
### Download PDB100 and AFDB50
```sh
# PDB100
foldseek databases PDB100 data/PDB100 tmp
# AFDB50
foldseek databases AFDB50 data/AFDB50 tmp
```
For other structural databases, please refer to the [Foldseek repo](https://github.com/steineggerlab/foldseek).

## 1. Run Foldseek on PDB100 and AFDB50 datasets
### Run locally
Foldseek searches each query protein against PDB100 and AFDB50 databases and saves the results in .m8 tabular files.
```sh
# run Foldseek search

query_dir="data/benchmark/ProteinGym_pdb"
outdir="data/Foldseek_search/"
tmpdir="data/tmp_search/"
mkdir -p "$outdir" "$tmpdir"

for query in ${query_dir}/*.pdb; do
    name=$(basename "$query" .pdb)
    echo "🔍 Processing $name ..."
    mkdir "${outdir}/${name}"
    foldseek easy-search "$query" data/PDB100 "${outdir}/${name}/alis_pdb100.m8" "$tmpdir" --threads 8 -s 9.5
    foldseek easy-search "$query" data/AFDB50 "${outdir}/${name}/alis_afdb50.m8" "$tmpdir" --threads 8 -s 9.5
done
```
#### Output

- Results will be saved as `.m8` tabular files containing Foldseek search hits.
- Adjust the `-s` parameter (sensitivity) and `--threads` as needed for your computational environment.

### Use Foldseek web server
If you prefer not to run Foldseek locally, there are two ways to submit queries to the Foldseek web server:
#### i) Submit via Foldseek Web API
- For automated or batch submission.
- Note: aria2/wget is required to download results from the server.
```sh
# Install aria2 (required if you are using aria2 to download results)
# If you prefer wget, uncomment the following line
sudo apt update && sudo apt install -y aria2
# Submit task to Foldseek web server through API
# Submit a single protein structure to Foldseek server
python script/foldseek_api.py --pdb SpCas9  --outdir data/Foldseek_search/ --pdb_dir data/example/ --wait_time 30
# Batch submit to Foldseek server
python script/foldseek_api.py --pdb data/example/example.txt  --outdir data/Foldseek_search/ --pdb_dir data/example/ --wait_time 30
```
#### ii) Submit via the official web interface
- For manual or small-scale queries.

Simply open the [Foldseek web server](https://search.foldseek.com/search) in your browser and upload your PDB or mmCIF file:
👉 https://search.foldseek.com/search

Finally, you will get the search results like this in `data/FoldSeek_search/` directory.
```sh
data/FoldSeek_search/
├── [your_protein_name]/
│   ├── alis_afdb50.m8
│   ├── alis_pdb100.m8
```

## 2. Prepare data for finetuning
### i) Dataset split for meta-transfer learning
In this step, we split the Foldseek-retrived structures into meta-training (dissimilar strcutural tasks) and meta-testing (homologous strcutural tasks) datasets.
```sh
python script/parse_foldseek_results.py --pdb data/example/example.txt --foldseek_dir data/Foldseek_search --outdir data/finetuned_dataset/
```
### ii) Data download
```sh
python script/fetch_PDB_parallel.py -i data/finetuned_dataset/PDB_IDs.txt -o data/finetuned_dataset/PDB -cpu 8
```
>- Note: Some PDB files may fail to download. In such cases, simply skip them. The overall training dataset is expected to contain ~100 structures.
>- Some of the PDB structures are solution NMR structures and generally have low resolution. Users should be aware of potential limitations when using these structures for modeling or analysis.
>- If your query structure has no matches in the provided databases, we strongly recommend creating your own dataset, *e.g.*, by retrieving homologs based on sequence similarity.
## 3. unZipro finetuning strategy
### ABE example

`runs/finetune_ABE.sh` combines ABE's 100 training and 20 validation IDs,
downloads their single-chain structures with `fetch_PDB_parallel.py`, and runs
FOMAML fine-tuning to reduce GPU memory usage. Existing structures are reused.
Downloads inherit the current `http_proxy` / `https_proxy` environment variables.

```sh
conda activate unZipro
# Download only; save per-item results in ABE/download_report.json.
bash runs/finetune_ABE.sh --download-only
# Download missing files, then run FOMAML fine-tuning for 20 epochs.
bash runs/finetune_ABE.sh
# Reduce query batch size further if needed.
bash runs/finetune_ABE.sh --batchsize 1
```

Structures are saved in `data/finetuned_dataset/PDB/`; the trained initialization
is saved as `Models/finetuned/ABE/unZipro_ABE.pt`. Biotite 0.40 requires NumPy
below 2; the repository pins NumPy 1.26.4. `PYTHON` can select another Python
interpreter, and `FETCH_WORKERS`, `CPU`, and `GPU` configure worker counts and
the GPU index.

### 3.1 Finetuning on a single protein
```sh
# Set protein name
name=P53_HUMAN
python script/unZipro_finetuning.py \
    --train "data/finetuned_dataset/${name}/train.csv" \
    --valid "data/finetuned_dataset/${name}/test.csv" \
    --config_path config/unZipro_pretrain.json \
    --project_name "unZipro_${name}" \
    --model "Models/finetuned/${name}" \
    --pdbdir data/finetuned_dataset/PDB/ \
    --cache_dir data/finetuned_dataset/tmp/ \
    --epochs 20
```

Each input list must contain at least two usable structures after missing or
invalid PDB files have been filtered out. The script reports the listed, found,
and usable counts before training. Check `--pdbdir` and download the structures
in step 2 if too few remain. Lists contain one PDB ID per line, without a header;
the `.pdb` extension is optional.

Prechecking skips structures with no valid target residues, including CA-only
PDBs without the required N/CA/C backbone. Each rejection includes the file path
and reason; these structures are excluded before the support/query split.

Each epoch forms one training task by splitting the training structures into
disjoint support and query sets. A fresh learner is cloned once from the learned
initialization and adapted on the support set for `--adapt_step` consecutive SGD
steps at `--adapt_lr`. The mean query loss over valid residues then produces one
Adam update of the initialization at `--meta_lr`. Query batches share the same
adapted learner. Full second-order MAML is the default; add `--first_order` to use
the FOMAML approximation and reduce memory use.

Validation uses a fixed support/query split and no feature noise. It adapts a
fresh clone on validation support structures and evaluates the validation query
structures in evaluation mode, leaving the learned initialization, BatchNorm
buffers, and optimizer unchanged. The saved model contains the initialization
from the epoch with the lowest validation loss. Set `--cpu 0` for loading and PDB
checking without worker processes. Both `--epochs` and `--epoch` are accepted.

### 3.2 Batch Submission for finetuning multiple proteins
```sh
protein_list="data/example/example.txt"
while read -r name; do
    echo "=== Fine-tuning ${name} ==="
    python script/unZipro_finetuning.py \
        --train "data/finetuned_dataset/${name}/train.csv" \
        --valid "data/finetuned_dataset/${name}/test.csv" \
        --config_path config/unZipro_pretrain.json \
        --project_name "unZipro_${name}" \
        --model "Models/finetuned/${name}" \
        --pdbdir "data/finetuned_dataset/PDB/" \
        --cache_dir "data/finetuned_dataset/tmp/" \
        --epochs 20

    echo "=== Finished ${name} ==="
done < "$protein_list"
```
>Note:
> - Runtime depends on structure lengths, `--adapt_step`, `--batchsize`, and whether full MAML or FOMAML is used.
