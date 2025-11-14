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
### 3.1 Finetuning on a single protein
```sh
# Set protein name
name=P53_HUMAN
python script/unZipro_finetuning.py --train data/finetuned_dataset/${name}/train.csv --valid data --config_path config/unZipro_pretrain.json finetuned_dataset/${name}/test.csv --project_name unZipro_${name} --model Models/finetuned/${name} --pdbdir data/finetuned_dataset/PDB/ --cache_dir data/finetuned_dataset/tmp/ --epoch 20 
```
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
> - Fine-tuning typically takes 1-3 minutes on GPU / 5-10 minutes on CPU.