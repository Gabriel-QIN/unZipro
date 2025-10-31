# unZipro

<a href=""><img src="https://img.shields.io/badge/Paper-bioRxiv-green" style="max-width: 100%;"></a>
<a href="https://huggingface.co/unZipro"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-red?label=unZipro" style="max-width: 100%;"></a>
<a href="https://colab.research.google.com/github/Gabriel-Qin/unZipro/blob/main/notebooks/unZipro.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Official PyTorch implementation of **unZipro** — an unsupervised zero-shot inverse folding framework for protein evolution and high-fitness variant prediction.

## Overview

unZipro (<u>un</u>supervised Zero-shot <u>i</u>nverse folding framework for <u>pro</u>tein evolution) is a lightweight graph neural network (GNN)-based framework designed for AI-guided protein engineering.

unZipro is comprised of a two-step learning:

🧠 Zero-shot transfer learning to capture universal sequence–structure constraints
🧩 Meta-learning for family-specific adaptation from structural subsets
We have achieved up to 28-fold improvement in desired protein properties and success rate of high-fitness variants (>1.1-fold compared with WT) up to 100% (an average of 61%).
Together, these capabilities enable accurate modeling of fitness landscapes and prioritization of high-fitness variants, even without supervised finetuning.
unZipro achieved up to 28× improvement in desired properties
and up to 100% success rate for high-fitness mutation prediction (>1.1× WT).

### 🚀 Key Features of unZipro

1. Zero-shot transfer: No need for extensive few-shot training or large experimental datasets
2. Efficient: Substantially reduces experimental screening by prioritizing top candidates (as few as ~10 variants)
3. Accurate: Predicts evolutionarily plausible amino acid substitutions tailored to target protein families
4. Broad applicability: Experimentally demonstrated successfully across enzymes, polymerases, transcription factors, virus-resistance proteins, and more
5. Structure-flexible: Supports both crystal structures and AlphaFold-predicted models

### 🌱 Applications

- Enzyme engineering 
- Optimization of genome editing tools (e.g., SpCas9, BE, and PE)
- Plant protein engineering (virus resistance, transcription factor tuning, etc.)
- Protein therapeutics
- General protein design tasks in biotechnology & agriculture

## Google Colab<a href="https://colab.research.google.com/github/Gabriel-Qin/unZipro/blob/main/notebooks/unZipro.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
We prepared a convenient google colab notebook to perform the unZipro code functionalities. However, as the pipeline requires significant amount of GPU memory to run for larger protein structures, we highly recommend to run it using a local installation and at least 32 Gb of GPU memory.

## Deploy unZipro on local server
### Installation
```sh
git clone https://github.com/Gabriel-Qin/unZipro.git
cd unZipro
conda create -n unZipro python=3.9
conda activate unZipro
pip install -r requirements.txt
```
### Pretraining
You can reproce the unZipro pre-training and evaluation following the instructions from [Pre-training](docs/pretrain.md).

Or pre-train on your own structure dataset
```python
python script/unZipro_pretrain.py \
    --train_list data/pretrained/train.txt \
    --valid_list data/pretrained/valid.txt \
    --pdbdir data/pretrained/PDB \
    --epochs 100 \
    --batchsize 10 \
    --model Models \
    --cachedir data/pretrained/tmp/ \
    --project_name unZipro_pretrain
```

### Finetuning
#### 1. Run Foldseek Search on PDB100 and AFDB50 Datasets
You can perform structure-based searches using Foldseek on your local machine.
For users who want to set up Foldseek on their local server, please refer to [here](docs/foldseek.md)..
If you prefer not to run locally, you can use the official Foldseek Web Server:
👉 https://search.foldseek.com/search
runs/evaluate_pretrained_model.sh

#### 2. Download data
```sh
python script/fetch_PDB_parallel.py -i data/finetuned/PDB_IDs.txt -o data/finetuned/PDB -m RCSB ps -cpu 8
python script/fetch_PDB_parallel.py -i data/finetuned/AFDB_IDs.txt -o data/finetuned/PDB -m AF ps -cpu 8
```

#### 3. Finetuning

#### 3. Finetuning

#### Run inference

### High-fitness mutation recommendation
>This is the core module of **unZipro** — predicting high-fitness mutations
>1. Predicts high-fitness amino acid substitutions directly from structure
>2. Ranks variants based on learned family-specific fitness landscape
>3. Works in zero-shot manner — no supervised fine-tuning required

#### 1️⃣ Inference using pretrained models



#### 2️⃣ Inference using finetuned models

##### 📊 Experimental valudation results

​We have validated unZipro on 10 diverse protein engineering tasks, including:

1. Deaminase (TadA8e) for improved base editing efficiency
2. Nucleases (SpCas9, CasΦ2, T5E) for improved gene-editing activity
3. Reverse transcriptase (MMLV-RT) for improved prime editing efficiency
4. Luciferase with improved fluorescence intensity
5. Plant transcription factors with enhanced transcriptional activity
6. Plant virus-resistance proteins with reduced virulence

We have achieved up to 28-fold improvement in desired protein properties and success rate of high-fitness variants (>1.1-fold compared with WT) up to 100% (an average of 61%).


💡 Outputs include mutation probabilities, ranked residue-wise substitution lists,
and sequence fitness heatmaps.

## 🙏 Acknowledgements
We thank the contributors of PyTorch, learn2learn, AFDB, and Foldseek for providing foundational tools for this work.

unZipro draws inspiration and leverages/modifies implementations from the following repositories:
jingraham/neurips19-graph-protein-design for the preprocessed CATH dataset and data pipeline implementation.
facebook/esm for their ESM implementations, pretrained model weights, and data pipeline components like Alphabet.
dauparas/ProteinMPNN for the ProteinMPNN implementation and multi-chain dataset.
A4Bio/PiFold for their PiFold implementation.
We express our sincere appreciation to the authors of these repositories for their invaluable contributions to the development of ByProt.

## 📖 Citation

If you use unZipro in your research, please cite:

(preprint / paper link to be added here)