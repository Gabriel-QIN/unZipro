# unZipro

![Framework](https://img.shields.io/badge/Protein-Evolution-blue?style=for-the-badge&logo=github)

Official PyTorch implementation of **unZipro** — an unsupervised zero-shot inverse folding framework for protein evolution and high-fitness variant prediction.

## Overview

unZipro (<u>un</u>supervised Zero-shot <u>i</u>nverse folding framework for <u>pro</u>tein evolution) is a lightweight graph neural network (GNN)-based framework designed for AI-guided protein engineering.

unZipro is comprised of a two-step learning:

🧠 Zero-shot transfer learning to capture universal sequence–structure constraints
🧩 Meta-learning for family-specific adaptation from structural subsets

Together, these capabilities enable accurate modeling of fitness landscapes and prioritization of high-fitness variants, even without supervised finetuning.
unZipro achieved up to 28× improvement in desired properties
and up to 100% success rate for high-fitness mutation prediction (>1.1× WT).

### 🚀 Key Features of unZipro

1. Zero-shot transfer: No need for extensive few-shot training or large experimental datasets
2. Efficient: Substantially reduces experimental screening by prioritizing top candidates (as few as 10 variants)
3. Accurate: Predicts evolutionarily plausible amino acid substitutions tailored to target protein families
4. Broad applicability: Works across enzymes, polymerases, transcription factors, virus-resistance proteins, and more
5. Structure-flexible: Supports both crystal structures and AlphaFold-predicted models, including low-confidence regions

### 📊 Experimental benchmark Results

​We have validated unZipro on 10 diverse protein engineering tasks, including:

1. Deaminase (TadA8e) for improved base editing efficiency
2. Nucleases (SpCas9, SpuFanzor, CasΦ2, T5E) for improved gene-editing activity
3. Reverse transcriptase (MMLV-RT) for improved prime editing efficiency
4. Luciferase with improved fluorescence intensity
5. Plant transcription factors with enhanced transcriptional activity
6. Wheat virus-resistance proteins with reduced virulence

We have achieved up to 28-fold improvement in desired protein properties and success rate of high-fitness variants (>1.1-fold compared with WT) up to 100% (an average of 61%).

### 🌱 Applications

- Enzyme engineering 
- Optimization of genome editing tools (e.g., SpCas9, BE, and PE)
- Plant protein engineering (virus resistance, transcription factor tuning, etc.)
- Protein therapeutics
- General protein design tasks in biotechnology & agriculture

## Getting started

### Installation
```sh
git clone https://github.com/Gabriel-Qin/unZipro.git
cd unZipro
conda create -n unzipro python=3.9
conda activate unzipro
pip install -r requirements.txt
```
### Pretrain on PDB50 datasets
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

### Finetuning on homolog datasets
#### 1. Run Foldseek search on PDB50 and AFDB50 datasets

#### 2. 

You can also evaluate all benchmarks automatically via
runs/evaluate_pretrained_model.sh


### 🌟 High-fitness mutation recommendation
>This is the core function of **unZipro** — predicting high-fitness mutations
based on structure-conditioned inverse folding model and family-specific finetuning
<blockquote style="border-left: 4px solid #4CAF50; padding: 10px 15px; background: #f9f9f9;">
<ul>
<li>Predicts beneficial amino acid substitutions directly from structure</li>
<li>Ranks variants based on learned fitness landscape</li>
<li>Works in zero-shot mode — no supervised fine-tuning required</li>
</ul>
</blockquote>


#### 1️⃣ Inference using pretrained models



#### 2️⃣ Inference using finetuned models

💡 Outputs include mutation probabilities, ranked residue-wise substitution lists,
and sequence fitness heatmaps.

## 🙏 Acknowledgements
We thank the contributors of PyTorch, learn2learn, and Foldseek for providing foundational tools for this work.

unZipro draws inspiration and leverages/modifies implementations from the following repositories:
jingraham/neurips19-graph-protein-design for the preprocessed CATH dataset and data pipeline implementation.
facebook/esm for their ESM implementations, pretrained model weights, and data pipeline components like Alphabet.
dauparas/ProteinMPNN for the ProteinMPNN implementation and multi-chain dataset.
A4Bio/PiFold for their PiFold implementation.
We express our sincere appreciation to the authors of these repositories for their invaluable contributions to the development of ByProt.

## 📖 Citation

If you use unZipro in your research, please cite:

(preprint / paper link to be added here)