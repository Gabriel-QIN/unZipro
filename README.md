# unZipro

![Framework](https://img.shields.io/badge/Protein-Evolution-blue?style=for-the-badge&logo=github)

This package provides an official implementation of **unZipro** using PyTorch.

## Overview

unZipro (<u>un</u>supervised Zero-shot <u>i</u>nverse folding framework for <u>pro</u>tein evolution) is a lightweight graph neural network (GNN)-based framework designed for AI-guided protein engineering.

unZipro integrates zero-shot transfer learning with inverse folding models to capture both:

- Universal sequence–structure constraints learned from diverse protein structures;
- Family-specific adaptation via meta-learning on subset of structural tasks.

This nuanced design allows efficient fitness landscape modelling and accurate nomination of high-fitness variants — even without experimental data for finetuning.

### 🚀 Key Features of unZipro

1. Zero-shot transfer: No need for extensive few-shot training or large experimental datasets
2. Efficient: Substantially reduces experimental screening by prioritizing top candidates (as few as 10 variants)
3. Accurate: Predicts evolutionarily plausible amino acid substitutions tailored to target protein families
4. Broad applicability: Works across enzymes, polymerases, transcription factors, virus-resistance proteins, and more
5. Structure-flexible: Supports both crystal structures and AlphaFold-predicted models, including low-confidence regions

### 📊 Experimental benchmark Results

​	We have validated unZipro on 10 diverse protein engineering tasks, including:

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


### High-fitness mutation recommendation

#### 1.Inference using pretrained models


#### 2. Inference using finetuned models

## Acknowledgements


## 📖 Citation

If you use unZipro in your research, please cite:

(preprint / paper link to be added here)