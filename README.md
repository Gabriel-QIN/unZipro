# unZipro

<a href=""><img src="https://img.shields.io/badge/Paper-bioRxiv-green" style="max-width: 100%;"></a>
<a href="https://huggingface.co/unZipro"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-red?label=unZipro" style="max-width: 100%;"></a>
<a href="https://colab.research.google.com/github/Gabriel-Qin/unZipro/blob/main/notebooks/unZipro.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Official PyTorch implementation of **unZipro** — an unsupervised zero-shot inverse folding framework for protein evolution and high-fitness variant prediction.

# Overview

unZipro (<u>un</u>supervised Zero-shot <u>i</u>nverse folding framework for <u>pro</u>tein evolution) is a lightweight graph neural network (GNN)-based framework designed for AI-guided protein engineering.

By combining general inverse folding constraints with family-specific adaptation, unZipro efficiently prioritizes high-fitness mutations without exhaustive screening.

## ⚙️ How it works
unZipro tackles protein engineering like “hunting for the needle in the haystack”:

- 🧠 Zero-shot transfer learning captures a universal protein fitness landscape.
- 🧩 Meta-learning adapts to family-specific fitness landscapes.
- Prioritization of the most promising high-fitness variants for experimental validation.

## 🚀 Key Features of unZipro

1. Zero-shot transfer – predict functional variants without extensive few-shot training or large experimental datasets.
2. Highly efficient – drastically reduces experimental screening (as few as ~10 variants) and computational costs.
3. High accuracy – achieves an average of 61% success for high-fitness mutations (>1.1× WT), with up to 100% success and 28× improvement in desired properties.
4. Broad applicability – experimentally validated across enzyme, nucleases, polymerases, transcription factors, virus-resistance proteins, with potential for more protein engineering applications.
5. Structure-flexible: supports both experimentally-resoveled structures and AlphaFold-predicted models.

## 🌱 Applications

- Enzyme engineering 
- Optimization of genome editing tools (e.g., SpCas9, Cas12, base editor, and prime editor)
- Plant protein engineering (virus resistance enhancement, transcription factor engineering, etc.)
- Protein therapeutics
- General protein design tasks in biotechnology & agriculture

# Google Colab  <a href="https://colab.research.google.com/github/Gabriel-Qin/unZipro/blob/main/notebooks/unZipro.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
We prepared a convenient google colab notebook to perform the unZipro code functionalities. However, as the pipeline requires significant amount of GPU memory to run for larger protein structures, we highly recommend to run it using a local installation and at least 32 Gb of GPU memory.

# Run unZipro on local machine
## Installation
```sh
git clone https://github.com/Gabriel-Qin/unZipro.git
cd unZipro

# Install dependencies and PyTorch automatically
bash runs/install_unZipro.sh
```
Alternatively, you can manually install all dependencies with:
```sh
pip install -r requirements.txt
```
After installation, verify the environment with:
```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())" # Expected output: `2.4.1+cu124 True`
```
This confirms that PyTorch is correctly installed and GPU acceleration is available.

## Pretraining
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

## Finetuning
Run unZipro fine-tuning easily with the following command.  
```
# Before runnning, you must prepare your train/valid dataset (each file with corresponding PDB IDs)
# Ensure your PDB files are in `pdbdir`
name=P53_HUMAN
python script/unZipro_finetuning.py --train data/finetuned_dataset/${name}/train.csv --valid data/finetuned_dataset/${name}/test.csv --project_name unZipro_${name} --model Models/finetuned/${name} --pdbdir data/finetuned_dataset/PDB/ --cache_dir data/finetuned_dataset/tmp/ --epoch 20 
```
For more details, see [Finetuning](docs/finetuning.md).

## High-fitness mutation prioritization

> unZipro predicts and prioritizes beneficial mutations directly from protein structures, enabling structure-aware protein engineering without supervised fine-tuning.

### Inference Example:
```sh
python script/main.py --pdb 6vpcE --pdb_dir data/example/ --outdir data/outputs/ --work_dir data/tmp --rank_by_prob
```
The outputs include:

- Per-residue mutation probabilities/logits

- Ranked potential high-fitness mutations

Following are some provided `examples`:

| Category               | Script                   | Description                                             |
| ---------------------- | ------------------------ | ------------------------------------------------------- |
| **Genome editors**     | `runs/run_ABE.sh`        | Adenine base editor (TadA8e)                            |
|                        | `runs/run_nuclease.sh`   | Three nucleases (SpCas9, CasΦ2/Cas12j2, T5E)            |
|                        | `runs/run_polymerase.sh` | MMLV reverse transcriptase under multiple conformations |
| **Fluorescent enzyme** | `runs/run_luciferase.sh` | Luciferase for improved fluorescence intensity          |
| **Plant proteins**     | `runs/run_plantTF.sh`    | DNA-binding domains of plant transcription factors      |
|                        | `runs/run_R_protein.sh`  | Plant virus-resistance (R) proteins      

## 🙏 Acknowledgements
We gratefully acknowledge the open-source community for providing valuable tools and insights that inspired the development of unZipro.
This work builds upon ideas and methodologies introduced by previous research in AI/ML, protein inverse folding, and homolog search.

In particular, we recognize the contributions of prior works including many graph-based protein design frameworks and Foldseek, which have laid the foundation for advances in structure-informed protein engineering.

We sincerely thank the authors of these repositories for their pioneering efforts and their invaluable contributions to the broader scientific community.

## License
Distributed under [BSD 3-Clause](https://github.com/Gabriel-QIN/unZipro/blob/master/LICENSE) license.

## 📖 Citation

If you use unZipro in your research, please cite:
