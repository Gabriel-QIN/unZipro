# Using unZipro to prioritize high-fitness mutations
Command-line Flags for `unZipro_mutation.py`:
| Flag                    | Type / Example                                       | Description                                                  |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| `--pdb`, `-i`           | `file` or `comma-separated list` (e.g., `8wutE,8wuvE`) | Input PDB file(s) or IDs. Supports both a text file containing PDB names and comma-separated entries. |
| `--pdbdir`, `-dir`      | `path`                                               | Directory containing the PDB structure files.                |
| `--gpu`, `-gpu`         | `int` (default: `0`)                                 | Specifies the GPU device ID; falls back to CPU if unavailable.                                        |
| `--param`, `-m`         | `path`                                               | Path to the pretrained or finetuned model parameter file (`.pt`). |
| `--outdir`, `-o`        | `path`                                               | Output directory for saving results.                         |
| `--name`, `-n`          | `str`                                                | Custom name prefix. |
| `--nneighbor`, `-nb`    | `int` (default: `20`)                                | Number of neighboring nodes considered in graph construction. |
| `--cachedir`, `-cd`     | `path`                                               | Directory to store cached protein structure features.        |
| `--probs`, `-p`         | `store_true`                                               | Output per-residue sequence probabilities.                   |
| `--logits`, `-l`        | `store_true`                                               | Output raw model logits instead of probabilities.            |
| `--rank_by_prob`, `-rp` | `store_true`                                              | Rank mutations by predicted mutation probability.            |
| `--res`, `-res`         | `comma-separated list` (e.g., `83,123`)              | Restrict output to specific residue positions.               |


### Genome editing proteins

```sh
# Adenine base editor (TadA8e)
python script/unZipro_mutation.py \
    --pdb 6vpcE,6vpcF \
    --param Models/finetuned/unZipro_ABE.pt \
    --outdir outputs/mutation/genome_editing/ABE \
    --name TadA8e \
    --rank_by_prob

# Cas9 nuclease
python script/unZipro_mutation.py \
    --pdb 4oo8D \
    --param Models/finetuned/unZipro_SpCas9.pt \
    --outdir outputs/mutation/genome_editing/Cas9 \
    --name SpCas9 \
    --rank_by_prob

# Cas12j2 nuclease
python script/unZipro_mutation.py \
    --pdb 7lysA \
    --param Models/finetuned/unZipro_Cas12j2.pt \
    --outdir outputs/mutation/genome_editing/Cas12 \
    --name Cas12j2 \
    --rank_by_prob

# T5 endonuclease
python script/unZipro_mutation.py \
    --pdb T5E \
    --param Models/finetuned/unZipro_T5E.pt \
    --outdir outputs/mutation/genome_editing/T5E \
    --name T5E \
    --rank_by_prob

# Prime editor (MMLV RT)
python script/unZipro_mutation.py \
    --pdb 8wutE,8wuvE \
    --param Models/finetuned/unZipro_MMLVRT.pt \
    --outdir outputs/mutation/genome_editing/PE \
    --name PE \
    --rank_by_prob
```

### luciferase
```sh
# 1cliA
python script/unZipro_mutation.py \
    --pdb 1cliA \
    --param Models/finetuned/unZipro_LUC.pt \
    --outdir outputs/mutation/enzyme \
    --name LUC \
    --rank_by_prob
```

### plant transcription factors
```sh
# OsPHR2
python script/unZipro_mutation.py \
    --pdb OsPHR2 \
    --param Models/finetuned/unZipro_OsPHR2.pt \
    --outdir outputs/mutation/plantTF \
    --name OsPHR2 \
    --rank_by_prob

# OsNAC3
python script/unZipro_mutation.py \
    --pdb OsNAC3 \
    --param Models/finetuned/unZipro_OsNAC3.pt \
    --outdir outputs/mutation/plantTF \
    --name OsNAC3 \
    --rank_by_prob
```

### plant virus-resistance protein
```sh
# hvms1
python script/unZipro_mutation.py \
    --pdb hvms1 \
    --param Models/finetuned/unZipro_HvMS1.pt \
    --outdir outputs/mutation/plant_resistance \
    --name hvms1 \
    --rank_by_prob
```