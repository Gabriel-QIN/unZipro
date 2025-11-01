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

For example to make a conda environment to run unZipro, see [install](runs/install_unZipro.sh):

Following are provided `examples`:

### Genome editing proteins
- `runs/run_ABE.sh` - adenine base editor
- `runs/run_nuclease.sh` - three nucleases
- `runs/run_polymerase.sh` - MMLV RT polymearse (based on different conformational states)
### industrial enzyme
- `runs/run_luciferase.sh` - firefly luciferase
### AlphaFold-predicted plant structures
- `runs/run_plantTF.sh` - DNA-binding domains of plant transcription factors
- `runs/run_R_protein.sh` - Plant virus-resistance protein