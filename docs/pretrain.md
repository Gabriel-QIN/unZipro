# unZipro pre-training documentation
### Pretrain on PDB50 datasets
#### 1. Download full PDB50 dataset
```python
python script/fetch_PDB_parallel.py -i data/pretrained/all_PDB50.txt -o data/pretrained/PDB -m RCSB -cpu 8
```
#### 2. Start pre-training
Input flags for [script/unZipro_pretrain.py]:
#### Data
- `--train_list`
  File containing training PDB IDs or structure paths.
- `--valid_list`
  File containing validation PDB IDs or structure paths.
- `--pdbdir` *(default: `data/pdb`)*
  Directory containing PDB structure files.
- `--cachedir` *(default: `data/tmp/`)*
  Directory for storing cached structural features.
#### Project
- `--project_name` *(default: `unZipro_default`)*
  Project name used for saving checkpoints and best models.
- `--model` *(default: `./Models`)*
  Directory for saving trained models and logits.
- `--logdir` *(default: `./Logs`)*
  Directory for training logs.
- `--config_dir` *(default: `config`)*
  Directory containing model configuration files.
- `--logging`
  Enable TensorBoard logging during training.
#### Training
- `--batchsize` *(default: `10`)*
  Batch size for model training.
- `--cpu` *(default: `16`)*
  Number of CPU cores for data loading.
- `--gpu` *(default: `0`)*
  GPU device ID.
- `--learning-rate` *(default: `0.002`)*
  Learning rate for optimization.
- `--epochs` *(default: `100`)*
  Number of training epochs.
- `--noise` *(default: `0.01`)*
  Noise level used during training augmentation.
- `--nneighbor` *(default: `20`)*
  Number of neighboring residues used for graph construction.
#### Model architecture
- `--dim-hidden-node0` *(default: `40`)*
  Hidden dimensions of the initial node embedding layers.
- `--layer-embed-node0` *(default: `20`)*
  Number of initial node embedding layers.
- `--iter-gcn` *(default: `5`)*
  Number of graph convolution iterations.
- `--knode-gcn` *(default: `20`)*
  Additional node feature dimensions used in GCN updates.
- `--kedge-gcn` *(default: `20`)*
  Additional edge feature dimensions used in GCN updates.
- `--dim-hidden-node` *(default: `256`)*
  Hidden dimensions for node embeddings.
- `--dim-hidden-edge` *(default: `256`)*
  Hidden dimensions for edge embeddings.
- `--layer-embed-node` *(default: `2`)*
  Number of node embedding layers.
- `--layer-embed-edge` *(default: `2`)*
  Number of edge embedding layers.
- `--dim-hidden-pred1` *(default: `128`)*
  Hidden dimensions of prediction layer 1.
- `--dim-hidden-pred2` *(default: `64`)*
  Hidden dimensions of prediction layer 2.
- `--layer-pred` *(default: `4`)*
  Number of prediction layers.
- `--fragsize` *(default: `9`)*
  Fragment size used in the prediction module.

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
#### 3. Evaluate the pre-trained model
You can evaluate the pre-trained unZipro model on a single benchmark dataset, for example TS50, by running:
```
dataset=TS50
python script/unZipro_evaluate.py --project_name unZipro_${dataset}_test \
    --input data/pretrained/benchmark/${dataset}.txt \
    --pdbdir data/pretrained/benchmark/${dataset} \
    --outdir outputs/seq_design \
    --config_path config/unZipro_pretrain.json \
    --param Models/unZipro_params.pt \
    --gpu 0 \
    --sampling_strategy argmax
```
#### 4. Batch evaluation on multiple benchmarks
To conveniently evaluate the model across multiple benchmark datasets (e.g., PDB50), you can run the provided shell script:
```sh
bash runs/evaluate_pretrained_model.sh
```