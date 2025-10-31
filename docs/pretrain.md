# unZipro pre-training documentation
### Pretrain on PDB50 datasets
#### 1. Download full PDB50 dataset
```python
python script/fetch_PDB_parallel.py -i data/pretrained/all_PDB50.txt -o data/pretrained/PDB -m RCSB -cpu 8
```
#### 2. Start pre-training
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
```dataset=TS50 && python script/unZipro_evaluate.py --project_name unZipro_${dataset}_test \
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