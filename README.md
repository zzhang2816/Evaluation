# Evaluation

### Repository Structure
- checkpoints/ (path to store the model parameters)
- dataset/
- debug/ (store jupyter notebooks for debugging)
- src/
  - models.py 
  - my_parser.py
  - utils.py (functions: train_model, generate_dataloader)
- baseline.ipynb
- inference.py (evaluate the model)
- train.py (main entrance)
- param_search.py (sweep 30 different configurations)
- confgs/ (store the configuations)
- myjob.condor (submit job to [HTGC2](https://cslab.cs.cityu.edu.hk/services/high-throughput-gpu-cluster-2-htgc2))
- train.sh (run param_search.py, this requires wandb login)

### Baseline [Autogulon] 

```
# Mac/Linux
python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel
python3 -m pip install -U "mxnet<2.0.0"
python3 -m pip install autogluon
```

### Installment

```
pip install -r requirements.txt
```

### Train

Using the model v1
```python3 train.py [optional: --use_wandb] --num_epochs 500 --batch_size 64 --save_to_path checkpoints/v1/ --weight_decay 1e-4 --lr 0.001 v1 --num_hiddens 128 --num_layers 2 --GRU_dropout 0.05 --l_dim 16 --embed_loc_size 10 --embed_time_size 10```

Using the model v2
```python3 train.py [optional: --use_wandb] --num_epochs 500 --batch_size 64 --save_to_path checkpoints/v2/ --weight_decay 1e-4 --lr 3e-4 v2 --num_hiddens 128 --num_layers 2 --GRU_dropout 0.1 --l_dim 16 --embed_loc_size 16 --embed_time_size 10```

### Hyperparameters Search 

Using the model v1
```python3 param_search.py --model model_v1```

Using the model v2
```python3 param_search.py --model model_v2```

### Inference
Using the model v1
```python3 inference.py --batch_size 64 --load_from_path "checkpoints/v1/12_27 22_34.pt" v1 --num_hiddens 128 --num_layers 2 --GRU_dropout 0.05 --l_dim 16 --embed_loc_size 10 --embed_time_size 10```

Using the model v2
```python3 inference.py --batch_size 64 --load_from_path "checkpoints/v2/12_28 12_22.pt" v2 --num_hiddens 128 --num_layers 2 --GRU_dropout 0.1 --l_dim 16 --embed_loc_size 16 --embed_time_size 10```

### Performance

Train
| First Header  | RMSE | SMAPE |
| ------------- | ------------- | ------------- |
| baseline  | --  | --  |
| v1  |  3.165555928013089 | 79.2556911952384  |
| v2  |  3.257202415292994 | 78.6829463664590  |

Val
| First Header  | RMSE | SMAPE |
| ------------- | ------------- | ------------- |
| baseline  |  5.629683235612225 | 46.20446178698806  |
| v1  |  4.131460239228742 | 78.59140156419419  |
| v2  |  4.130958288532827 | 77.64709082245047  |