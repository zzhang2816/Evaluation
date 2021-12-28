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
- train.py (main entrance)
- param_search.py (sweep 30 different configurations)
- confgs/ (store the configuations)

### Baseline [Autogulon] 

```
# Mac/Linux
python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel
python3 -m pip install -U "mxnet<2.0.0"
python3 -m pip install autogluon
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

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |