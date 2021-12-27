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

```python3 train.py --num_epochs 5 --batch_size 16 --save_to_path "checkpoints/v1/"  v1 --num_hiddens 16 --l_dim 16```

Using the model v2
```python3 train.py [optional: --use_wandb] --num_epochs 10 --batch_size 16 --save_to_path "checkpoints/v2/" v2 --num_hiddens 16 --l_dim 16 --dropout 0 ```

### Hyperparameters Search 

Using the model v1

```python3 param_search.py --model model_v1```

Using the model v2

```python3 param_search.py --model model_v2```