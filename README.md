# Evaluation

### Repository Structure
- checkpoints
- dataset
- debug
- src
- baseline.ipynb
- train.py

### Train
Using the model v1
python3 train.py --num_epochs 5 --batch_size 16 --save_to_path "checkpoints/v1/"  v1 --num_hiddens 16 --l_dim 16

Using the model v2
python3 train.py [optional: --use_wandb] --num_epochs 10 --batch_size 16 --save_to_path "checkpoints/v2/" v2 --num_hiddens 16 --l_dim 16 --dropout 0 

python3 param_search.py --model model_v1