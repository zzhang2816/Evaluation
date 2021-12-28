import numpy as np
from src.utils import Mydataset,generate_dataloaders,train_model
from src.models import Model_v1, Model_v2
from src.my_parser import arg_parser
import wandb


if __name__ =="__main__":
    my_confg = arg_parser()
    if my_confg.use_wandb:
        wandb.login()
        wandb.init(project="Eval")
    train_set=np.load("dataset/train.npz")
    val_set=np.load("dataset/val.npz")
    # min_x, max_x = train_set['x'].min(axis=0), train_set['x'].max(axis=0)
    train_data = Mydataset(train_set, min_x = None, max_x = None)
    val_data = Mydataset(val_set, min_x = None, max_x = None)
    train_loader, val_loader=generate_dataloaders(train_data, val_data,my_confg.batch_size)
    if my_confg.model == "model_v1":
        net = Model_v1(my_confg)
    elif my_confg.model == "model_v2":
        net = Model_v2(my_confg)
    train_model(net,train_loader, val_loader, my_confg)

