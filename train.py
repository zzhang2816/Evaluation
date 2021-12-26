import numpy as np
from src.utils import Mydataset,generate_dataloaders
from src.models import Model_v2
from src.my_parser import arg_parser
import wandb

if __name__ =="__main__":
    my_confg = arg_parser()
    if my_confg.use_wandb:
        wandb.login()
        wandb.init(project="trial", config={'lr':0.001,'num_layers':2,'num_hiddens':8})

    train_set=np.load("dataset/train.npz")
    val_set=np.load("dataset/val.npz")
    train_data = Mydataset(train_set)
    val_data = Mydataset(val_set)
    train_loader, val_loader=generate_dataloaders(train_data, val_data,batch_size=32)
    net = Model_v2(my_confg)


