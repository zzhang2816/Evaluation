import torch
from torch import nn
import os
import wandb
from torch.utils.data import Dataset,DataLoader
import numpy as np
import datetime

from wandb.sdk.lib import disabled
from .models import Model_v1, Model_v2
import argparse

class Mydataset(Dataset):
    def __init__(self,dataset):
        self.X=torch.tensor(dataset['x'],dtype = torch.float32)
        self.y=torch.tensor(dataset['y'],dtype = torch.float32)
        self.locations=torch.tensor(dataset['locations'],dtype = torch.int)
        self.times=torch.tensor(dataset['times'],dtype = torch.int)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,index):
        return self.X[index],self.y[index],self.locations[index],self.times[index]


def generate_dataloaders(train_data,val_data,batch_size):
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    X, y,locations,times=next(iter(train_loader))
    print(f'TRAIN: the shape of X: {X.shape}; the shape of y: {y.shape}\
the shape of locations: {locations.shape};the shape of times: {times.shape};')

    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    X, y,locations,times=next(iter(val_loader))
    print(f'VAL: the shape of X: {X.shape}; the shape of y: {y.shape}\
the shape of locations: {locations.shape};the shape of times: {times.shape};')
    return train_loader, val_loader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,saved_name,patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path=saved_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)                 
        self.val_loss_min = val_loss


def train_model(net,train_loader, val_loader, my_confg):
        if my_confg.load_from_path:
                net.load_state_dict(torch.load(my_confg.load_from_path))
        if not os.path.isdir(my_confg.save_to_path):
            os.mkdir(my_confg.save_to_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=my_confg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=my_confg.weight_decay)
        
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',factor=0.5, verbose = True, min_lr=1e-6, patience = 5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)

        save_name = datetime.datetime.now().strftime("%m_%d %H_%M")
        early_stopping = EarlyStopping(my_confg.save_to_path+f"{save_name}.pt",patience = 10, verbose=True)

        net.to(device)
        
        epoch_trainlosses=[]
        epoch_vallosses=[]
        for epoch in range(my_confg.num_epochs):
            for (dataset, loader) in [("train", train_loader), ("val", val_loader)]: 
                if dataset == "train":
                        torch.set_grad_enabled(True)
                        net.train()
                else:
                        torch.set_grad_enabled(False)
                        net.eval()
                total_epoch_loss = 0
                for batch_idx, (X,y,locations,times) in enumerate(loader): 
                    X=X.to(device)
                    y=y.to(device)
                    locations=locations.to(device)
                    times=times.to(device)

                    y_hat=net(X,locations,times)
                    l=loss(y_hat,y)
                    
                    total_epoch_loss += l.cpu().detach().numpy()*X.shape[0]
                    if(batch_idx%100==0):
                        message=""
                        message += f"Epoch {epoch+1}/{my_confg.num_epochs} {dataset} progress: {int((batch_idx / len(loader)) * 100)}% "
                        message += f'loss: {l.data.item():.4f}'
                        print(message)

                    if dataset == "train" :
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
          
                avg_epoch_loss = total_epoch_loss/ len(loader.dataset)
                if dataset == "train" :
                    epoch_trainlosses.append(avg_epoch_loss)
                if dataset == 'val':
                    epoch_vallosses.append(avg_epoch_loss)
                
            if my_confg.use_wandb:
                wandb.log({"train_loss": epoch_trainlosses[-1],"val_loss":epoch_vallosses[-1]})
            else:
                print(f'Epoch: {epoch}; train_loss: {epoch_trainlosses[-1]}; val_loss: {epoch_vallosses[-1]}')

            scheduler.step()        
            early_stopping(epoch_vallosses[-1], net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        if my_confg.use_wandb:
            wandb.finish()
        log_loss(epoch_trainlosses, epoch_vallosses, my_confg, save_name)


def log_loss(epoch_trainlosses, epoch_vallosses,my_confg,save_name):
    save_path = my_confg.save_to_path+save_name
    start_time = save_path.split('/')[-1]
    f = open(f"{save_path}.txt","w")
    finish_time = datetime.datetime.now().strftime("%m_%d %H_%M")
    log = f'Training time: {start_time} --> {finish_time}\n'
    log += f'Total epoch: {len(epoch_trainlosses)}\n'
    log += f'Config: {str(my_confg)}\n'
    f.write(log)
    for i, (train_l, val_l) in enumerate(zip(epoch_trainlosses, epoch_vallosses)):
        f.write(f'Epoch {i}, train_loss: {train_l:.4f}, val_loss: {val_l:.4f}\n')
    f.close()


def sweep_train():
        wandb.init()
        my_confg = wandb.config
        my_confg = argparse.Namespace(**my_confg)
        train_set=np.load("dataset/train.npz")
        val_set=np.load("dataset/val.npz")
        train_data = Mydataset(train_set)
        val_data = Mydataset(val_set)
        train_loader, val_loader=generate_dataloaders(train_data, val_data,my_confg.batch_size)
        if my_confg.model == "model_v1":
            net = Model_v1(my_confg)
        elif my_confg.model == "model_v2":
            net = Model_v2(my_confg)

        if my_confg.load_from_path:
                net.load_state_dict(torch.load(my_confg.load_from_path))
        if not os.path.isdir(my_confg.save_to_path):
            os.mkdir(my_confg.save_to_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=my_confg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',factor=0.5, verbose = True, min_lr=1e-6, patience = 5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)

        save_name = datetime.datetime.now().strftime("%m_%d %H_%M")
        early_stopping = EarlyStopping(my_confg.save_to_path+f"{save_name}.pt",patience = 10, verbose=True)

        net.to(device)
        
        epoch_trainlosses=[]
        epoch_vallosses=[]
        for epoch in range(my_confg.num_epochs):
            for (dataset, loader) in [("train", train_loader), ("val", val_loader)]: 
                if dataset == "train":
                        torch.set_grad_enabled(True)
                        net.train()
                else:
                        torch.set_grad_enabled(False)
                        net.eval()
                total_epoch_loss = 0
                for batch_idx, (X,y,locations,times) in enumerate(loader): 
                    X=X.to(device)
                    y=y.to(device)
                    locations=locations.to(device)
                    times=times.to(device)

                    y_hat=net(X,locations,times)
                    l=loss(y_hat,y)
                    
                    total_epoch_loss += l.cpu().detach().numpy()*X.shape[0]
                    if(batch_idx%100==0):
                        message=""
                        message += f"Epoch {epoch+1}/{my_confg.num_epochs} {dataset} progress: {int((batch_idx / len(loader)) * 100)}% "
                        message += f'loss: {l.data.item():.4f}'
                        print(message)

                    if dataset == "train" :
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
          
                avg_epoch_loss = total_epoch_loss/ len(loader.dataset)
                if dataset == "train" :
                    epoch_trainlosses.append(avg_epoch_loss)
                if dataset == 'val':
                    epoch_vallosses.append(avg_epoch_loss)

            wandb.log({"train_loss": epoch_trainlosses[-1],"val_loss":epoch_vallosses[-1]})
            print(f'Epoch: {epoch}; train_loss: {epoch_trainlosses[-1]}; val_loss: {epoch_vallosses[-1]}')

            scheduler.step()        
            early_stopping(epoch_vallosses[-1], net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        log_loss(epoch_trainlosses, epoch_vallosses,my_confg,save_name)
