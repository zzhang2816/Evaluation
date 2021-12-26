import torch
from torch import nn
import os
import wandb
from torch.utils.data import Dataset,DataLoader

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


def train(net,train_loader, val_loader, my_confg):
        if my_confg.load_from_path:
                net.load_state_dict(torch.load(my_confg.load_from_path))
        if not os.path.isdir(my_confg.save_to_path):
            os.mkdir(my_confg.save_to_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',factor=0.5, verbose = True, min_lr=1e-6, patience = 5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)

        # saved_name=f'fold {fold_i}.pt'
        # early_stopping = EarlyStopping(model_save,saved_name,patience = 10, verbose=True)

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
                for batch_idx, (X,y) in enumerate(loader): 
                    X=X.to(device)
                    y=y.to(device)

                    y_hat=net(X)
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
                
                # print(f'Epoch: {epoch}; Avg_loss: {avg_epoch_loss}')
            if my_confg.use_wandb:
                wandb.log({"train_loss": epoch_trainlosses[-1],"val":epoch_vallosses[-1]})

            # epoch_valloss=epoch_vallosses[-1]
            scheduler.step()        
            # early_stopping(epoch_valloss, model)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            if epoch%10 == 0:
                torch.save(net.state_dict(), my_confg.save_to_path+f"{epoch}.pt")
        if epoch%10!=0:
            torch.save(net.state_dict(), my_confg.save_to_path+f"{my_confg.num_epochs}.pt")
        if my_confg.use_wandb:
            wandb.finish()
        log_loss(epoch_trainlosses, epoch_vallosses)
        