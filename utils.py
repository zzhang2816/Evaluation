import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Mydataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype = torch.float32)
        self.y=torch.tensor(y,dtype = torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,index):
        return self.X[index],self.y[index]

def generate_dataloaders(train_set,val_set,batch_size):
    train_data = Mydataset(train_set['x'],train_set['y'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    X, y=next(iter(train_loader))
    print(f'TRAIN: the shape of X: {X.shape}; the shape of y: {y.shape}')

    val_data = Mydataset(val_set['x'],val_set['y'])
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    X, y=next(iter(val_loader))
    print(f'VAL: the shape of X: {X.shape}; the shape of y: {y.shape}')
    return train_loader, val_loader
