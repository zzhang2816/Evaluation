import numpy as np
from src.utils import Mydataset, pred
from src.models import Model_v1, Model_v2
from src.my_parser import arg_parser
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

def smape(A, F): # A: Actual, F: Forecast
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def score_model(y_true, y_pred):
    metric1 = mean_squared_error(y_true, y_pred, squared=False)
    metric2 = smape(y_true, y_pred)
    print(f'RMSE: {metric1}, SMAPE: {metric2}')

if __name__ =="__main__":
    my_confg = arg_parser()
    train_set=np.load("dataset/train.npz")
    val_set=np.load("dataset/val.npz")
    test_set=np.load("dataset/test.npz")

    # min_x, max_x = train_set['x'].min(axis=0), train_set['x'].max(axis=0)
    train_data = Mydataset(train_set, min_x=None, max_x =None, isTest=True)
    val_data = Mydataset(val_set, min_x=None, max_x =None, isTest=True)
    test_data = Mydataset(test_set, min_x=None, max_x =None, isTest=True)

    train_loader = DataLoader(dataset=train_data, batch_size=my_confg.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=my_confg.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=my_confg.batch_size, shuffle=False)

    if my_confg.model == "model_v1":
        net = Model_v1(my_confg)
    elif my_confg.model == "model_v2":
        net = Model_v2(my_confg)
    net.load_state_dict(torch.load(my_confg.load_from_path,map_location=torch.device('cpu')))

    y_true_train = train_set['y'].reshape(-1)
    y_true_val = val_set['y'].reshape(-1)
    train_predictions = pred(net, train_loader).reshape(-1)
    val_predictions = pred(net, val_loader).reshape(-1)
    test_predictions = pred(net, test_loader).reshape(-1)

    print("Train:")
    score_model(y_true_train, train_predictions)
    print("Val:")
    score_model(y_true_val, val_predictions)
    np.savez("pred.npz", y=test_predictions)
