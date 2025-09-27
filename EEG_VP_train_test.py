import numpy as np
import matplotlib.pyplot as plt
import time

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torch.nn.parameter import Parameter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import models

###################################### Hyperparameters ###################################
batch_size = 256
num_epochs = 100
lr = 0.001   # learning rate
C = 62       # the number of channels
T = 5        # the time samples of EEG signals
network_name = "GLMNet_mlp"
output_dir = "/content/drive/MyDrive/EEG2Video_outputs/models"
os.makedirs(output_dir, exist_ok=True)
saved_model_path = os.path.join(output_dir, network_name + "_40c.pth")

run_device = "cuda"

##########################################################################################

def my_normalize(data_array):
    data_array = data_array.reshape(data_array.shape[0], C*T)
    normalize = StandardScaler()
    normalize.fit(data_array)
    return (normalize.transform(data_array)).reshape(data_array.shape[0], C, T)

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))

def Get_subject(f):
    if(f[1] == '_'):
        return int(f[0])
    return int(f[0])*10 + int(f[1])

def Get_Dataloader(datat, labelt, istrain, batch_size):
    features = torch.tensor(datat, dtype=torch.float32)
    labels = torch.tensor(labelt, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=istrain)

class Accumulator:  #@save
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        return sum(self.times) / len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

def cal_accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = (y_hat == y)
    return torch.sum(cmp, dim=0)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(cal_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def topk_accuracy(output, target, topk=(1, )):       
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res

# 训练函数
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, save_path):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)
    loss = nn.CrossEntropyLoss()
    
    timer, num_bathces = Timer(), len(train_iter)
    best_val_acc, best_test_acc = 0, 0
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], cal_accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        val_acc = evaluate_accuracy_gpu(net, val_iter)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net, save_path)
        
        if epoch % 3 == 0:        
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, val acc {val_acc:.3f}, test acc {test_acc:.3f}')
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    return best_val_acc   

# =====================================================
# Label handling: generate directly instead of GT_label
# =====================================================
# Each block = 40 classes × 10 clips = 400 samples
# So labels = repeat 0–39 for each block
All_label = np.tile(np.arange(40).repeat(10), 7).reshape(7, 400)

def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s/")

All_sub_top1 = []
All_sub_top5 = []

for subname in sub_list:
    load_npy = np.load(os.path.join("/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s", subname))

    print("Loaded:", subname, load_npy.shape)

    all_test_label = np.array([])
    all_test_pred = np.array([])

    All_train = rearrange(load_npy, 'a b c d e f -> a (b c d) e f')
    print("Reshaped:", All_train.shape)

    Top_1 = []
    Top_K = []

    for test_set_id in range(7):
        val_set_id = (test_set_id - 1) % 7
        train_data = np.empty((0, 62, 5))
        train_label = np.empty((0))
        for i in range(7):
            if i == test_set_id:
                continue
            train_data = np.concatenate((train_data, All_train[i].reshape(400, 62, 5)))
            train_label = np.concatenate((train_label, All_label[i]))
        test_data = All_train[test_set_id]
        test_label = All_label[test_set_id]
        val_data = All_train[val_set_id]
        val_label = All_label[val_set_id]

        train_data = train_data.reshape(train_data.shape[0], 62*5)
        test_data = test_data.reshape(test_data.shape[0], 62*5)
        val_data = val_data.reshape(val_data.shape[0], 62*5)

        # Normalize
        normalize = StandardScaler()
        normalize.fit(train_data)
        train_data = normalize.transform(train_data)
        test_data = StandardScaler().fit_transform(test_data)
        val_data  = StandardScaler().fit_transform(val_data)
            
        modelnet = models.glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310)

        norm_train_data = train_data.reshape(train_data.shape[0], C, T)
        norm_test_data  = test_data.reshape(test_data.shape[0], C, T)
        norm_val_data   = val_data.reshape(val_data.shape[0], C, T)
        train_iter = Get_Dataloader(norm_train_data, train_label, istrain=True, batch_size=batch_size)
        test_iter  = Get_Dataloader(norm_test_data, test_label, istrain=False, batch_size=batch_size)
        val_iter   = Get_Dataloader(norm_val_data, val_label, istrain=False, batch_size=batch_size)

        accu = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device, save_path=saved_model_path)
        
        print("acc : =", accu)

        loaded_model = torch.load(saved_model_path).to(run_device)

        block_top_1, block_top_k = [], []
        for X, y in test_iter:
            X, y = X.to(run_device), y.to(run_device)
            y_hat = loaded_model(X)
            top_K_results = topk_accuracy(y_hat, y, topk=(1,5))
            block_top_1.append(top_K_results[0])
            block_top_k.append(top_K_results[1])
            y_hat = torch.argmax(y_hat, axis=1).cpu().numpy().reshape(-1)
            all_test_pred = np.concatenate((all_test_pred, y_hat))    
        all_test_label = np.concatenate((all_test_label, test_label))

        print("top1_acc = ", np.mean(block_top_1))
        print("top5_acc = ", np.mean(block_top_k))
        Top_1.append(np.mean(block_top_1))
        Top_K.append(np.mean(block_top_k))

    print(metrics.classification_report(all_test_label, all_test_pred))

    All_sub_top1.append(np.mean(Top_1))
    All_sub_top5.append(np.mean(Top_K))

    out_dir = "/content/drive/MyDrive/EEG2Video_outputs/classification_results"
    os.makedirs(out_dir, exist_ok=True)
    save_results = np.vstack([all_test_pred, all_test_label])
    np.save(os.path.join(out_dir, network_name+'_Predict_Label_' + subname + '.npy'), save_results)

print("TOP1: ", np.mean(All_sub_top1), np.std(All_sub_top1))
print("TOP5: ", np.mean(All_sub_top5), np.std(All_sub_top5))

np.save('./ClassificationResults/40c_top1/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top1))
np.save('./ClassificationResults/40c_top5/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top5))
