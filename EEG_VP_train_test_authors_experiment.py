import numpy as np
import matplotlib.pyplot as plt
import time

import os
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
from tqdm import tqdm
import models

###################################### Hyperparameters ###################################
batch_size = 256
num_epochs = 100
lr = 0.001   # learning rate
C = 62       # number of channels
T = 5        # time samples of EEG signals
output_dir = '/content/drive/MyDrive/EEG2Video_outputs'
os.makedirs(output_dir, exist_ok=True)
network_name = "GLMNet_mlp"
saved_model_path = os.path.join(output_dir, network_name + '_40c.pth')

run_device = "cuda"
##########################################################################################

def my_normalize(data_array):
    data_array = data_array.reshape(data_array.shape[0], C*T)
    normalize = StandardScaler()
    normalize.fit(data_array)
    return (normalize.transform(data_array)).reshape(data_array.shape[0], C, T)

def Get_Dataloader(datat, labelt, istrain, batch_size):
    features = torch.tensor(datat, dtype=torch.float32)
    labels = torch.tensor(labelt, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=istrain)

class Accumulator:
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
    def sum(self):
        return sum(self.times)

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

def topk_accuracy(output, target, topk=(1,)):
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

def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, save_path):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)
    loss = nn.CrossEntropyLoss()

    timer, num_batches = Timer(), len(train_iter)
    best_val_acc = 0

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
    return best_val_acc

# Ground truth labels
GT_label = np.array([
    [23,22,9,6,18,14,5,36,25,19,28,35,3,16,24,40,15,27,38,33,34,4,39,17,1,26,20,29,13,32,37,2,11,12,30,31,8,21,7,10],
    [27,33,22,28,31,12,38,4,18,17,35,39,40,5,24,32,15,13,2,16,34,25,19,30,23,3,8,29,7,20,11,14,37,6,21,1,10,36,26,9],
    [15,36,31,1,34,3,37,12,4,5,21,24,14,16,39,20,28,29,18,32,2,27,8,19,13,10,30,40,17,26,11,9,33,25,35,7,38,22,23,6],
    [16,28,23,1,39,10,35,14,19,27,37,31,5,18,11,25,29,13,20,24,7,34,26,4,40,12,8,22,21,30,17,2,38,9,3,36,33,6,32,15],
    [18,29,7,35,22,19,12,36,8,15,28,1,34,23,20,13,37,9,16,30,2,33,27,21,14,38,10,17,31,3,24,39,11,32,4,25,40,5,26,6],
    [29,16,1,22,34,39,24,10,8,35,27,31,23,17,2,15,25,40,3,36,26,6,14,37,9,12,19,30,5,28,32,4,13,18,21,20,7,11,33,38],
    [38,34,40,10,28,7,1,37,22,9,16,5,12,36,20,30,6,15,35,2,31,26,18,24,8,3,23,19,14,13,21,4,25,11,32,17,39,29,33,27]
])
GT_label = GT_label - 1
All_label = np.concatenate([GT_label[b].repeat(10).reshape(1, 400) for b in range(7)], axis=0)

# Subjects
sub_list = ["sub10.npy"]

All_sub_top1 = []
All_sub_top5 = []

for subname in sub_list:
    de_npy  = np.load(f"/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s_authors/{subname}")
    psd_npy = np.load(f"/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s_authors/{subname}")

    All_train_de  = rearrange(de_npy,  'a b c d e f -> a (b c d) e f')
    All_train_psd = rearrange(psd_npy, 'a b c d e f -> a (b c d) e f')

    Top_1, Top_K = [], []
    all_test_label, all_test_pred = np.array([]), np.array([])

    for test_set_id in tqdm(range(7), desc=f"Subject {subname} folds"):
        val_set_id = (test_set_id - 1) % 7

        # ===== DE =====
        train_de = np.concatenate([All_train_de[i].reshape(400, 62, 5) for i in range(7) if i != test_set_id])
        test_de, val_de = All_train_de[test_set_id], All_train_de[val_set_id]

        # ===== PSD =====
        train_psd = np.concatenate([All_train_psd[i].reshape(400, 62, 5) for i in range(7) if i != test_set_id])
        test_psd, val_psd = All_train_psd[test_set_id], All_train_psd[val_set_id]

        # labels
        train_label = np.concatenate([All_label[i] for i in range(7) if i != test_set_id])
        test_label, val_label = All_label[test_set_id], All_label[val_set_id]

        # reshape + normalize DE
        train_de, test_de, val_de = [x.reshape(x.shape[0], 62*5) for x in [train_de, test_de, val_de]]
        scaler = StandardScaler().fit(train_de)
        train_de, test_de, val_de = scaler.transform(train_de), scaler.transform(test_de), scaler.transform(val_de)

        # reshape + normalize PSD
        train_psd, test_psd, val_psd = [x.reshape(x.shape[0], 62*5) for x in [train_psd, test_psd, val_psd]]
        scaler = StandardScaler().fit(train_psd)
        train_psd, test_psd, val_psd = scaler.transform(train_psd), scaler.transform(test_psd), scaler.transform(val_psd)

        # dataloaders
        train_iter_de = Get_Dataloader(train_de.reshape(train_de.shape[0], C, T), train_label, True, batch_size)
        test_iter_de  = Get_Dataloader(test_de.reshape(test_de.shape[0], C, T), test_label, False, batch_size)
        val_iter_de   = Get_Dataloader(val_de.reshape(val_de.shape[0], C, T), val_label, False, batch_size)

        train_iter_psd = Get_Dataloader(train_psd.reshape(train_psd.shape[0], C, T), train_label, True, batch_size)
        test_iter_psd  = Get_Dataloader(test_psd.reshape(test_psd.shape[0], C, T), test_label, False, batch_size)
        val_iter_psd   = Get_Dataloader(val_psd.reshape(val_psd.shape[0], C, T), val_label, False, batch_size)

        # Train DE model
        print("\n🔵 Training DE model")
        model_de = models.glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310)
        train(model_de, train_iter_de, val_iter_de, test_iter_de, num_epochs, lr, run_device, save_path=saved_model_path+"_de")
        model_de = torch.load(saved_model_path+"_de", weights_only=False).to(run_device)

        # Train PSD model
        print("🟢 Training PSD model")
        model_psd = models.glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310)
        train(model_psd, train_iter_psd, val_iter_psd, test_iter_psd, num_epochs, lr, run_device, save_path=saved_model_path+"_psd")
        model_psd = torch.load(saved_model_path+"_psd", weights_only=False).to(run_device)

        # Evaluate with combined probabilities
        block_top_1, block_top_k = [], []
        for (X_de, y), (X_psd, _) in zip(test_iter_de, test_iter_psd):
            X_de, y, X_psd = X_de.to(run_device), y.to(run_device), X_psd.to(run_device)
            probs_de  = torch.softmax(model_de(X_de), dim=1)
            probs_psd = torch.softmax(model_psd(X_psd), dim=1)
            combined  = (probs_de + probs_psd) / 2
            top_K_results = topk_accuracy(combined, y, topk=(1,5))
            block_top_1.append(top_K_results[0])
            block_top_k.append(top_K_results[1])
            preds = torch.argmax(combined, axis=1).cpu().numpy()
            all_test_pred = np.concatenate((all_test_pred, preds))
        all_test_label = np.concatenate((all_test_label, test_label))

        print(f"Fold {test_set_id} → Top1: {np.mean(block_top_1):.3f}, Top5: {np.mean(block_top_k):.3f}")
        Top_1.append(np.mean(block_top_1))
        Top_K.append(np.mean(block_top_k))

    print(metrics.classification_report(all_test_label, all_test_pred))
    print("Mean Top1:", np.mean(Top_1), "Mean Top5:", np.mean(Top_K))
    All_sub_top1.append(np.mean(Top_1))
    All_sub_top5.append(np.mean(Top_K))

print(All_sub_top1)
print(All_sub_top5)
print("TOP1:", np.mean(np.array(All_sub_top1)), np.std(np.array(All_sub_top1)))
print("TOP5:", np.mean(np.array(All_sub_top5)), np.std(np.array(All_sub_top5)))
