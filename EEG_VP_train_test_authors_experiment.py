# ==========================================
# DE + PSD + Segments classification with probability fusion
# ==========================================
import numpy as np
import time, os
import torch
from torch import nn
from torch.utils import data
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from einops import rearrange
from tqdm import tqdm
import models

# === Hyperparameters ===
batch_size    = 256
num_epochs    = 100
lr            = 0.001
C             = 62
T             = 5
output_dir    = '/content/drive/MyDrive/EEG2Video_outputs'
os.makedirs(output_dir, exist_ok=True)
network_name  = "GLMNet_fusion"
saved_model_path = os.path.join(output_dir, network_name + '_40c.pth')
run_device    = "cuda"

# === Utils ===
def Get_Dataloader(datat, labelt, istrain, batch_size):
    features = torch.tensor(datat, dtype=torch.float32)
    labels   = torch.tensor(labelt, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=istrain)

class Accumulator:
    def __init__(self, n): self.data = [0.0]*n
    def add(self,*args): self.data=[a+float(b) for a,b in zip(self.data,args)]
    def __getitem__(self,idx): return self.data[idx]

def cal_accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=torch.argmax(y_hat,axis=1)
    return torch.sum(y_hat==y,dim=0)

def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if not device: device=next(iter(net.parameters())).device
    metric=Accumulator(2)
    for X,y in data_iter:
        X,y=X.to(device),y.to(device)
        metric.add(cal_accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def topk_accuracy(output,target,topk=(1,)):
    with torch.no_grad():
        maxk=max(topk); bs=target.size(0)
        _,pred=output.topk(maxk,1,True,True); pred=pred.t()
        correct=pred.eq(target.view(1,-1).expand_as(pred))
        res=[]
        for k in topk:
            correct_k=correct[:k].reshape(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(1.0/bs).item())
        return res

def train(net,train_iter,val_iter,test_iter,num_epochs,lr,device,save_path):
    net.apply(lambda m: nn.init.xavier_uniform_(m.weight) 
              if isinstance(m,(nn.Linear,nn.Conv2d)) else None)
    net.to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0)
    loss=nn.CrossEntropyLoss()
    best_val=0
    for epoch in range(num_epochs):
        metric=Accumulator(3)
        net.train()
        for X,y in train_iter:
            optim.zero_grad(); X,y=X.to(device),y.to(device)
            out=net(X); l=loss(out,y); l.backward(); optim.step()
            metric.add(l*X.shape[0],cal_accuracy(out,y),X.shape[0])
        val_acc=evaluate_accuracy_gpu(net,val_iter)
        if val_acc>best_val:
            best_val=val_acc; torch.save(net,save_path)
        if epoch%3==0: print(f"[{epoch}] loss {metric[0]/metric[2]:.3f} val {val_acc:.3f}")
    return best_val

# === Labels ===
GT_label=np.array([...])  # (same as your current GT_label definition)
GT_label=GT_label-1
All_label=np.concatenate([GT_label[b].repeat(10).reshape(1,400) for b in range(7)],axis=0)

# === Subjects ===
sub_list=["sub10.npy"]

All_sub_top1,All_sub_top5=[],[]

for subname in sub_list:
    de_npy  = np.load(f"/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s_authors/{subname}")
    psd_npy = np.load(f"/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s_authors/{subname}")
    seg_npy = np.load(f"/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments_authors/{subname}")

    All_train_de  = rearrange(de_npy,  'a b c d e f -> a (b c d) e f')
    All_train_psd = rearrange(psd_npy, 'a b c d e f -> a (b c d) e f')
    All_train_seg = rearrange(seg_npy, 'a b c d e f -> a (b c d) e f')

    Top_1,Top_K=[],[]
    all_test_label,all_test_pred=np.array([]),np.array([])

    for test_set_id in tqdm(range(7),desc=f"Subject {subname} folds"):
        val_set_id=(test_set_id-1)%7

        # ===== DE =====
        train_de=np.concatenate([All_train_de[i].reshape(400,62,5) for i in range(7) if i!=test_set_id])
        test_de,val_de=All_train_de[test_set_id],All_train_de[val_set_id]

        # ===== PSD =====
        train_psd=np.concatenate([All_train_psd[i].reshape(400,62,5) for i in range(7) if i!=test_set_id])
        test_psd,val_psd=All_train_psd[test_set_id],All_train_psd[val_set_id]

        # ===== Segments ===== (62,400)
        train_seg=np.concatenate([All_train_seg[i].reshape(400,62,400) for i in range(7) if i!=test_set_id])
        test_seg,val_seg=All_train_seg[test_set_id],All_train_seg[val_set_id]

        # labels
        train_label=np.concatenate([All_label[i] for i in range(7) if i!=test_set_id])
        test_label,val_label=All_label[test_set_id],All_label[val_set_id]

        # === Normalize DE/PSD ===
        def norm_data(train,test,val,feat_dim):
            train=train.reshape(train.shape[0],feat_dim)
            test =test.reshape(test.shape[0],feat_dim)
            val  =val.reshape(val.shape[0],feat_dim)
            scaler=StandardScaler().fit(train)
            return (scaler.transform(train),
                    scaler.transform(test),
                    scaler.transform(val))
        train_de,test_de,val_de=norm_data(train_de,test_de,val_de,62*5)
        train_psd,test_psd,val_psd=norm_data(train_psd,test_psd,val_psd,62*5)
        train_seg,test_seg,val_seg=norm_data(train_seg,test_seg,val_seg,62*400)

        # === Reshape back ===
        train_iter_de =Get_Dataloader(train_de.reshape(train_de.shape[0],C,T),train_label,True,batch_size)
        test_iter_de  =Get_Dataloader(test_de.reshape(test_de.shape[0],C,T),test_label,False,batch_size)
        val_iter_de   =Get_Dataloader(val_de.reshape(val_de.shape[0],C,T),val_label,False,batch_size)

        train_iter_psd=Get_Dataloader(train_psd.reshape(train_psd.shape[0],C,T),train_label,True,batch_size)
        test_iter_psd =Get_Dataloader(test_psd.reshape(test_psd.shape[0],C,T),test_label,False,batch_size)
        val_iter_psd  =Get_Dataloader(val_psd.reshape(val_psd.shape[0],C,T),val_label,False,batch_size)

        train_iter_seg=Get_Dataloader(train_seg.reshape(train_seg.shape[0],1,C,400),train_label,True,batch_size)
        test_iter_seg =Get_Dataloader(test_seg.reshape(test_seg.shape[0],1,C,400),test_label,False,batch_size)
        val_iter_seg  =Get_Dataloader(val_seg.reshape(val_seg.shape[0],1,C,400),val_label,False,batch_size)

        # === Train models ===
        print("\n🔵 Training DE model")
        model_de=models.glfnet_mlp(out_dim=40,emb_dim=64,input_dim=310)
        train(model_de,train_iter_de,val_iter_de,test_iter_de,num_epochs,lr,run_device,saved_model_path+"_de")
        model_de=torch.load(saved_model_path+"_de",weights_only=False).to(run_device)

        print("🟢 Training PSD model")
        model_psd=models.glfnet_mlp(out_dim=40,emb_dim=64,input_dim=310)
        train(model_psd,train_iter_psd,val_iter_psd,test_iter_psd,num_epochs,lr,run_device,saved_model_path+"_psd")
        model_psd=torch.load(saved_model_path+"_psd",weights_only=False).to(run_device)

        print("🔴 Training Segments model")
        model_seg=models.glfnet(out_dim=40)   # conv net
        train(model_seg,train_iter_seg,val_iter_seg,test_iter_seg,num_epochs,lr,run_device,saved_model_path+"_seg")
        model_seg=torch.load(saved_model_path+"_seg",weights_only=False).to(run_device)

        # === Fusion ===
        block_top_1,block_top_k=[],[]
        for (Xd,y),(Xp,_),(Xs,_) in zip(test_iter_de,test_iter_psd,test_iter_seg):
            Xd,Xp,Xs,y=Xd.to(run_device),Xp.to(run_device),Xs.to(run_device),y.to(run_device)
            probs_de =torch.softmax(model_de(Xd),dim=1)
            probs_psd=torch.softmax(model_psd(Xp),dim=1)
            probs_seg=torch.softmax(model_seg(Xs),dim=1)
            combined=(probs_de+probs_psd+probs_seg)/3
            topk_res=topk_accuracy(combined,y,topk=(1,5))
            block_top_1.append(topk_res[0]); block_top_k.append(topk_res[1])
            preds=torch.argmax(combined,axis=1).cpu().numpy()
            all_test_pred=np.concatenate((all_test_pred,preds))
        all_test_label=np.concatenate((all_test_label,test_label))

        print(f"Fold {test_set_id} → Top1 {np.mean(block_top_1):.3f}, Top5 {np.mean(block_top_k):.3f}")
        Top_1.append(np.mean(block_top_1)); Top_K.append(np.mean(block_top_k))

    print(metrics.classification_report(all_test_label,all_test_pred))
    print("Mean Top1:",np.mean(Top_1),"Mean Top5:",np.mean(Top_K))
    All_sub_top1.append(np.mean(Top_1)); All_sub_top5.append(np.mean(Top_K))

print("TOP1:",np.mean(All_sub_top1),np.std(All_sub_top1))
print("TOP5:",np.mean(All_sub_top5),np.std(All_sub_top5))
