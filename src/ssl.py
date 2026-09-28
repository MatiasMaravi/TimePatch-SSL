"""NT-Xent self-supervised pretraining."""
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .augmentations import SSLPairDataset

def nt_xent(z1,z2,temp=.2):
    B=z1.size(0); z=torch.cat([z1,z2],0)
    sim=F.cosine_similarity(z[:,None,:],z[None,:,:],dim=-1)/temp
    sim.fill_diagonal_(-1e9)
    target=torch.cat([torch.arange(B,device=z.device)+B,torch.arange(B,device=z.device)])
    return F.cross_entropy(sim,target)

def pretrain_ssl(model,X,device,epochs=6,batch_size=64,lr=1e-3,weight_decay=1e-4,temp=.2):
    dl=DataLoader(SSLPairDataset(X),batch_size=batch_size,shuffle=True,drop_last=True)
    model=model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    for e in range(epochs):
        model.train(); losses=[]
        for a,b in dl:
            a,b=a.to(device),b.to(device)
            loss=nt_xent(model.forward_ssl(a),model.forward_ssl(b),temp)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"SSL epoch {e+1}/{epochs}: {np.mean(losses):.4f}")
    return model
