"""Contrastive augmentations from the notebook."""
import random
import torch
from torch.utils.data import Dataset

def weak_augment(x):
    x=x.clone()
    if random.random() < .8: x=x + .02*torch.randn_like(x)
    if random.random() < .5:
        x=x*torch.empty((1,x.shape[1]),device=x.device).uniform_(.9,1.1)
    if random.random() < .4: x=torch.roll(x,random.randint(-5,5),dims=0)
    if random.random() < .3:
        c=random.randrange(x.shape[1])
        width=random.randint(5,14)
        start=random.randrange(max(1,x.shape[0]-width+1))
        x[start:start+width,c]=0
    return x

class SSLPairDataset(Dataset):
    def __init__(self,X): self.X=torch.as_tensor(X,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        x=self.X[i]
        return weak_augment(x), weak_augment(x)
