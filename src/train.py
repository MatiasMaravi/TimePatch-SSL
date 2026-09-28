"""End-to-end supervised fine-tuning with validation-F1 model selection."""
import copy, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .datasets import WindowDataset
from .evaluate import evaluate_probs

def predict_probs(model,X,device,batch_size=128):
    dl=DataLoader(WindowDataset(X),batch_size=batch_size,shuffle=False)
    model.eval(); out=[]
    with torch.no_grad():
        for xb in dl:
            out.append(torch.softmax(model(xb.to(device)),dim=1)[:,1].cpu().numpy())
    return np.concatenate(out)

def finetune(model,X_train,y_train,X_val,y_val,device,epochs=8,batch_size=64,
             lr=2e-4,weight_decay=1e-4):
    dl=DataLoader(WindowDataset(X_train,y_train),batch_size=batch_size,shuffle=True)
    model=model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn=nn.CrossEntropyLoss(); best_f1=-np.inf; best=None
    for e in range(epochs):
        model.train(); losses=[]
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
            loss=loss_fn(model(xb),yb); opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        m=evaluate_probs(y_val,predict_probs(model,X_val,device,batch_size))
        print(f"FT epoch {e+1}/{epochs}: loss={np.mean(losses):.4f}, val_f1={m['f1']:.4f}")
        if m["f1"]>best_f1:
            best_f1=m["f1"]; best=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
    if best is None: raise RuntimeError("No valid checkpoint was produced.")
    model.load_state_dict(best); return model
