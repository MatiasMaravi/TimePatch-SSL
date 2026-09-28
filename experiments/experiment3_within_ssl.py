"""Experiment 3: within-dataset SSL + supervised fine-tuning.

Uses LOSO subject-independent evaluation. SSL pretraining is restricted to
fitting participants; held-out and validation participants are excluded.
"""
from __future__ import annotations
import numpy as np
from src.preprocessing import TrainingOnlyRobustScaler
from src.models import TimePatchTransformer
from src.ssl import pretrain_ssl
from src.train import finetune, predict_probs
from src.evaluate import evaluate_probs
from experiments._common import set_seed, participant_train_val

def run_loso_ssl(X,y,groups,hierarchical,device,seed=42,
                 model_kwargs=None,ssl_kwargs=None,train_kwargs=None):
    model_kwargs=model_kwargs or {}; ssl_kwargs=ssl_kwargs or {}; train_kwargs=train_kwargs or {}
    rows=[]
    for heldout in np.unique(groups):
        test_idx=np.flatnonzero(groups==heldout)
        dev_ids=np.unique(groups[groups!=heldout])
        fit_idx,val_idx,fit_ids,val_ids=participant_train_val(groups,dev_ids,.15,seed)

        scaler=TrainingOnlyRobustScaler().fit(X[fit_idx])
        Xfit=scaler.transform(X[fit_idx]); Xval=scaler.transform(X[val_idx]); Xtest=scaler.transform(X[test_idx])

        set_seed(seed)
        model=TimePatchTransformer(hierarchical=hierarchical,**model_kwargs)
        model=pretrain_ssl(model,Xfit,device,**ssl_kwargs)
        model=finetune(model,Xfit,y[fit_idx],Xval,y[val_idx],device,**train_kwargs)
        p=predict_probs(model,Xtest,device)
        m=evaluate_probs(y[test_idx],p,.5)
        m.update(participant=str(heldout),n_windows=int(len(test_idx)),
                 ssl_participants=[str(x) for x in fit_ids],
                 validation_participants=[str(x) for x in val_ids])
        rows.append(m)
    return rows

def summarize(rows):
    return {k:float(np.nanmean([r[k] for r in rows]))
            for k in ("accuracy","f1","balanced_accuracy","auroc")}

if __name__=="__main__":
    print("Run LOSO SSL for standard and hierarchical encoders at 30 s and 60 s.")
