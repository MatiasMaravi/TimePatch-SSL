"""Experiment 1: within-dataset supervised subject-independent evaluation.

Runs leave-one-subject-out evaluation for 30-s and 60-s windows, using a
participant-disjoint validation subset from the development participants.
"""
from __future__ import annotations
import argparse, numpy as np, torch
from src.preprocessing import TrainingOnlyRobustScaler
from src.models import TimePatchTransformer
from src.train import finetune, predict_probs
from src.evaluate import evaluate_probs
from experiments._common import set_seed, participant_train_val, save_json

WINDOWS=(30,60)

def run_loso(X,y,groups,hierarchical,device,seed=42,model_kwargs=None,train_kwargs=None):
    model_kwargs=model_kwargs or {}; train_kwargs=train_kwargs or {}
    rows=[]
    for heldout in np.unique(groups):
        test_idx=np.flatnonzero(groups==heldout)
        dev_ids=np.unique(groups[groups!=heldout])
        fit_idx,val_idx,fit_ids,val_ids=participant_train_val(groups,dev_ids,.15,seed)

        scaler=TrainingOnlyRobustScaler().fit(X[fit_idx])
        Xfit=scaler.transform(X[fit_idx]); Xval=scaler.transform(X[val_idx]); Xtest=scaler.transform(X[test_idx])

        set_seed(seed)
        model=TimePatchTransformer(hierarchical=hierarchical,**model_kwargs)
        model=finetune(model,Xfit,y[fit_idx],Xval,y[val_idx],device,**train_kwargs)
        probs=predict_probs(model,Xtest,device)
        m=evaluate_probs(y[test_idx],probs,.5)
        m.update(participant=str(heldout),n_windows=int(len(test_idx)),
                 fit_participants=[str(x) for x in fit_ids],
                 validation_participants=[str(x) for x in val_ids])
        rows.append(m)
    return rows

def summarize(rows):
    keys=("accuracy","f1","balanced_accuracy","auroc")
    return {k:float(np.nanmean([r[k] for r in rows])) for k in keys}

def run_configuration(X,y,groups,device,seed=42):
    result={}
    for hierarchical,name in ((False,"standard"),(True,"hierarchical")):
        rows=run_loso(X,y,groups,hierarchical,device,seed)
        result[name]={"summary":summarize(rows),"participants":rows}
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--note",action="store_true")
    args=parser.parse_args()
    print("Import run_configuration() and pass the 30-s or 60-s dataset arrays.")
    print("Run both WINDOW configurations:", WINDOWS)
