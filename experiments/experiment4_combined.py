"""Experiment 4: combined-dataset subject-independent evaluation.

The 80/20 split is performed on participants, never on windows. WESAD uses
12 development / 3 test participants; Stress-Predict uses 28 / 7. The two
development partitions are pooled for model development and the held-out
partitions are pooled for final evaluation.
"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from src.preprocessing import TrainingOnlyRobustScaler
from src.models import TimePatchTransformer
from src.train import finetune, predict_probs
from src.evaluate import evaluate_probs, per_participant_metrics
from experiments._common import set_seed, participant_train_val

def indices_from_ids(groups,ids):
    return np.flatnonzero(np.isin(np.asarray(groups),np.asarray(ids)))

def load_split(path):
    with open(path,encoding="utf-8") as f: return json.load(f)

def run_combined(Xw,yw,gw,Xs,ys,gs,split_json,device,seed=42,
                 hierarchical=True,use_ssl=True,model_kwargs=None,
                 ssl_kwargs=None,train_kwargs=None):
    model_kwargs=model_kwargs or {}; ssl_kwargs=ssl_kwargs or {}; train_kwargs=train_kwargs or {}
    spec=load_split(split_json)["datasets"]
    wdev=spec["WESAD"]["development_participant_ids"]; wtest=spec["WESAD"]["test_participant_ids"]
    sdev=spec["Stress-Predict"]["development_participant_ids"]; stest=spec["Stress-Predict"]["test_participant_ids"]

    wi_dev,wi_test=indices_from_ids(gw,wdev),indices_from_ids(gw,wtest)
    si_dev,si_test=indices_from_ids(gs,sdev),indices_from_ids(gs,stest)

    assert set(np.unique(gw[wi_dev])).isdisjoint(set(np.unique(gw[wi_test])))
    assert set(np.unique(gs[si_dev])).isdisjoint(set(np.unique(gs[si_test])))

    # Keep dataset identity in participant keys so equal labels (e.g. S14) cannot collide.
    gdev=np.concatenate([np.array(["WESAD:"+str(x) for x in gw[wi_dev]]),
                         np.array(["STRESS:"+str(x) for x in gs[si_dev]])])
    Xdev=np.concatenate([Xw[wi_dev],Xs[si_dev]]); ydev=np.concatenate([yw[wi_dev],ys[si_dev]])
    dev_ids=np.unique(gdev)
    fit_idx,val_idx,fit_ids,val_ids=participant_train_val(gdev,dev_ids,.15,seed)

    Xtest=np.concatenate([Xw[wi_test],Xs[si_test]])
    ytest=np.concatenate([yw[wi_test],ys[si_test]])
    gtest=np.concatenate([np.array(["WESAD:"+str(x) for x in gw[wi_test]]),
                          np.array(["STRESS:"+str(x) for x in gs[si_test]])])

    scaler=TrainingOnlyRobustScaler().fit(Xdev[fit_idx])
    Xfit=scaler.transform(Xdev[fit_idx]); Xval=scaler.transform(Xdev[val_idx]); Xtest=scaler.transform(Xtest)

    set_seed(seed)
    model=TimePatchTransformer(hierarchical=hierarchical,**model_kwargs)
    if use_ssl:
        from src.ssl import pretrain_ssl
        model=pretrain_ssl(model,Xfit,device,**ssl_kwargs)
    model=finetune(model,Xfit,ydev[fit_idx],Xval,ydev[val_idx],device,**train_kwargs)
    p=predict_probs(model,Xtest,device)
    return {"metrics":evaluate_probs(ytest,p,.5),
            "per_participant":per_participant_metrics(ytest,p,gtest,.5),
            "development_participants":[str(x) for x in dev_ids],
            "fit_participants":[str(x) for x in fit_ids],
            "validation_participants":[str(x) for x in val_ids],
            "test_participants":[str(x) for x in np.unique(gtest)]}

if __name__=="__main__":
    print("Experiment 4 uses participant-level 80/20 splits for both datasets.")
    print("Execute separately for 30-s and 60-s window arrays.")
