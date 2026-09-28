"""Experiment 2: cross-dataset transfer.

Development is restricted to the source dataset. The source is partitioned
into participant-disjoint fitting/validation groups; the target dataset is
evaluation-only. The classification threshold is fixed at 0.5.
"""
from __future__ import annotations
import numpy as np
from src.preprocessing import TrainingOnlyRobustScaler
from src.models import TimePatchTransformer
from src.train import finetune, predict_probs
from src.evaluate import evaluate_probs, audit_binary_scores, per_participant_metrics
from experiments._common import set_seed, participant_train_val

def run_cross_dataset(X_source,y_source,g_source,X_target,y_target,g_target,
                      hierarchical,device,seed=42,use_ssl=False,
                      ssl_kwargs=None,model_kwargs=None,train_kwargs=None):
    model_kwargs=model_kwargs or {}; train_kwargs=train_kwargs or {}; ssl_kwargs=ssl_kwargs or {}
    source_ids=np.unique(g_source)
    fit_idx,val_idx,fit_ids,val_ids=participant_train_val(g_source,source_ids,.15,seed)

    scaler=TrainingOnlyRobustScaler().fit(X_source[fit_idx])
    Xfit=scaler.transform(X_source[fit_idx])
    Xval=scaler.transform(X_source[val_idx])
    Xtarget=scaler.transform(X_target)

    set_seed(seed)
    model=TimePatchTransformer(hierarchical=hierarchical,**model_kwargs)
    if use_ssl:
        from src.ssl import pretrain_ssl
        # SSL sees source fitting participants only; target participants are untouched.
        model=pretrain_ssl(model,Xfit,device,**ssl_kwargs)
    model=finetune(model,Xfit,y_source[fit_idx],Xval,y_source[val_idx],
                   device,**train_kwargs)
    p=predict_probs(model,Xtarget,device)
    return {
        "metrics":evaluate_probs(y_target,p,.5),
        "audit":audit_binary_scores(y_target,p,.5),
        "per_participant":per_participant_metrics(y_target,p,g_target,.5),
        "source_fit_participants":[str(x) for x in fit_ids],
        "source_validation_participants":[str(x) for x in val_ids],
        "target_participants":[str(x) for x in np.unique(g_target)],
        "predictions":{"y_true":np.asarray(y_target).astype(int).tolist(),
                       "p_stress":np.asarray(p).astype(float).tolist(),
                       "y_pred":(np.asarray(p)>=.5).astype(int).tolist(),
                       "participant":[str(x) for x in g_target]},
    }

if __name__=="__main__":
    print("Experiment 2 supports WESAD -> Stress-Predict and Stress-Predict -> WESAD")
    print("for both 30-s and 60-s windows. Target labels are evaluation-only.")
