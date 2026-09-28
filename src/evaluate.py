"""Binary and participant-level evaluation, including Table 2 audit support."""
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,roc_auc_score,confusion_matrix,precision_score,recall_score

def evaluate_probs(y_true,p_stress,threshold=.5):
    y=np.asarray(y_true); p=np.asarray(p_stress); pred=(p>=threshold).astype(int)
    r={"accuracy":accuracy_score(y,pred),"f1":f1_score(y,pred,zero_division=0),
       "balanced_accuracy":balanced_accuracy_score(y,pred),
       "precision":precision_score(y,pred,zero_division=0),
       "recall":recall_score(y,pred,zero_division=0),
       "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    try: r["auroc"]=roc_auc_score(y,p)
    except ValueError: r["auroc"]=float("nan")
    return r

def audit_binary_scores(y_true,p_stress,threshold=.5):
    r=evaluate_probs(y_true,p_stress,threshold)
    try: r["auroc_inverted"]=roc_auc_score(y_true,1-np.asarray(p_stress))
    except ValueError: r["auroc_inverted"]=float("nan")
    return r

def per_participant_metrics(y_true,p_stress,groups,threshold=.5):
    rows=[]; y=np.asarray(y_true); p=np.asarray(p_stress); g=np.asarray(groups)
    for sid in np.unique(g):
        idx=np.flatnonzero(g==sid)
        row={"participant":str(sid),"n_windows":len(idx)}
        row.update(evaluate_probs(y[idx],p[idx],threshold)); rows.append(row)
    return rows
