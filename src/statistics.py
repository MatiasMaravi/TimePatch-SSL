"""Subject-level paired statistical analysis used for Table 4."""
import numpy as np
from scipy.stats import wilcoxon, rankdata

BONFERRONI_ALPHA = .05/6

def rank_biserial(a,b):
    d=np.asarray(a,float)-np.asarray(b,float); d=d[d!=0]
    if len(d)==0: return 0.0
    r=rankdata(np.abs(d))
    rp=r[d>0].sum(); rm=r[d<0].sum()
    return float((rp-rm)/(rp+rm))

def wilcoxon_compare(a,b):
    """Compare paired participant values after they have been aligned by participant ID."""
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape: raise ValueError("Paired arrays must have identical shape.")
    stat,p=wilcoxon(a,b,alternative="two-sided")
    return {"W":float(stat),"p_value":float(p),"rank_biserial":rank_biserial(a,b),
            "mean_a":float(a.mean()),"mean_b":float(b.mean()),
            "mean_difference":float(np.mean(a-b)),
            "significant_bonferroni":bool(p < BONFERRONI_ALPHA)}

def paired_metric_from_records(records_a,records_b,metric):
    """Align result records by participant before the paired test."""
    A={str(r["participant"]):float(r[metric]) for r in records_a}
    B={str(r["participant"]):float(r[metric]) for r in records_b}
    ids=sorted(set(A)&set(B))
    if not ids: raise ValueError("No shared participant IDs.")
    result=wilcoxon_compare([A[i] for i in ids],[B[i] for i in ids])
    result["participants"]=ids
    return result
