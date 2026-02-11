import pandas as pd
import numpy as np
import math
from scipy.stats import ttest_1samp

POSITIVE_FILE = ".observed_results.csv"
SHUFFLE_FILE = ".permuted_results.csv"
OUTPUT_FILE = ".uncorrected_pvalues.csv"

df_pos = pd.read_csv(POSITIVE_FILE)
df_shuf = pd.read_csv(SHUFFLE_FILE)


def safe_eval_list(s):
    return eval(s, {"nan": np.nan})


df_pos['subnet'] = df_pos['subnet'].apply(safe_eval_list)
df_shuf['subnet'] = df_shuf['subnet'].apply(safe_eval_list)


def replace_nan(lst):
    return [-1 if pd.isna(x) else x for x in lst]


df_pos['subnet'] = df_pos['subnet'].apply(replace_nan)
df_shuf['subnet'] = df_shuf['subnet'].apply(replace_nan)

results = []

for gene_id in df_pos["gene_id"].unique():
    pos_r2 = df_pos[df_pos["gene_id"] == gene_id].sort_values("r2")["r2"].values
    if len(pos_r2) < 6:
        raise ValueError(f"Gene {gene_id} less than 6 folds, cannot remove 2 head/tail.")
    pos_r2_iqr = np.sort(pos_r2)[4:-2]
    pos_mean = np.mean(pos_r2_iqr)
    pos_no1 = pos_r2_iqr[-1]
    
    gene_shuf = df_shuf[df_shuf["gene_id"] == gene_id]
    shuf_means = []
    for shuffle_id in gene_shuf["shuffle_id"].unique():
        shuf_r2 = gene_shuf[gene_shuf["shuffle_id"] == shuffle_id].sort_values("r2")["r2"].values
        if len(shuf_r2) < 6:
            raise ValueError(f"Gene {gene_id} shuffle {shuffle_id} less than 6 folds.")
        shuf_iqr = np.sort(shuf_r2)[4:-2]
        shuf_means.append(np.mean(shuf_iqr))
    
    stat, pval = ttest_1samp(shuf_means, popmean=pos_mean, alternative="less")
    
    subnet_rows = df_pos[df_pos["gene_id"] == gene_id].sort_values("r2").iloc[4:-2]
    
    subnets = np.array(subnet_rows["subnet"].tolist())
    vote = []
    for col in subnets.T:
        ones = np.sum(col)
        zeros = len(col) - ones
        if ones >= zeros:
            vote.append(1)
        else:
            vote.append(0)
    
    if pos_no1 < 0.01:
        pval = 0.999
    
    results.append({
        "gene_id": gene_id,
        "pos_mean": pos_mean,
        "pos_no1": pos_no1,
        "p_value": pval,
        "-log(p_value)": -math.log10(pval),
        "voted_subnet": vote
    })

df_final = pd.DataFrame(results)
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"Done! Saved to {OUTPUT_FILE}")
