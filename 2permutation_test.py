import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import copy
import random
from joblib import Parallel, delayed
from tqdm import tqdm

random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def ld_pruning(X, threshold=0.8):
    if X.shape[1] == 0:
        return X, np.array([])
    
    stds = X.std(axis=0)
    non_zero_var_idx = np.where(stds > 0)[0]
    if len(non_zero_var_idx) == 0:
        return X[:, []], np.array([])
    
    X_clean = X[:, non_zero_var_idx]
    if X_clean.shape[1] == 1:
        return X_clean, non_zero_var_idx
    
    corr_matrix = np.corrcoef(X_clean, rowvar=False) ** 2
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] < 2:
        return X_clean, non_zero_var_idx
    
    mask = np.triu(corr_matrix, k=1) > threshold
    to_remove = set()
    for i in range(mask.shape[0]):
        for j in range(i + 1, mask.shape[1]):
            if mask[i, j]:
                to_remove.add(j)
    
    keep_idx = [i for i in range(X_clean.shape[1]) if i not in to_remove]
    X_pruned = X_clean[:, keep_idx]
    keep_idx_original = non_zero_var_idx[keep_idx]
    return X_pruned, keep_idx_original


class MLPFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class GenomicMLP(nn.Module):
    def __init__(self, num_snps):
        super().__init__()
        self.snp_weights = nn.Parameter(torch.ones(num_snps) / num_snps)
        self.mlp = MLPFunction(1)
    
    def forward(self, x):
        weighted_sum = torch.matmul(x, self.snp_weights).unsqueeze(-1)
        return self.mlp(weighted_sum)


class MultiOmicsModel(nn.Module):
    def __init__(self, num_snps, omics_dims):
        super().__init__()
        self.genomic = GenomicMLP(num_snps) if num_snps > 0 else None
        self.omics_mlps = nn.ModuleList([MLPFunction(1) for _ in range(len(omics_dims))])
        self.omics_dims = omics_dims
        self.num_snps = num_snps
    
    def forward(self, snp, omics, m_params):
        y = torch.zeros(snp.shape[0], device=snp.device, dtype=snp.dtype)
        
        if self.num_snps > 0 and m_params[0] > 0:
            y = y + self.genomic(snp) * m_params[0]
        
        omics_start = 0
        for k, dim in enumerate(self.omics_dims):
            if dim > 0:
                omics_k = omics[:, omics_start:omics_start + dim]
                y = y + self.omics_mlps[k](omics_k) * m_params[k + 1]
                omics_start += dim
        return y.squeeze(-1)


def fit(x_train, y_train, x_val, y_val, x_test, y_test, omics_dims, num_snps):
    if num_snps > 0:
        snp_train = torch.tensor(x_train[:, :num_snps], dtype=torch.float32)
        snp_val = torch.tensor(x_val[:, :num_snps], dtype=torch.float32)
        snp_test = torch.tensor(x_test[:, :num_snps], dtype=torch.float32)
        omics_train = torch.tensor(x_train[:, num_snps:], dtype=torch.float32)
        omics_val = torch.tensor(x_val[:, num_snps:], dtype=torch.float32)
        omics_test = torch.tensor(x_test[:, num_snps:], dtype=torch.float32)
    else:
        snp_train = torch.zeros((x_train.shape[0], 0), dtype=torch.float32)
        snp_val = torch.zeros((x_val.shape[0], 0), dtype=torch.float32)
        snp_test = torch.zeros((x_test.shape[0], 0), dtype=torch.float32)
        omics_train = torch.tensor(x_train, dtype=torch.float32)
        omics_val = torch.tensor(x_val, dtype=torch.float32)
        omics_test = torch.tensor(x_test, dtype=torch.float32)
    
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    
    outer_model = MultiOmicsModel(num_snps, omics_dims)
    
    best_val_r2 = -float('inf')
    best_m = None
    best_model_state = None
    
    valid_combinations = []
    for i in range(16):
        bits = [(i >> k) & 1 for k in range(4)]
        valid = True
        
        if bits[0] == 1 and num_snps == 0:
            valid = False
        
        for k in range(3):
            if bits[k + 1] == 1 and omics_dims[k] == 0:
                valid = False
                break
        
        if sum(bits) == 0:
            valid = False
        
        if valid:
            valid_combinations.append(bits)
    
    for bits in valid_combinations:
        m_params = torch.tensor(bits, dtype=torch.float32)
        
        inner_model = MultiOmicsModel(num_snps, omics_dims)
        inner_model.load_state_dict(outer_model.state_dict())
        
        optimizer = optim.Adam(inner_model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_fold_val_r2 = -float('inf')
        best_fold_state = None
        patience = 15
        patience_counter = 0
        
        for epoch in range(300):
            inner_model.train()
            output = inner_model(snp_train, omics_train, m_params).squeeze()
            loss = nn.MSELoss()(output, y_train_torch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                inner_model.eval()
                val_output = inner_model(snp_val, omics_val, m_params).squeeze()
                val_r2 = r2_score(y_val, val_output.numpy())
                
                if val_r2 > best_fold_val_r2:
                    best_fold_val_r2 = val_r2
                    best_fold_state = copy.deepcopy(inner_model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        outer_model.load_state_dict(best_fold_state)
        
        if best_fold_val_r2 > best_val_r2:
            best_val_r2 = best_fold_val_r2
            best_m = m_params.clone()
            best_model_state = best_fold_state
    
    final_model = MultiOmicsModel(num_snps, omics_dims)
    final_model.load_state_dict(best_model_state)
    
    with torch.no_grad():
        y_pred = final_model(snp_test, omics_test, best_m)
        r2 = r2_score(y_test, y_pred.numpy())
    
    output_m = best_m.numpy().copy()
    if num_snps == 0:
        output_m[0] = np.nan
    for k in range(3):
        if omics_dims[k] == 0:
            output_m[k + 1] = np.nan
    
    return r2, output_m


pheno = pd.read_csv(r"./sample_174.csv").iloc[:, 1]
pheno = np.array(pheno)
snp_data = pd.read_csv(r"./snp_2247428_174_012.csv")
rna_data = pd.read_csv(r"./rna_174_28279_norm.csv")
ribo_data = pd.read_csv(r"./ribo_174_18426_norm.csv")
pro_data = pd.read_csv(r"./pro_174_6210_norm.csv")
id_all_gene = pd.read_csv(r"gene_test.csv").iloc[:, 0].unique().tolist()
N_shuffles = 20
kf = KFold(n_splits=10, shuffle=True, random_state=2025)
kf_splits = list(kf.split(range(len(pheno))))


def run_gene_shuffle_fold(id_genei, shuffle_id, fold_idx):
    gene_genei = None
    snp_subset = snp_data[snp_data.iloc[:, 2] == id_genei]
    if len(snp_subset) > 0:
        gene_genei = np.array(snp_subset)[:, 9:].T.astype(np.int8)
        gene_genei, _ = ld_pruning(gene_genei, threshold=0.8)
    
    rna_genei = None
    ribo_genei = None
    pro_genei = None
    
    if id_genei in rna_data.columns:
        rna_genei = np.array(rna_data[id_genei].values.reshape(-1, 1))
        rna_genei = StandardScaler().fit_transform(rna_genei)
    
    if id_genei in ribo_data.columns:
        ribo_genei = np.array(ribo_data[id_genei].values.reshape(-1, 1))
        ribo_genei = StandardScaler().fit_transform(ribo_genei)
    
    if id_genei in pro_data.columns:
        pro_genei = np.array(pro_data[id_genei].values.reshape(-1, 1))
        pro_genei = StandardScaler().fit_transform(pro_genei)
    
    if gene_genei is None and rna_genei is None and ribo_genei is None and pro_genei is None:
        print(f"Warning: No data found for gene {id_genei}")
        return None
    
    feat_list = []
    omics_dims = []
    num_snps = 0
    
    if gene_genei is not None:
        feat_list.append(gene_genei)
        num_snps = gene_genei.shape[1]
    
    if rna_genei is not None:
        feat_list.append(rna_genei)
        omics_dims.append(1)
    else:
        omics_dims.append(0)
    
    if ribo_genei is not None:
        feat_list.append(ribo_genei)
        omics_dims.append(1)
    else:
        omics_dims.append(0)
    
    if pro_genei is not None:
        feat_list.append(pro_genei)
        omics_dims.append(1)
    else:
        omics_dims.append(0)
    
    feat_all = np.concatenate(feat_list, axis=1)
    y_scaled_original = StandardScaler().fit_transform(pheno.reshape(-1, 1)).ravel()
    
    RNG = np.random.default_rng(shuffle_id)
    y_scaled = RNG.permutation(y_scaled_original)
    
    train_val_idx, test_idx = kf_splits[fold_idx]
    X_train_val, X_test = feat_all[train_val_idx], feat_all[test_idx]
    y_train_val, y_test = y_scaled[train_val_idx], y_scaled[test_idx]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=2025, shuffle=True
    )
    
    r2, best_subnet = fit(X_train, y_train, X_val, y_val, X_test, y_test, omics_dims, num_snps)
    
    return {
        "gene_id": id_genei,
        "shuffle_id": shuffle_id,
        "fold": fold_idx + 1,
        "r2": r2,
        "subnet": best_subnet.tolist(),
        "omics_dims": omics_dims
    }


all_tasks = [
    (id_genei, shuffle_id, fold_idx)
    for id_genei in id_all_gene
    for shuffle_id in range(1, N_shuffles + 1)
    for fold_idx in range(10)
]

all_results = Parallel(n_jobs=-1, verbose=1)(
    delayed(run_gene_shuffle_fold)(id_genei, shuffle_id, fold_idx)
    for (id_genei, shuffle_id, fold_idx) in tqdm(all_tasks, desc="Gene-Shuffle-Fold")
)

all_results = [r for r in all_results if r is not None]

df_results = pd.DataFrame(all_results)
df_results.to_csv(".permuted_results.csv", index=False)

print("Done! Saved to .permuted_results.csv")
