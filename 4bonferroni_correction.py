import pandas as pd
import math

INPUT_FILE = ".uncorrected_pvalues.csv"
OUTPUT_FILE = ".corrected_pvalues.csv"
GOOD_FILE = "significant_genes.csv"

df = pd.read_csv(INPUT_FILE)

if "p_value" not in df.columns:
    raise ValueError("p_value column not found in input file")

m = len(df)
print(f"Total tests: {m}")

df["p_value"] = df["p_value"] * m
df["p_value"] = df["p_value"].clip(upper=1.0)


def safe_log10(p):
    if p <= 0:
        return 300
    return -math.log10(p)


df["-log(p_value)"] = df["p_value"].apply(safe_log10)

cols_to_drop = ["pos_mean", "pos_no1"]
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved Bonferroni corrected file to {OUTPUT_FILE}")

df_good = df[df["p_value"] < 0.001]
df_good.to_csv(GOOD_FILE, index=False)
print(f"Saved significant genes (p<0.001) to {GOOD_FILE}")

print("Done.")
