
from sklearn.model_selection import KFold
import pandas as pd 
import numpy as np

EXPERIMENTS = ["rest", "normal", "abnormal"]
MODEL_NAME = "seq2seq"
N_SPLITS = 5
PIECE_LEN = 30
ENCODER_STEPS = 20
QUANTILES = [0.1, 0.5, 0.9]

def make_kfolds(df, n_splits=5, random_state=42):
  user_ids = df['id'].unique()
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
  folds = []
  # kf.split(): [(train_indices1,test_indices1), (train_indices2,test_indices2),...]
  for fold_idx, (train_indices, test_indices) in enumerate(kf.split(user_ids), start=1):
    train_ids = user_ids[train_indices] # [user1,user4,user12,...]
    test_ids = user_ids[test_indices] # [user2,user3]
    train_df = df[df['id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['id'].isin(test_ids)].reset_index(drop=True)
    folds.append((train_df, test_df))
    print(f"Fold {fold_idx}: {len(train_ids)} train users, {len(test_ids)} test users")
  return folds 

raw_data = pd.read_csv("data/result/uwb_hr_normal.csv")
folds = make_kfolds(raw_data, n_splits=5)

# # For debug
# for i, (train_df, test_df) in enumerate(folds, start=1):
#   print(f"\n=== Fold {i} ===")
#   print("Train IDs:", train_df["id"].unique())
#   print("Test IDs:", test_df["id"].unique())

all_experiments_losses = {}
