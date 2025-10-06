
from sklearn.model_selection import KFold
import pandas as pd 
import numpy as np
import os 

DATA_ROOT = "data"
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

def batch_data(data, val_columns):
  data_map = {}
  for _, sliced in data.groupby("id"):
    col_mappings = {"outputs":["hr"], "inputs":val_columns}
    for k in col_mappings:
      cols = col_mappings[k] # ["hr"]/ ["hr","xt1","xt2",...]
      arr = _batch_single_entity(sliced[cols].copy())


# # For debug
# raw_data = pd.read_csv(os.path.join(DATA_ROOT, )"data/result/uwb_hr_normal.csv")
# folds = make_kfolds(raw_data, n_splits=5)
# for i, (train_df, test_df) in enumerate(folds, start=1):
#   print(f"\n=== Fold {i} ===")
#   print("Train IDs:", train_df["id"].unique())
#   print("Test IDs:", test_df["id"].unique())

all_experiments_losses = {}
for experiment_name in EXPERIMENTS:
  print(f"=== Start experiment: {experiment_name} ===")
  file_name = f"uwb_hr_{experiment_name}.csv"
  data_path = os.path.join(DATA_ROOT, "result", file_name)
  raw_data = pd.read_csv(data_path, dtype={"hr": float})
  all_losses = {f"p{int(q*100)}": [] for q in QUANTILES}
  # Cross validation
  folds = make_kfolds(raw_data, n_splits=N_SPLITS)
  for fold_idx, (train_df, test_df) in enumerate(folds, start=1):
    print(f"\n--- Fold {fold_idx}/{N_SPLITS} ---")
    for q in QUANTILES:
      fold_losses = []
      for col in range(0, 300, 10):
        chosen_columns = [f"xt{i}" for i in range(col, col+10)] # first 10 columns, next 10 columns...
        target_col = ['hr']
        val_columns = ['hr'] + chosen_columns # hr + Xethru columns
        # Scale data 
        input_scaler = StandardScaler().fit(train_df[chosen_columns]) # for Xethru only
        target_scaler = StandardScaler().fit(train_df[['hr']]) # for heart-rate only
        
        scaled_train = train_df.copy()
        scaled_test= test_df.copy()
        scaled_train[chosen_columns] = input_scaler.transform(train_df[chosen_columns])
        scaled_train[target_col] = target_scaler.transform(train_df[target_col])
        scaled_test[chosen_columns] = input_scaler.transform(test_df[chosen_columns])
        scaled_test[target_col] = target_scaler.transform(test_df[target_col])
        # Batchify
        train_data = batch_data(scaled_train, val_columns)