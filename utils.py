import os, yaml

# load e.g. normal.yaml
def load_config(expr_name, config_dir="configs"):
  path = os.path.join(base_dir, f"{config_name}.yaml")
  with open(path, "r") as f:
    cfg = yaml.safe_load(f)
  cfg["data_folder"] = os.path.join(cfg["csv_folder"], cfg["data_csv"])
  return cfg
  