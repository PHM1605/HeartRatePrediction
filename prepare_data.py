import argparse, glob, os, datetime
import pandas as pd 
from utils import load_config

def merge_hr_xethru(data_folder, output_csv, chosen_columns):
  hr_paths = sorted(glob.glob(os.path.join(data_folder, "*_hr.csv")))
  xt_paths = sorted(glob.glob(os.path.join(data_folder, "*_xethru.csv")))
  
  df_list = []
  for hr_path, xt_path in zip(hr_paths, xt_paths):
    df_hr = pd.read_csv(hr_path, header=None)
    df_xt = pd.read_csv(xt_path, header=None)
    # parsing "18-06-2019" and "11:52:44 AM"
    start_time = datetime.datetime.strptime(
      f"{df_hr.iloc[1,2]} {df_hr.iloc[1,3]}",
      "%d-%m-%Y %I:%M:%S %p"
    )

    df_hr = df_hr.iloc[3:, [1,2]].reset_index(drop=True)
    df_xt = df_xt.iloc[::10, 1:].reset_index(drop=True) # downsample every 10 rows
    # Trim longer signal
    n = min(len(df_hr), len(df_xt))
    df_hr, df_xt = df_hr.iloc[:n], df_xt.iloc[:n]

    # Build datetime from start_time + hh:mm:ss
    
    times = []
    # df_hr column starts from 1
    for t in df_hr[1]:
      tmp = datetime.datetime.strptime(t, "%H:%M:%S")
      times.append(start_time + datetime.timedelta(seconds=tmp.hour*3600+tmp.minute*60+tmp.second))

    # Merge
    df_xt.columns = [f"xt{i}" for i in range(df_xt.shape[1])]
    df = pd.DataFrame({
      "id": os.path.basename(hr_path).split("_")[0], # user1; same user_id for all rows
      "time": times,
      "hr": df_hr[2].astype(float)
    })
    df = pd.concat([df, df_xt.iloc[:, :len(chosen_columns)]], axis=1) # 1st column is <hr>, next 300 columns are <xt1>,<xt2>,...
    df_list.append(df)
  
  merged = pd.concat(df_list).reset_index(drop=True) # 1 file = all users; first N rows of user1; next M rows of user2...
  merged.to_csv(output_csv, index=False)
  print(f"Saved merged csv => {output_csv}")

def main():
  parser = argparse.ArgumentParser(
    description = "Merge <user>_<state>_hr.csv and <user>_<state>_xethru.csv into one dataset csv" 
  )
  parser.add_argument("--experiment", required=True, choices=["normal", "rest", "abnormal"],
    help="Experiment folder name under data/mecs")
  parser.add_argument("--columns", type=int, default=300,
    help="Number of XeThru channels to include (default: 300)")

  args = parser.parse_args() 
  base = "data/mecs"
  data_folder = os.path.join(base, args.experiment)
  output_csv = os.path.join("data", "result", f"uwb_hr_{args.experiment}.csv")
  print(f"### Preparing dataset for experiment '{args.experiment}' ###")
  merge_hr_xethru(data_folder, output_csv, chosen_columns=range(args.columns))
  print("Done.")

if __name__ == "__main__":
  main()
