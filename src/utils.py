import yaml

import pandas as pd


def load_config(config_path: str = "src/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def csv_to_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)