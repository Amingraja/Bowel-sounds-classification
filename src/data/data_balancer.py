from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config
from src.data.data_preprocessor import DataPreprocessor


class DataBalancer:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.config = load_config()

    def balance_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = self.data_preprocessor.process_folder(
            input_folder=self.config["train_folder"], 
            output_folder=self.config["output_train_folder"], 
            min_duration=self.config.get("min_duration", 0.1)
        )

        target_count = 300
        balanced_rows = []
        for label, group in df_train.groupby('label'):
            if label == 'h':
                balanced_rows.append(group.sample(n=target_count, replace=True, random_state=42))
            else:
                balanced_rows.append(group.sample(n=target_count, random_state=42))
        df_train = pd.concat(balanced_rows).reset_index(drop=True)

        val_ratio = self.config["val_ratio"]
        train_df, val_df = train_test_split(df_train, test_size=val_ratio, stratify=df_train["label"], random_state=42)

        return train_df, val_df
    
    def balance_test(self) -> pd.DataFrame:
        df_test = self.data_preprocessor.process_folder(
            input_folder=self.config["test_folder"], 
            output_folder=self.config["output_test_folder"],
            min_duration=self.config.get("min_duration", 0.1)
        )

        return df_test
    
