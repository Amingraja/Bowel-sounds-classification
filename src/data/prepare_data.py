import pandas as pd

from src.data.data_balancer import DataBalancer
from src.utils import load_config


if __name__ == "__main__":
    data_balancer = DataBalancer()
    config = load_config()

    train_df, val_df = data_balancer.balance_train_val()
    test_df = data_balancer.balance_test()
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    print("Class distribution (train):\n", train_df["label"].value_counts())
    train_df.to_csv(config["train"], index=False)
    test_df.to_csv(config["test"], index=False)
    val_df.to_csv(config["val"], index=False)

