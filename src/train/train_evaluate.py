from datasets import Dataset, DatasetDict

from src.utils import csv_to_df, load_config
from src.data.data_preprocessor import DataPreprocessor
from src.train.model_trainer import ModelTrainer


if __name__ == "__main__":
    config = load_config()
    model_trainer = ModelTrainer()

    train_df = csv_to_df(config["train"])
    val_df = csv_to_df(config["val"])
    test_df = csv_to_df(config["test"])

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })
    dataset = dataset.map(lambda x: DataPreprocessor.preprocess_audio(x, min_duration=0.2))
    # def preprocess_labels(batch):
    #     batch["labels"] = int(batch["label"])  # ensure Python int
    #     return batch

    # dataset = dataset.map(preprocess_labels)


    model_trainer.train_evaluate(dataset)

    

