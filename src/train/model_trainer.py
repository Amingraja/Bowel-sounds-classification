from datetime import datetime
from functools import partial
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import ClassLabel
from sklearn.metrics import f1_score, precision_score, confusion_matrix
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Trainer, TrainingArguments

from src.utils import load_config
from src.data.data_preprocessor import DataPreprocessor


class ModelTrainer:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.config = load_config()
        self.feature_extractor = None
        self.model_name = None
        self.class_labels = None
        self.num_classes = None
    
    def train_evaluate(self, dataset):
        ensemble_models = self.config.get("ensemble_models", [
            
            "microsoft/wavlm-base"
        ])
        unique_labels = sorted(set(dataset['train']['label']))
        self.class_labels = ClassLabel(names=unique_labels)
        self.num_classes = self.get_class_num(dataset, self.class_labels)

        for model_name in ensemble_models:
            self.model_name = model_name
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.feature_extractor = feature_extractor
            ds_features = dataset.map(self.extract_features)

            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name, num_labels=self.num_classes, mask_time_length=1
            )
            training_args = TrainingArguments(**self.config["training_args"])
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=ds_features['train'],
                eval_dataset=ds_features['validation'],
                tokenizer=feature_extractor,
                compute_metrics=self.compute_metrics
            )

            print("Starting training...")
            trainer.train()
            print("Training finished.")

            evaluation_results = trainer.evaluate(eval_dataset=ds_features['test'])
            model.save_pretrained(f"./fine_tuned_model_{model_name.replace('/', '_')}")
            print(f"Model saved to ./fine_tuned_model_{model_name.replace('/', '_')}")
            print("\nFinal evaluation results on test set:")
            for k, v in evaluation_results.items():
                if isinstance(v, (int, float)):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

    @staticmethod
    def get_class_num(dataset, class_labels):
        for split in dataset:
            dataset[split] = dataset[split].cast_column('label', class_labels)
        return class_labels.num_classes
    
    def extract_features(self, batch):
        features = self.feature_extractor(batch['speech'], sampling_rate=16000, return_tensors='pt', padding=True)
        batch['input_values'] = features.input_values[0]
        return batch
    
    def compute_metrics(self, eval_pred):
        logits, labels_true = eval_pred
        preds = np.argmax(logits, axis=-1)
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        metrics = {}

        metrics["macro_f1"] = f1_score(labels_true, preds, average="macro")
        metrics["macro_precision"] = precision_score(labels_true, preds, average="macro", zero_division=0)

        # Per-class metrics
        f1_per_class = f1_score(labels_true, preds, average=None, labels=range(self.num_classes))
        prec_per_class = precision_score(labels_true, preds, average=None, labels=range(self.num_classes), zero_division=0)
        print("\nPer-class metrics:")
        for i, label in enumerate(self.class_labels.names):
            metrics[f"f1_{label}"] = f1_per_class[i]
            metrics[f"precision_{label}"] = prec_per_class[i]
            print(f"  {label}: F1={f1_per_class[i]:.4f}, Precision={prec_per_class[i]:.4f}")

        # Confusion matrix
        if len(set(labels_true)) > 1:
            cm = confusion_matrix(labels_true, preds, labels=list(range(probabilities.shape[1])))
            metrics["confusion_matrix"] = cm.tolist()
            
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            cm_percent = np.nan_to_num(cm_percent)

            cm_dir = f"./confusion_matrices_{self.model_name.replace('/', '_')}"
            os.makedirs(cm_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cm_path = os.path.join(cm_dir, f"confusion_matrix_{timestamp}.png")
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                        xticklabels=self.class_labels.names, yticklabels=self.class_labels.names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix (%) - {self.model_name}")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close()
            print(f"Confusion matrix saved to {cm_path}")

        return metrics