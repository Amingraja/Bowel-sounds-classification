from typing import List

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

from app.services.config import LABEL2IDS
from app.schemas.prediction_schemas import PredictionRequestSchema, PredictionResponseSchema

class PredictionService:
    def __init__(self, model_infos):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_infos = model_infos

    def predict(self, request: PredictionRequestSchema) -> List[PredictionResponseSchema]:
        predictions = self.predict_multi_event(request.audio_path)

        output = []
        for prediction in predictions:
            output.append(
                PredictionResponseSchema(
                    start_time=prediction["start"],
                    end_time=prediction["end"],
                    label=prediction["label"]
                )
            )

        return output
    
    def prepare_models_extractors(self):
        models, extractors = [], []
        for info in self.model_infos:
            m = Wav2Vec2ForSequenceClassification.from_pretrained(info.PATH)
            m.eval().to(self.device)
            models.append(m)

            fe = AutoFeatureExtractor.from_pretrained(info.NAME)
            extractors.append(fe)

        return models, extractors
    
    def predict_segment(self, audio_segment, sr=16000):
        audio_np = audio_segment.numpy().astype(np.float32)
        probs = []
        id2label = {v: k for k, v in LABEL2IDS.items()}

        models, extractors = self.prepare_models_extractors()
        with torch.no_grad():
            for m, fe in zip(models, extractors):
                inputs = fe(audio_np, sampling_rate=sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = m(**inputs).logits
                probs.append(torch.nn.functional.softmax(logits, dim=-1).cpu().numpy())

        avg_prob = np.mean(probs, axis=0)
        return id2label[np.argmax(avg_prob, axis=-1).item()]

    @staticmethod
    def merge_predictions(pred_segments):
        if not pred_segments:
            return []
        merged = [pred_segments[0].copy()]
        for seg in pred_segments[1:]:
            if seg["label"] == merged[-1]["label"] and seg["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        return merged

    def predict_multi_event(self, audio_path, window_size=0.1, stride=0.1):
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sr = 16000
        audio = audio.squeeze()
        duration = audio.shape[0] / sr

        predictions = []
        start_time = 0.0
        while start_time < duration:
            end_time = min(start_time + window_size, duration)
            segment = audio[int(start_time*sr):int(end_time*sr)]
            pred_label = self.predict_segment(segment, sr)
            predictions.append({"start": start_time, "end": end_time, "label": pred_label})
            start_time += stride

        return self.merge_predictions(predictions)

    @staticmethod
    def save_predictions_txt(pred_segments, output_path):
        with open(output_path, "w") as f:
            for seg in pred_segments:
                f.write(f"{seg['start']:.6f}\t{seg['end']:.6f}\t{seg['label']}\n")
        print(f"Predictions saved to {output_path}")