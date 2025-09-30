import glob
import os

import numpy as np
import pandas as pd
import torchaudio
from pydub import AudioSegment


class DataPreprocessor:
    @staticmethod
    def clean_txt_file(input_txt, valid_labels={"mb", "sb", "v", "b", "h", "n"}):
        df = pd.read_csv(input_txt, sep="\t", names=["start", "end", "label"])
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df[df["label"].isin(valid_labels)]
        df["label"] = df["label"].replace({"b": "sb", "v": "crs", "n": "crs"})
        return df

    @staticmethod
    def adaptive_preprocess(df):
        df = df.sort_values("start").reset_index(drop=True)
        df["duration"] = df["end"] - df["start"]
        medians = df.groupby("label")["duration"].median().to_dict()
        processed = []
        i = 0
        while i < len(df):
            row = df.iloc[i]
            label, start, end = row["label"], row["start"], row["end"]
            duration = row["duration"]
            median_dur = medians[label]

            if label in ["mb", "sb"]:
                while i + 1 < len(df) and df.iloc[i + 1]["label"] == label and (end - start) < median_dur:
                    i += 1
                    end = df.iloc[i]["end"]
                processed.append({"start": start, "end": end, "label": label})
            elif label in ["h", "crs"]:
                while duration > median_dur:
                    processed.append({"start": start, "end": start + median_dur, "label": label})
                    start += median_dur
                    duration = end - start
                if duration > 0:
                    processed.append({"start": start, "end": end, "label": label})
            else:
                processed.append({"start": start, "end": end, "label": label})
            i += 1
        return pd.DataFrame(processed)

    @staticmethod
    def process_folder(input_folder, output_folder, min_duration=0.1):
        os.makedirs(output_folder, exist_ok=True)
        all_rows = []

        txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
        
        for txt_file in txt_files:
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            audio_candidates = [f for f in os.listdir(input_folder)
                                if f.startswith(base_name) and f.lower().endswith(('.wav', '.mp3', '.flac'))]
            if not audio_candidates:
                print(f"⚠️ Skipping {base_name}, no matching audio found.")
                continue
            audio_file = os.path.join(input_folder, audio_candidates[0])

            df = DataPreprocessor.clean_txt_file(txt_file)
            processed_df = DataPreprocessor.adaptive_preprocess(df)

            audio = AudioSegment.from_file(audio_file)

            for idx, row in processed_df.iterrows():
                duration = row["end"] - row["start"]
                if duration < min_duration:
                    continue
                start_ms = int(row["start"] * 1000)
                end_ms = int(row["end"] * 1000)
                label = row["label"]
                sub_audio = audio[start_ms:end_ms]
                sub_audio_filename = f"{base_name}_{idx}_{label}.wav"
                sub_audio_path = os.path.join(output_folder, sub_audio_filename)
                sub_audio.export(sub_audio_path, format="wav")

                all_rows.append({
                    "audio_path": sub_audio_path,
                    "label": label,
                    "parent_id": base_name,
                    "duration": duration
                })

        csv_path = os.path.join(output_folder, "segments.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"✅ CSV saved at: {csv_path}")
        return pd.DataFrame(all_rows)
    
    @staticmethod
    def preprocess_audio(batch, min_duration=0.2):
        audio_path = batch['audio_path']
        audio_path = os.path.normpath(batch['audio_path'])
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            audio_np = audio.squeeze().numpy().astype(np.float32)
            if len(audio_np) < 16000 * min_duration:
                audio_np = np.zeros(int(16000 * min_duration), dtype=np.float32)
            batch['speech'] = audio_np
        except Exception as e:
            print(f"⚠️ Could not load {audio_path}: {e}")
            batch['speech'] = np.zeros(int(16000 * min_duration), dtype=np.float32)
        return batch
