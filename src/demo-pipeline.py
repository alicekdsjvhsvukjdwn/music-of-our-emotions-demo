from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.layers import InputLayer

# --- Patch compat: vieux modèles .h5 qui utilisent `batch_shape` ---
_old_from_config = InputLayer.from_config

@classmethod
def _patched_from_config(cls, config):
    # Certains modèles stockent batch_shape, mais la version Keras actuelle attend input_shape
    if "batch_shape" in config and "input_shape" not in config:
        bs = config.pop("batch_shape")  # ex: [None, 18, 256, 1]
        if isinstance(bs, (list, tuple)) and len(bs) >= 2:
            config["input_shape"] = tuple(bs[1:])  # -> (18,256,1)
    return _old_from_config(config)

InputLayer.from_config = _patched_from_config
# ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def build_input_tensor(eeg_csv, gsr_csv, rsp_csv):
    # === EEG: ton fichier eeg_features1.csv est en "flat" (120 valeurs attendues dans ton notebook)
    eeg_features = pd.read_csv(eeg_csv, delimiter=",")
    # garde uniquement la 1ère ligne (comme ton notebook)
    row = eeg_features.iloc[0].values.astype(float)

    # EEG: 16 channels * 5 bands = 80
    # Asymmetry: 8 pairs * 5 bands = 40
    # => total EEG = 120
    # Si ton csv contient exactement ça, row.shape devrait être >= 120.
    if row.shape[0] < 120:
        raise ValueError(f"EEG row trop courte ({row.shape[0]} cols). Attendu >= 120.")

    eeg_80 = row[:80].reshape(16, 5)
    asym_40 = row[80:120].reshape(8, 5)

    # === GSR / Resp (comme ton notebook: première ligne après header)
    gsr = pd.read_csv(gsr_csv, header=None, skiprows=1).iloc[0].astype(float).values
    rsp = pd.read_csv(rsp_csv, header=None, skiprows=1).iloc[0].astype(float).values

    combined = np.hstack([
        eeg_80.flatten(),
        asym_40.flatten(),
        gsr.flatten(),
        rsp.flatten()
    ])

    # ton notebook pad -> reshape (18,256) -> expand dims -> batch
    combined = np.pad(combined, (0, 18*256 - combined.size), mode="constant")
    x = combined.reshape(18, 256)
    x = np.expand_dims(x, axis=-1)      # (18,256,1)
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)  # safe norm
    x = np.expand_dims(x, axis=0)       # (1,18,256,1)
    return x

def main():
    model = tf.keras.models.load_model(
        MODELS / "final_model_optimized.h5",
        custom_objects={"mse": MeanSquaredError()},
        compile=False
    )

    x = build_input_tensor(
        DATA / "eeg_features1.csv",
        DATA / "gsr_features1.csv",
        DATA / "rsp_features1.csv",
    )

    preds = model.predict(x, verbose=0)

    # Si ton modèle a 3 sorties (liste/tuple)
    if isinstance(preds, (list, tuple)) and len(preds) == 3:
        valence_pred, arousal_pred, dominance_pred = preds
        valence = float(valence_pred[0])
        arousal = float(arousal_pred[0])
        dominance = float(dominance_pred[0])
    else:
        # Si sortie unique de shape (1,3)
        preds = np.array(preds).reshape(-1)
        if preds.size != 3:
            raise ValueError(f"Sortie modèle inattendue: {np.array(preds).shape}")
        valence, arousal, dominance = map(float, preds)

    vad = pd.DataFrame([{
        "file_name": 1,
        "valence": valence,
        "arousal": arousal,
        "dominance": dominance
    }])
    vad.to_csv(OUT / "predictions_vad.csv", index=False)

    # 2) RandomForest : VAD -> 13 émotions
    rf = joblib.load(MODELS / "random_forest_emotion.pkl")
    X = vad[["arousal", "dominance", "valence"]]
    emo = rf.predict(X)

    emotion_cols = [
        "Amusing","Annoying","Anxious","tense","Beautiful",
        "Calm","relaxing","serene","Dreamy","Energizing",
        "pump-up","Erotic","desirous"
    ]
    emo_df = pd.DataFrame(emo, columns=emotion_cols)
    emo_df.insert(0, "file_name", 1)

    # normalisation (comme ton notebook)
    emo_vals = emo_df[emotion_cols]
    emo_df[emotion_cols] = emo_vals.div(emo_vals.sum(axis=1), axis=0)

    emo_df.to_csv(OUT / "predictions_emotion_demo.csv", index=False)

    # 3) Prompt simple
    row = emo_df.iloc[0]
    parts = []
    for e in emotion_cols:
        p = float(row[e]) * 100
        if p > 10:
            parts.append(f"{p:.1f}% {e.lower()}")
    prompt = "Une musique avec " + ", ".join(parts) + "." if parts else "Une musique neutre sans émotion dominante."
    pd.DataFrame([{"file_name": 1, "prompt": prompt}]).to_csv(OUT / "prompts_demo.csv", index=False)

    print("OK ✅")
    print("VAD:", (valence, arousal, dominance))
    print("Prompt:", prompt)

if __name__ == "__main__":
    main()