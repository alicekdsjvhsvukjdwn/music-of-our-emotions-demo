from pathlib import Path
import pandas as pd
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

emotion_cols = [
    "Amusing","Annoying","Anxious","tense","Beautiful",
    "Calm","relaxing","serene","Dreamy","Energizing",
    "pump-up","Erotic","desirous"
]

def main():
    vad = pd.read_csv(DATA / "demo_vad.csv")

    rf = joblib.load(MODELS / "random_forest_emotion.pkl")
    X = vad[["arousal", "dominance", "valence"]]
    emo = rf.predict(X)

    emo_df = pd.DataFrame(emo, columns=emotion_cols)
    emo_df.insert(0, "file_name", vad["file_name"].values)

    # --- Normalisation robuste -> distribution ---
    vals = emo_df[emotion_cols].to_numpy(dtype=float)

    # 1) clip négatifs
    vals = np.clip(vals, 0.0, None)

    # 2) si une ligne est toute à 0, fallback softmax sur les valeurs originales
    row_sums = vals.sum(axis=1, keepdims=True)
    zero_rows = (row_sums[:, 0] == 0)

    if np.any(zero_rows):
        raw = pd.DataFrame(emo, columns=emotion_cols).to_numpy(dtype=float)
        # softmax stable
        z = raw[zero_rows]
        z = z - z.max(axis=1, keepdims=True)
        z = np.exp(z)
        z = z / (z.sum(axis=1, keepdims=True) + 1e-12)
        vals[zero_rows] = z
        row_sums = vals.sum(axis=1, keepdims=True)

    # 3) normalisation finale
    probs = vals / (row_sums + 1e-12)
    emo_df[emotion_cols] = probs
    emo_df.to_csv(OUT / "predictions_emotion_demo.csv", index=False)

    # --- Prompt ---
    row = emo_df.iloc[0]
    parts = []
    for e in emotion_cols:
        p = float(row[e]) * 100
        if p >= 10:
            parts.append(f"{p:.1f}% {e.lower()}")

    prompt = "Une musique avec " + ", ".join(parts) + "." if parts else "Une musique neutre sans émotion dominante."
    pd.DataFrame([{"file_name": int(row["file_name"]), "prompt": prompt}]).to_csv(OUT / "prompts_demo.csv", index=False)

    print("OK ✅ (no CNN)")
    print("Somme %:", sum(float(row[e]) for e in emotion_cols) * 100)
    print("Prompt:", prompt)

if __name__ == "__main__":
    main()