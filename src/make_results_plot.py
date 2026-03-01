from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "outputs" / "predictions_emotion_demo.csv")

row = df.iloc[0].drop("file_name").sort_values(ascending=False)
row = row[row > 0.01].head(6)  # garde seulement >1% et top 6

plt.figure()
plt.bar(row.index, row.values * 100)
plt.ylabel("Pourcentage (%)")
plt.title("Top émotions prédites (démo offline)")
plt.xticks(rotation=35, ha="right")
plt.ylim(0, max(5, (row.values.max() * 100) * 1.15))
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "top_emotions.png", dpi=220)