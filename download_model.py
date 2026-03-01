from pathlib import Path
import urllib.request

REPO_OWNER = "aliceckdsjvhsvukjdwn"
REPO_NAME  = "music-of-our-emotions-demo"
TAG        = "v0.1"
ASSET_NAME = "random_forest_emotion.pkl"

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / ASSET_NAME

    url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{TAG}/{ASSET_NAME}"
    print("Downloading:", url)
    urllib.request.urlretrieve(url, out_path)
    print("Saved to:", out_path)

if __name__ == "__main__":
    main()
