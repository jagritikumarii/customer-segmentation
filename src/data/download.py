from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

DATA_URLS = [
    "https://raw.githubusercontent.com/selva86/datasets/master/Mall_Customers.csv",
    "https://raw.githubusercontent.com/ashita03/Mall-Customer-Segmentation-Analysis/master/Mall_Customers.csv",
]

def download_dataset(path: str) -> pd.DataFrame:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return pd.read_csv(destination)

    last_error = None

    for url in DATA_URLS:
        try:
            print(f"Downloading dataset from {url}")
            urlretrieve(url, destination)
            return pd.read_csv(destination)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to download the Mall Customers dataset."
    ) from last_error

if __name__ == "__main__":
    df = download_dataset("data/raw/Mall_Customers.csv")
    print(f"Loaded {len(df)} customers.")
