"""Download the immutable source archive without credentials or paid services."""
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/resolve/dff09c709999b5b1f1f1806024811c65a6b14dfe/LEVIR-MCI-dataset.zip"
SIZE = 2771943190

if __name__ == "__main__":
    dest = ROOT / "data/raw/LEVIR-MCI-dataset.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size != SIZE:
            raise ValueError("Existing archive has unexpected size; preserve and inspect it")
        print("Source archive already present; use audit to verify SHA-256")
    else:
        with requests.get(URL, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with dest.with_suffix(".part").open("wb") as f:
                for chunk in response.iter_content(4 << 20):
                    f.write(chunk)
        if dest.with_suffix(".part").stat().st_size != SIZE:
            raise ValueError("Incomplete download retained as .part")
        dest.with_suffix(".part").rename(dest)
        print(dest)
