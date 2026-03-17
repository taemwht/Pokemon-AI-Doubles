"""
Download the gen9vgc2026regi-1760 chaos JSON from the Smogon stats server.
This is the master meta-data source for StatInference (item/ability priors).
"""
import json
import urllib.request
from pathlib import Path

# Smogon stats: latest chaos folder (update month if needed)
BASE_URL = "https://www.smogon.com/stats/2026-02/chaos"
FILENAME = "gen9vgc2026regi-1760.json"
OUTPUT_DIR = Path(__file__).resolve().parent


def download_chaos_json(
    url_path: str = f"{BASE_URL}/{FILENAME}",
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or OUTPUT_DIR / FILENAME
    print(f"Downloading {url_path} ...")
    req = urllib.request.Request(url_path, headers={"User-Agent": "Pokemon-AI-Doubles/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    # Smogon may return gzipped; if so decompress
    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)
    data = json.loads(raw.decode("utf-8"))
    out_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"Saved to {out_path} ({len(data.get('data', {}))} Pokémon).")
    return out_path


if __name__ == "__main__":
    download_chaos_json()
