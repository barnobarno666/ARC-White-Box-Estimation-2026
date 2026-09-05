import os
import sys
import shutil
import urllib.request
from pathlib import Path

DEST_DIR = Path(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini")
DEST_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = DEST_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Copy metadata.json and README.md if available locally
SNAPSHOT_DIR = Path(r"C:\Users\defaultuser0.LAPTOP-LRB3T941\.cache\huggingface\hub\datasets--aicrowd--arc-whestbench-public-2026\snapshots\aa99830fdc09fad15407b10e8e3459d3e18bba0a")
for fname in ["metadata.json", "README.md"]:
    src = SNAPSHOT_DIR / fname
    dst = DEST_DIR / fname
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"Copied {fname} to {dst}")

# Download mini-00000-of-00007.parquet
url = "https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/v2-phase2/data/mini-00000-of-00007.parquet"
target_file = DATA_DIR / "mini-00000-of-00007.parquet"
temp_file = DATA_DIR / "mini-00000-of-00007.parquet.part"

if target_file.exists():
    print(f"File already exists: {target_file} ({target_file.stat().st_size} bytes)")
    sys.exit(0)

print(f"Streaming from {url} to {target_file}...")
req = urllib.request.Request(url, headers={"User-Agent": "whest-downloader"})
with urllib.request.urlopen(req) as resp, open(temp_file, "wb") as f:
    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 10 * 1024 * 1024  # 10 MB chunks
    last_print = 0

    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        f.write(chunk)
        downloaded += len(chunk)
        if downloaded - last_print >= 50 * 1024 * 1024 or downloaded == total_size:
            pct = (downloaded / total_size * 100) if total_size else 0
            print(f"Downloaded: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({pct:.1f}%)")
            last_print = downloaded

temp_file.rename(target_file)
print(f"Successfully downloaded {target_file} ({target_file.stat().st_size} bytes)")
