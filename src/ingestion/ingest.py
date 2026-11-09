"""
Stage 1: Data Ingestion
Ingests User, Item, and Interaction data from MovieLens 100K and saves raw CSVs.
"""

import os
import sys
import io
import zipfile
import ssl
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logger, load_config

logger = setup_logger("ingestion")

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def ingest_data(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    raw_dir = config["paths"]["raw_dir"]
    url = config["ingestion"]["url"]
    os.makedirs(raw_dir, exist_ok=True)

    logger.info(f"Downloading MovieLens 100K from {url}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    with opener.open(url, timeout=30) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # 1. Parse Interactions (u.data)
        with z.open("ml-100k/u.data") as f:
            interactions = pd.read_csv(
                f, sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"]
            )

        # 2. Parse Users (u.user)
        with z.open("ml-100k/u.user") as f:
            users = pd.read_csv(
                f, sep="|",
                names=["user_id", "age", "gender", "occupation", "zip_code"],
                encoding="latin-1"
            )

        # 3. Parse Items (u.item)
        item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url", "unknown"] + GENRE_NAMES
        with z.open("ml-100k/u.item") as f:
            raw_items = pd.read_csv(f, sep="|", names=item_cols, encoding="latin-1")

        def extract_genres(row):
            active = [g for g in GENRE_NAMES if row[g] == 1]
            return "|".join(active) if active else "Unknown"

        raw_items["genres"] = raw_items.apply(extract_genres, axis=1)
        items = raw_items[["item_id", "title", "release_date", "genres"]].copy()

    # Append a few simulated invalid records to test validation & error quarantine
    invalid_sample = pd.DataFrame([
        {"user_id": 1, "item_id": 1, "rating": 7.0, "timestamp": 881250949},       # Invalid rating (> 5)
        {"user_id": 2, "item_id": 1, "rating": -1.0, "timestamp": 881250949},      # Invalid rating (< 1)
        {"user_id": 999999, "item_id": 1, "rating": 4.0, "timestamp": 881250949},  # Unknown user (FK violation)
        {"user_id": 1, "item_id": 888888, "rating": 4.0, "timestamp": 881250949},  # Unknown item (FK violation)
        {"user_id": 1, "item_id": 2, "rating": None, "timestamp": 881250949},      # Missing rating
        {"user_id": 1, "item_id": 3, "rating": 4.0, "timestamp": -999},            # Invalid timestamp
    ])
    interactions = pd.concat([interactions, invalid_sample], ignore_index=True)

    # Save CSVs
    users.to_csv(os.path.join(raw_dir, "users.csv"), index=False)
    items.to_csv(os.path.join(raw_dir, "items.csv"), index=False)
    interactions.to_csv(os.path.join(raw_dir, "interactions.csv"), index=False)

    logger.info(f"Ingested {len(users)} users, {len(items)} items, and {len(interactions)} interactions.")
    return {"users": len(users), "items": len(items), "interactions": len(interactions)}


if __name__ == "__main__":
    ingest_data()
