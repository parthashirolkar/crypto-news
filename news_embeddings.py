import os
from app import generate_embedding
import argparse
import warnings
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams


warnings.simplefilter(action="ignore")
load_dotenv()


MODEL_NAME = "deepseek-r1:8b"
parser = argparse.ArgumentParser(description="Generate embeddings for news articles")
parser.add_argument("file_path", help="Path to the CSV file containing news articles")
args = parser.parse_args()


def main():
    df = pd.read_csv(args.file_path, parse_dates=["publish_date"])

    client = QdrantClient(
        url=os.environ.get("QDRANT_ENDPOINT"), api_key=os.environ.get("QDRANT_API_KEY")
    )

    if not client.collection_exists("crypto_news"):
        print("Creating collection...")
        client.create_collection(
            collection_name="crypto_news",
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=i,
            vector=generate_embedding(row["article"]),
            payload={
                "title": row["title"],
                "article": row["article"],
                "publish_date": row["publish_date"],
                "keywords": row["keywords"],
                "url": row["url"],
            },
        )
        for i, row in tqdm(df.iterrows(), desc="Creating embeddings", total=len(df))
    ]

    client.upsert(collection_name="crypto_news", points=points)
    print("Done.")


if __name__ == "__main__":
    main()
