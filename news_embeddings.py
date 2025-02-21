import os
import argparse
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import time
from tenacity import retry, stop_after_attempt, wait_exponential

warnings.simplefilter(action="ignore")
load_dotenv()

MODEL_NAME = "deepseek-r1:8b"
COLLECTION_NAME = "crypto_news_chunked"
embedding_function = OllamaEmbeddings(model=MODEL_NAME)
parser = argparse.ArgumentParser(description="Generate embeddings for news articles")
parser.add_argument("file_path", help="Path to the CSV file containing news articles")
args = parser.parse_args()


def chunk_text(text, chunk_size=512, overlap=50):
    """Splits text into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return text_splitter.split_text(text)


def main():
    df = pd.read_csv(args.file_path, parse_dates=["publish_date"])

    client = QdrantClient(
        url=os.environ.get("QDRANT_ENDPOINT"), api_key=os.environ.get("QDRANT_API_KEY")
    )

    # Debug: Print collection status
    if client.collection_exists(COLLECTION_NAME):
        print(
            "Collection exists. Current point count:",
            client.get_collection(COLLECTION_NAME).points_count,
        )
    else:
        print("Creating new collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )

    batch_size = 20

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def upsert_with_retry(client, points):
        return client.upsert(collection_name=COLLECTION_NAME, points=points)

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        points = []

        print(f"\nProcessing batch {i // batch_size + 1} ({i} to {i + len(batch_df)})")

    for _, row in tqdm(
        batch_df.iterrows(), desc="Creating embeddings", total=len(batch_df)
    ):
        try:
            chunks = chunk_text(row["article"])
            for chunk_index, chunk in enumerate(chunks):
                vector = embedding_function.embed_query(
                    f"Title: {row['title']}\nChunk: {chunk}"
                )

                if len(vector) != 4096:
                    print(f"Warning: Unexpected vector dimension: {len(vector)}")
                    continue

                unique_id = uuid.uuid4().int & (1 << 63) - 1
                points.append(
                    PointStruct(
                        id=unique_id,
                        vector=vector,
                        payload={
                            "article_id": row["url"],  # Ensure article_id exists!
                            "chunk_index": chunk_index,
                            "title": row["title"],
                            "chunk": chunk,
                            "publish_date": str(row["publish_date"]),
                            "keywords": row["keywords"],
                            "url": row["url"],
                        },
                    )
                )
        except Exception as e:
            print(f"Error processing row {row.name}: {e}")
            continue

        if points:
            try:
                print(f"Upserting {len(points)} points...")
                upsert_with_retry(client, points)
                time.sleep(1)  # Add a small delay between batches
                print(
                    f"Points after upsert: {client.get_collection(COLLECTION_NAME).points_count}"
                )
            except Exception as e:
                print(f"Error during upsert after all retries: {e}")
                print("Continuing with next batch...")
                continue

    final_count = client.get_collection(COLLECTION_NAME).points_count
    print(f"\nFinal collection point count: {final_count}")


if __name__ == "__main__":
    main()
