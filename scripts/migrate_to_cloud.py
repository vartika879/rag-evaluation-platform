"""
Migrate all collections from local Qdrant to Qdrant Cloud.

Usage:
    1. Set QDRANT_URL and QDRANT_API_KEY in .env
    2. Run: python scripts/migrate_to_cloud.py
"""
import os
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import (
    QDRANT_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_DIM,
)


def migrate():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env")
        sys.exit(1)

    print(f"Source (local): {QDRANT_PATH}")
    print(f"Target (cloud): {QDRANT_URL}")
    print()

    # Connect to both
    local = QdrantClient(path=str(QDRANT_PATH))
    cloud = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    # List local collections
    collections = [c.name for c in local.get_collections().collections]
    print(f"Found {len(collections)} collections: {collections}")
    print()

    for coll_name in collections:
        print(f"--- Migrating '{coll_name}' ---")

        info = local.get_collection(coll_name)
        points_count = info.points_count
        print(f"  Points: {points_count}")

        # Create collection in cloud
        existing = [c.name for c in cloud.get_collections().collections]
        if coll_name in existing:
            print(f"  Collection already exists in cloud — deleting first")
            cloud.delete_collection(coll_name)

        cloud.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

        # Scroll through local points in batches
        batch_size = 100
        offset = None
        migrated = 0

        while True:
            points, offset = local.scroll(
                collection_name=coll_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            cloud_points = [
                PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=p.payload,
                )
                for p in points
            ]

            cloud.upsert(collection_name=coll_name, points=cloud_points)
            migrated += len(cloud_points)
            print(f"  Uploaded {migrated}/{points_count}")

            if offset is None:
                break

        print(f"  ✓ '{coll_name}' migrated ({migrated} points)\n")

    # Verify
    print("=" * 60)
    print("Verification:")
    cloud_collections = [c.name for c in cloud.get_collections().collections]
    for coll_name in collections:
        info = cloud.get_collection(coll_name)
        print(f"  {coll_name}: {info.points_count} points in cloud")

    print()
    print("✅ Migration complete!")
    print()
    print("Next: update your .env to use cloud mode:")
    print(f"  QDRANT_URL={QDRANT_URL}")
    print(f"  QDRANT_API_KEY=<your key>")

    local.close()
    cloud.close()


if __name__ == "__main__":
    migrate()