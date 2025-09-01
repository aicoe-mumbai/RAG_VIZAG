#!/usr/bin/env python3
# milvus_export.py

import logging
from pymilvus import connections, Collection
import pandas as pd
from sqlalchemy import create_engine

# 1) (Optional) suppress the "failed to get mvccTs" warning
logging.getLogger("pymilvus.client.grpc_handler").setLevel(logging.ERROR)

# 2) Connect to Milvus
HOST = "http://172.16.34.233"
PORT = "19530"
connections.connect(alias="default", host=HOST, port=PORT)

# 3) Reference your collection and trigger compaction
COLLECTION_NAME = "test_sdc"
collection = Collection(name=COLLECTION_NAME)

print(f"Triggering compaction on '{COLLECTION_NAME}'...")
compaction_id = collection.compact()

# 4) Block until compaction finishes
print(f"Waiting for compaction job {compaction_id} to complete...")
collection.wait_for_compaction_completed()

# 5) Load collection into memory
collection.load()

# 6) Stream all entities via a QueryIterator
print("Streaming data via query_iterator()...")
iterator = collection.query_iterator(
    batch_size=1000,              # fetch 1000 entities per round
    expr="",                      # no filter, get everything
    output_fields=["source", "page", "text", "pk"],
    # limit=-1 by default means "all matching entities"
)

all_data = []
while True:
    batch = iterator.next()
    if not batch:
        iterator.close()
        break
    all_data.extend(batch)

# 7) Convert to DataFrame and export to SQLite
print(f"Fetched {len(all_data)} records. Exporting to SQLite...")
df = pd.DataFrame(all_data)
engine = create_engine('sqlite:///test_sdc_test_exported_collection.db')
df.to_sql('test_sdc', con=engine, if_exists='replace', index=False)

print("✅ Data successfully exported to 'test_sdc_test_exported_collection.db' (table: test_sdc).")
