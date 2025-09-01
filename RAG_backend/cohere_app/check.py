
from pymilvus import connections, Collection, MilvusClient
collection_name="QA"

# Use the same approach as collection_files to get consistent results
connections.connect("default", host="172.16.34.233", port=19530)
collection = Collection(collection_name)
collection.load()
iterator = collection.query_iterator(filter="",batch_size=1000, output_fields=["source"])
results = []
i=0
while True:
    print(i)
    result = iterator.next()
    i+1
    if not result:
        iterator.close()
        break
    results.extend(result)
file_names = list(set([result['source'] for result in results]))
print(file_names)
