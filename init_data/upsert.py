from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sampleDataset import pinecone_sample_data

load_dotenv()
pinecone_key = os.getenv('SECRET_KEY')
pc = Pinecone(api_key=pinecone_key)
index_name = os.getenv('INDEX_NAME')
data = pinecone_sample_data
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(embeddings[0])

while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)