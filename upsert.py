from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
load_dotenv()
pinecone_key = os.getenv('SECRET_KEY')
pc = Pinecone(api_key=pinecone_key)
index_name = "quickstart"
data =[
    {"id": "vec37", "text": "TempleOS is a lightweight operating system created by Terry A. Davis, known for its unique programming language and biblical themes."},
    {"id": "vec38", "text": "Designed to be a simple and efficient environment, TempleOS features a 16-color display and a 640x480 resolution, emphasizing minimalism and performance."}
]



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