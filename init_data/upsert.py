from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import sys
import os
import uuid
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

pinecone_key = os.getenv('SECRET_KEY')
pc = Pinecone(api_key=pinecone_key)
index_name = os.getenv('INDEX_NAME')

filename = 'vol1chap1.txt'  # Replace with your file name

def chunk_text_file(filename, chunk_size=500, overlap_size=50):
    # Read the file contents
    with open(filename, 'r') as file:
        text = file.read()

    # Split the text into words
    words = text.split()

    # Break into chunks with overlap
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append({
            "id": str(uuid.uuid4()),
            "vol": 7,
            "chapter": 3,
            "text": chunk
        })
        
        # Move forward by chunk_size minus overlap to create overlap
        i += chunk_size - overlap_size

    # Print the number of chunks
    print(f"Total number of chunks: {len(chunks)}")
    return chunks

# Generate chunks with overlap
data = chunk_text_file(filename)

# Embed the text
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(embeddings[0])

# Wait until the index is ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

# Prepare the vectors
vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {
            'text': d['text'],
            "vol": d['vol'],
            "chapter": d['chapter']
        }
    })

# Upsert the vectors
index.upsert(
    vectors=vectors,
    namespace="ns1"
)
