from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()
app = Flask(__name__)

pinecone_key = os.getenv('SECRET_KEY')
pc = Pinecone(api_key=pinecone_key)

print(pinecone_key)
model = SentenceTransformer("intfloat/multilingual-e5-large")

@app.route('/')
def home():
    return 'running'

@app.route('/<text>')
def transform(text):
    try:
        embeddings = model.encode(text).tolist()
        return jsonify(embeddings)
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

@app.route('/upsert')
def upsert_vectors():
    index_name = "quickstart"
    try:
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error message with status code 500

if __name__ == '__main__':
    app.run(debug=True)
