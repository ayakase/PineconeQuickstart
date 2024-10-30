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

# @app.route('/upsert')
# def upsert_vectors():
#     index_name = "quickstart"
#     data = [
#         {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
#         {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
#         {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
#         {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
#         {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
#         {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
#     ]

#     embeddings = pc.inference.embed(
#         model="multilingual-e5-large",
#         inputs=[d['text'] for d in data],
#         parameters={"input_type": "passage", "truncate": "END"}
#     )
#     print(embeddings[0])
#     # Wait for the index to be ready
#     while not pc.describe_index(index_name).status['ready']:
#         time.sleep(1)

#     index = pc.Index(index_name)

#     vectors = []
#     for d, e in zip(data, embeddings):
#         vectors.append({
#             "id": d['id'],
#             "values": e['values'],
#             "metadata": {'text': d['text']}
#         })

#     index.upsert(
#         vectors=vectors,
#         namespace="ns1"
#     )
#     return "ok"
    # index_name = "quickstart"
    # try:
    #     pc.create_index(
    #         name=index_name,
    #         dimension=1024,
    #         metric="cosine",  # Replace with your model metric
    #         spec=ServerlessSpec(
    #             cloud="aws",
    #             region="us-east-1"
    #         )
    #     )
    #     return jsonify({"status": "ok"})
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500  # Return error message with status code 500


@app.route('/query/<text>')
def query(text):
    if not text:
        return 'empty'
    index_name = "quickstart"
    index = pc.Index(index_name)
    query = text
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="ns1",
        vector=embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    print(results.matches)
    print(f"Type of results.matches: {type(results.matches)}, Content: {results.matches}")
    clean_matches = [
        {
            "id": match.get("id"),
            "metadata": match.get("metadata"),
            "score": match.get("score"),
            "values": match.get("values") if isinstance(match.get("values"), list) else []
        }
        for match in results.matches
    ]

    # Return the results as JSON
    return jsonify({
        "message": "ok",
        "matches": clean_matches  # Use cleaned matches
    })

if __name__ == '__main__':
    app.run(debug=True)
