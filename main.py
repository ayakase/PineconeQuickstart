from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify
app = Flask(__name__)

model = SentenceTransformer("intfloat/multilingual-e5-large")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]


@app.route('/')
def home():
    return 'running'

@app.route('/<text>')
def transform(text):
    embeddings = model.encode(text).tolist() 
    return jsonify(embeddings=embeddings) 

if __name__ == '__main__':
    app.run(debug=True)

# embeddings = model.encode(sentences)
# print(embeddings.shape)
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
