Initial

Transform incoming data to embedding and save to Pinecone vector databases

(Just found out that I didnt actually need sentence_transformer for this cuz Pinecone has built-in embedding function, gonna keep for studying purpose anyway)


# .env
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY= pinecone-api-key
INDEX_NAME=your-index-name

update index name, sample data

run create index.py first, then upsert data, then run flask server