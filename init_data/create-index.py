from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
load_dotenv()
pinecone_key = os.getenv('SECRET_KEY')
pc = Pinecone(api_key=pinecone_key)

index_name = "itfield"
pc.create_index(
    name=index_name,
    dimension=1024, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
