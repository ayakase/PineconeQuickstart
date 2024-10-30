from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
load_dotenv()
pinecone_key = os.getenv('SECRET_KEY')
pc = Pinecone(api_key=pinecone_key)
index_name = "quickstart"
data = [
    {"id": "vec7", "text": "Linux distributions like Ubuntu, Fedora, and Debian are popular choices for desktop and server environments."},
    {"id": "vec8", "text": "Android's user interface is designed to be intuitive, allowing users to easily navigate their devices."},
    {"id": "vec9", "text": "The Linux community thrives on collaboration, with many developers contributing to various open-source projects."},
    {"id": "vec10", "text": "Android supports multiple programming languages, with Java and Kotlin being the most commonly used for app development."},
    {"id": "vec11", "text": "Linux is known for its security features, making it a preferred choice for many enterprises."},
    {"id": "vec12", "text": "The Android ecosystem is vast, with millions of apps available on the Google Play Store."},
    {"id": "vec13", "text": "Linux commands are powerful tools that allow users to perform complex tasks through the command line."},
    {"id": "vec14", "text": "Custom ROMs for Android devices allow users to enhance their experience with additional features and optimizations."},
    {"id": "vec15", "text": "Many developers prefer Linux for programming due to its support for various languages and development tools."},
    {"id": "vec16", "text": "Android and Linux together empower a range of devices, from smartphones to embedded systems."}
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