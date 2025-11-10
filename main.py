from CodeBuffer import *
import chromadb
import os

persist_dir = './vec_db'
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

client = chromadb.PersistentClient(path=persist_dir)
customEmbedder = Qwen3Embedder()
collection = client.get_or_create_collection(
    name="collection",
    embedding_function= customEmbedder
)



sentences = ['test', 'I am a prickly pear', 'morbius', 'customEmbedder', 'hello world']

collection.upsert(documents=sentences, ids=[str(i) for i in range(len(sentences))], metadatas=[{"Name": i} for i in sentences])

results = collection.query(
    query_embeddings=customEmbedder(['I like to try things']),  # Get the embeddings for the query
    n_results=2  # Number of top results you want to retrieve
)
print(results)