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



sentences = ['The computer I am on is called Toaster', 'The computer is running on ToastOS', 'The creator of this code is called Derek']

collection.upsert(documents=sentences, ids=[str(i) for i in range(len(sentences))], metadatas=[{"Name": i} for i in sentences])
question = "What is this computer called?"

results = collection.query(
    query_embeddings=customEmbedder([question]),  # Get the embeddings for the query
    n_results=1  # Number of top results you want to retrieve
)
context = results['documents'][0][0]

chatbot = Qwen3Chat()
print(chatbot(question, context))