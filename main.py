from CodeBuffer import *
import chromadb
import os
import uuid

########### INITIATION
customEmbedder = Qwen3Embedder()
chatbot = Qwen3Chat()

persist_dir = './vec_db'
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(
    name="collection",
    embedding_function= customEmbedder
)


########## DOC FORMAT AND INGEST
def split_string_by_length(s, length):
    if length <= 0:
        raise ValueError("Length must be a positive integer")

    return [s[i:i+length] for i in range(0, len(s), length)]

def batch_upsert(coll, documents, batch_size):
    chunks = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    for i in chunks:
        coll.upsert(documents=i, ids=[str(uuid.uuid4()) for _ in range(len(i))])

docPath = "./documents/Dracula.md"
docString = open(docPath, 'r').read()

sentences = split_string_by_length(docString, 512)

# collection.upsert(documents=sentences, ids=[str(i) for i in range(len(sentences))])
batch_upsert(collection, sentences, 10)




##########RETRIEVAL AND CHATTING
question = "Who's journal is this"

results = collection.query(
    query_embeddings=customEmbedder([question]),  # Get the embeddings for the query
    n_results=10  # Number of top results you want to retrieve
)

context = "" .join(results['documents'][0])


print(chatbot(question, context))