from HF_Models import *


def ask(question, chatbot, embedder, collection):

    results = collection.query(
        query_embeddings=embedder([question]),  # Get the embeddings for the query
        n_results=10  # Number of top results you want to retrieve
    )

    context = "" .join(results['documents'][0])

    print(chatbot(question, context))