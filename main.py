from HF_Models import *
from VecDB_Functions import *
from Chat_Functions import *
import chromadb
import os
import time

########### INITIATION
initiationStart = time.time()
customEmbedder = Qwen3Embedder()
chatbot = Qwen3Chat()

persist_dir = './vec_db'
doc_dir = './documents'
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)
if not os.path.exists(doc_dir):
    os.makedirs(doc_dir)
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(
    name="collection",
    embedding_function= customEmbedder
)

initiationEnd = time.time()
print(f"Initiation took {initiationEnd - initiationStart:.2f} seconds")


def help_print():
    print('---------HELP--------')
    print(f"{len(list_documents(collection))} Documents within system")
    print("To run a command, do a forward slash '/' followed by a keyword, then press enter. Commands include:")
    print("/ask followed by a question to begin Q&A")
    print("/list to list all documents")
    print("/import followed by document name to import specific document, or /import all to import all documents to database")
    print("/system to see what hardware the application is running on")
    print("/help to see this again")
    print("/exit to close program\n")
    return

def menu_loop():
    user_input = input()

    if user_input[:4] =="/ask":
        question = user_input.replace("/ask", "").strip()
        ask(question, chatbot, customEmbedder, collection)
        #Initialise chatbot here

    elif user_input == "/list":
        print('\n'.join(list_documents(collection)))

    elif user_input[:7] == "/import":
        docName = user_input.replace("/import", "").strip()
        if docName == "all":
            import_all(collection, doc_dir)
        else:
            try:
                import_doc(collection, doc_dir, docName)
            except Exception as e:
                print(e)

    elif user_input == "/help":
        help_print()
    elif user_input == "/system":
        if torch.cuda.is_available():
            print("CUDA Version: " + str(torch.version.cuda))
            print("VRAM Available:" + str(round(torch.cuda.get_device_properties(0).total_memory/1024**3))+ "GB")
        else:
            print("CUDA not detected, running on CPU")
    elif user_input == "/exit":
        print("Exiting program")
        return False
    else:
        print("Unknown command, please try again")
    print("\n")
    return True


help_print()
while True:
    if menu_loop():
        pass
    else: break