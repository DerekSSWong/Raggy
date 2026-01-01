import time
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os
from HF_Models import *
from Document_Loading_Functions import *

#VRAM:Batch Size
#4:1
#8:8
#12:16
def ingest_doc(coll, docDict):

    batch_size = 1
    if torch.cuda.is_available():
        gb_available = round(torch.cuda.get_device_properties(0).total_memory/1024**3)
        if gb_available >= 12:
            batch_size = 16
        elif gb_available >= 8:
            batch_size = 8

    for i in docDict.keys():
        print(f'Ingesting {i}, batch size: {batch_size}')
        delete_doc(coll, i)
        ingestStart = time.time()
        chunked_text = chunk_documents(docDict[i], 500, 100)
        batch_upsert(coll, chunked_text, batch_size,i)
        ingestEnd = time.time()
        print(f"Finished importing {i}, took {ingestEnd - ingestStart:.2f} seconds")

def import_all(coll, docDir):
    docDict = read_dir(docDir)
    ingest_doc(coll, docDict)

def import_doc(coll, docDir, docName):
    docDict = read_file(docDir, docName)
    ingest_doc(coll, docDict)


#Turns doc string into an array of substrings
def chunk_documents(docString, chunkSize, chunkOverlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    chunks = text_splitter.split_text(docString)
    return chunks

#Imports chunked document into vector db
def batch_upsert(coll, documents, batch_size, docTitle):
    chunks = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    metaDataDict = {"Name": docTitle}
    for i in chunks:
        coll.add(documents=i, ids=[str(uuid.uuid4()) for _ in range(len(i))], metadatas= [metaDataDict.copy() for _ in range(len(i))])

def list_documents(coll):
    results = coll.get(include=["metadatas"])
    # keys = {k for meta in results["metadatas"] for k in meta.keys()}
    dictList = results["metadatas"]
    strSet = set([i['Name'] for i in dictList])
    return strSet

def delete_doc(coll, docName):
    qOut = coll.get(where={'Name':docName})
    if len(qOut['ids']) > 0:
        coll.delete(ids = qOut['ids'])
