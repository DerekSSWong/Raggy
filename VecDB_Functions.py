import time
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os
from HF_Models import *
from Document_Loading_Functions import *

def ingest_doc(coll, docDict):
    for i in docDict.keys():
        print(f'Ingesting {i}')
        delete_doc(coll, i)
        ingestStart = time.time()
        chunked_text = chunk_documents(docDict[i], 1024, 64)
        batch_upsert(coll, chunked_text, 5,i)
        ingestEnd = time.time()
        print(f"Finished ingesting {i}, took {ingestEnd - ingestStart:.2f} seconds")

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
