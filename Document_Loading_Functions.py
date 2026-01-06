import os
import pymupdf
from docx import Document

def read_file(docDir, docName):
    docDict = {}
    fileExt = docName.split(".")[-1]
    docPath = os.path.join(docDir, docName)
    if os.path.isfile(docPath):
        if fileExt == "md" or fileExt == "txt":
            with open(docPath, "r", errors="ignore") as f:
                docDict[docName] = f.read()
        elif fileExt == "pdf":
            doc = pymupdf.open(docPath)
            stringOut = ""
            for page in doc:
                stringOut += page.get_text()
            docDict[docName] = stringOut
        elif fileExt == "docx":
            doc = Document(docPath)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            docDict[docName] = "\n".join(full_text)
    return docDict

def read_dir(docDir):
    docDict = {}
    for docName in os.listdir(docDir):
        docDict = {**docDict, **read_file(docDir, docName)}
    return docDict