import os
import pymupdf

def read_file(docDir, docName):
    docDict = {}
    fileExt = docName.split(".")[-1]
    docPath = os.path.join(docDir, docName)
    if os.path.isfile(docPath):
        if fileExt == "md":
            with open(docPath, "r", errors="ignore") as f:
                docDict[docName] = f.read()
        elif fileExt == "pdf":
            doc = pymupdf.open(docPath)
            stringOut = ""
            for page in doc:
                stringOut += page.get_text()
            docDict[docName] = stringOut
    return docDict

def read_dir(docDir):
    docDict = {}
    for docName in os.listdir(docDir):
        docDict = {**docDict, **read_file(docDir, docName)}
    return docDict