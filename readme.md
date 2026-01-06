# Raggy

A lightweight RAG application. Uses a local Qwen0.6B model to perform embedding and Q&A.

## Features
- Capable of reading text files (.txt), markdowns (.md), word documents (.docx), and .pdf files.
- All models are stored locally, no need to connect to any APIs, ensuring full privacy.
- Can be run on either CPU or GPU, automatically selects suitable mode on setup.

## GPU Requirements (If on GPU mode)
- CUDA 12.6+
- At least 4GB VRAM

## Installation (Windows install script)

- Download and run the latest Python Install Manager from the [official site](https://www.python.org/downloads/windows/).
- Clone this repository, or download it as .zip.
- Run Powershell as admin
- Enter: ``set-executionpolicy remotesigned``
- The command above allows the setup script to be run on your PC.
- **If** you chose to download the project as a zip:
  - unzip it but keep the code in the "Raggy-main" folder
  - Run Powershell within the folder and enter the following command:
  - ``Unblock-File .\setup_windows.ps1, .\run_windows.ps1``
- To start the setup script, right-click on ``setup_windows.ps1`` and run script with Powershell
- After setup, right-click on ``run_windows.ps1`` to run the program.

## How to Use
- Put documents in the documents folder
- Type ``/import`` followed by the document's name (e.g. Paper.pdf), or ``/import all`` to import all documents in the document folder
- After importing the documents, type ``/ask`` followed by a question to begin Q&A
