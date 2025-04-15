from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import warnings

warnings.filterwarnings("ignore")

pdf_path = os.path.join("LA County Permitting", "How to Register for an EPIC-LA Account.pdf")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

print(texts)

clean_texts = [doc.page_content.strip() for doc in texts]
for chunk in clean_texts:
    print(chunk)
