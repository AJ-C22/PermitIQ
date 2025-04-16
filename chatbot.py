import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import warnings

warnings.filterwarnings("ignore")

folder = "LA County Permitting"
all_chunks = []

print(f"Loading PDFs from {folder}...")
for pdf in os.listdir(folder):
    if pdf.endswith(".pdf"):
        print(f"  Processing: {pdf}")
        try:
            loader = PyPDFLoader(os.path.join(folder, pdf))
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"    Error processing {pdf}: {e}")
print("PDF Loading complete.")

print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Creating vector store...")
vectorstore = Chroma.from_documents(all_chunks, embeddings)
print("Vector store created.")

print("Initializing Ollama LLM (Quantized)...")
LLM_MODEL = "mistral"
llm = Ollama(model=LLM_MODEL)

print("Setting up retriever (k=2)...")
K_CHUNKS = 2
retriever = vectorstore.as_retriever(search_kwargs={"k": K_CHUNKS})

print("Creating QA Chain...")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)
print("Ready to answer questions.")

while True:
    query = input("\nAsk something about your PDFs (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    try:
        print("Processing query...")
        answer = qa.run(query)
        print("\n🤖:", answer)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if "Ollama" in str(e) and ("404" in str(e) or "connection refused" in str(e)):
             print(f"  >> Check if the Ollama service is running and the model '{LLM_MODEL}' is pulled.")

print("\nExiting chatbot.")


