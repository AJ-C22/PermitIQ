import os
import warnings
import shutil
import torch
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['dbapi2'] = pysqlite3.dbapi2
    print("✅ Patched sqlite3 with pysqlite3 (modern SQLite for Chroma)")
except ImportError:
    print("❌ pysqlite3 not available. Chroma may not work due to old SQLite version.")

warnings.filterwarnings("ignore")

# --- Configuration ---
PDF_FOLDER = "LA County Permitting"
# <<< Define where to save the persistent database >>>
PERSIST_DIRECTORY = "chroma_db_permitting"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# --- End Configuration ---

# Check device for embeddings
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for embeddings: {device}")

# Clean up old database directory if it exists
if os.path.exists(PERSIST_DIRECTORY):
    print(f"Removing existing vector store at {PERSIST_DIRECTORY}")
    shutil.rmtree(PERSIST_DIRECTORY)

all_chunks = []
print(f"Loading PDFs from: {PDF_FOLDER}")

# Ensure the PDF folder exists
if not os.path.isdir(PDF_FOLDER):
    print(f"Error: PDF folder '{PDF_FOLDER}' not found.")
    print("Please create the folder and add your PDF files.")
    exit()

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

if not pdf_files:
    print(f"Error: No PDF files found in the folder '{PDF_FOLDER}'.")
    exit()

for pdf in pdf_files:
    print(f"  Processing: {pdf}")
    try:
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, pdf))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    except Exception as e:
        print(f"    Error processing {pdf}: {e}")

if not all_chunks:
    print("Error: No text chunks could be extracted from the PDFs.")
    exit()

print(f"\nTotal chunks extracted: {len(all_chunks)}")

print(f"Initializing embeddings model: {EMBEDDING_MODEL}")
# Allow automatic device detection or specify if needed
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    # model_kwargs={'device': device} # Uncomment if auto-detection fails
)


print(f"Creating and persisting vector store at: {PERSIST_DIRECTORY}")
# Create Chroma with persist_directory
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

print("\nVector store created and saved successfully!") 