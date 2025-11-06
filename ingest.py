import os
from dotenv import load_dotenv
import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Directory containing policy documents
DOCUMENTS__DIR = "./documents"
# ChromaDB persistent storage directory
PERSIST_DIR = "./chroma_db"
# ChromaDB collection name 
COLLECTION_NAME = "policy_docs"

# 1. Initialize OpenAI Embeddings model
# This model will create the vectors for our text.

try:
    embeddings_model = OpenAIEmbeddings()
except Exception as e:
    print(f"Error initializing OpenAIEmbeddings: {e}")
    print("Please ensure that your OPENAI_API_KEY is set correctly in the .env file.")
    exit()

# 2. Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=PERSIST_DIR)

#Get or create collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"} # the distance metric to use
)


# 3. Load documents from the specified directory
print(f"Loading documents from {DOCUMENTS__DIR}...")
# Use TextLoader to load .txt files
loader = DirectoryLoader(
    DOCUMENTS__DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True
)

docs = loader.load()

if not docs:
    print("No documents found. Please check the DOCUMENTS__DIR path and ensure there are .txt files present.")
    exit()

print(f"Loaded {len(docs)} documents.")


# 4. Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""] # SPlitting by paragraphs first
)
chunks = text_splitter.split_documents(docs)
print(f"Split {len(docs)} documents into {len(chunks)} chunks.")


# 5. Embed and store in ChromaDB
print("Embedding Documents and storing in ChromaDB...")
total_chunks = len(chunks)

for i, chunk in enumerate(chunks):
    # Create embedding for the current chunk
    embedding = embeddings_model.embed_query(chunk.page_content)

    # Store the chunk and its embedding in ChromaDB
    # Create unique id for chunk
    chunk_id = f"{os.path.basename(chunk.metadata['source'])}_{i}"
    collection.add(
        ids=[chunk_id],
        embeddings=embedding,
        documents=[chunk.page_content],
        metadatas=[chunk.metadata] # Storing source file as metadata
    )

    print(f"Processed {i+1}/{total_chunks} chunks. Chunk Id = {chunk_id}", end="\r")

print("\nAll documents have been embedded and stored in ChromaDB.")
print(f"Total Documents Processed: {len(docs)}")
print(f"Total Chunks Created: {len(chunks)}")
print(f"ChromaDB Collection Name: {COLLECTION_NAME}")
print('-'*100)
print(f"Ingestion Complete.")