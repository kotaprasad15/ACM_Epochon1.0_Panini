import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Test Qdrant connection
url = "http://localhost:6333"
try:
    client = QdrantClient(url=url)
    print(client.get_collections())
    print("Qdrant connection successful!")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    exit()

# Load embeddings
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
print("Embeddings model loaded successfully!")

# Load documents
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)
print(f"First split text: {texts[1]}")

# Store in Qdrant
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)
print("Vector DB Successfully Created!")
