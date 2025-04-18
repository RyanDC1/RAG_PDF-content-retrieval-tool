from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

""" 
Specify documents
example: 
documentLoaders = [PyPDFLoader('./Doc1.pdf'), PyPDFLoader('./Doc2.pdf')...]
"""
documents = [PyPDFLoader('./Context.pdf')]

docs = []

for doc in documents:
    loaded_docs = doc.load()
    print(f"Loaded {len(loaded_docs)} documents")

    docs.extend(loaded_docs)

# Split the files into chunks using Langchain text splitter
# Chunking will make it easier to generate vector embeddings of large files
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=256
)
docs = text_splitter.split_documents(docs)
print(f"Number of chunks: {len(docs)}")


# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    show_progress=True
)

# store embeddings
vector_store = Chroma.from_documents(
    docs, 
    embedding, 
    persist_directory='./chroma_db_vectors',
)

print(f"Total embeddings added to vectorDB: {vector_store._collection.count()}")