from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from chromadb import PersistentClient
import chromadb

def load_documents(file_path):
    loaders = [PyPDFLoader(file_path)]
    docs = []
    for file in loaders:
        docs.extend(file.load())
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def create_embeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})

def initialize_chroma_vector_store(docs, embedding_function, persist_directory='./Chroma_db', collection_name='test'):
    persistent_client = PersistentClient(path=persist_directory)
    collection = persistent_client.get_or_create_collection(collection_name)
    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    vector_store.add_documents(docs)
    return vector_store


if __name__ == "__main__":
    # Load and process documents
    docs = load_documents('./COURSE.pdf')
    docs = split_documents(docs)
    
    # Create embeddings
    embedding_function = create_embeddings()
    
    # Initialize Chroma vector store
    vector_store = initialize_chroma_vector_store(docs, embedding_function)
    
    # Print the count of documents in the vector store
    print(vector_store._collection.count())
    