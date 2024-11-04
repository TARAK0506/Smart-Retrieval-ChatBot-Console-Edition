import chromadb
from chromadb import PersistentClient
from langchain_chroma import Chroma
def delete_collections(persist_directory='./Chroma_db', collection_name='test'):
    try:
        client = PersistentClient(path=persist_directory)
        
        # List available collections before deletion
        print(f"Available collections before deletion: {client.list_collections()}")
        
        # Delete the specified collection
        client.delete_collection(collection_name)
        print(f"Collection {collection_name} deleted successfully.")
        
        # List available collections after deletion
        print(f"Available collections after deletion: {client.list_collections()}")
    except Exception as e:
        print(f"Unable to delete collection: {e}")
        
 # Delete collections
delete_collections()