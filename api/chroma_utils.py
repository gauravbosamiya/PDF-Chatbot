from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
import os
from dotenv import load_dotenv


load_dotenv()
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

def load_and_split_document(file_path:str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_spliter.split_documents(documents)


def index_document_to_chroma(file_path:str, file_id:int)->bool:
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata['file_id'] = file_id
            
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document {e}")
        return False
    
def delete_doc_from_chroma(file_id:int):
    try:
        doc = vectorstore.get(where={"file_id":file_id})
        print(f"Found {len(doc['ids'])} document chunk for file id {file_id}")

        vectorstore._collection.delete(where={"file_id":file_id})
        print(f"Deleted all document with file_id {file_id}")

        return True
    except Exception as e:
        print(f"Error deleting document with file_id: {file_id} from chroma: {str(e)}")
        return False