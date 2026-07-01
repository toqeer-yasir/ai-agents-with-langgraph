import os
import hashlib
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class Rag:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "rag_docs"):
        """Initializing or creating db, creating splitter object"""
        self.embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_store = Chroma(
            collection_name = collection_name,
            embedding_function = self.embedding_fn,
            persist_directory = db_path
        )
        
        is_new_store = (len(self.vector_store.get()["ids"]) == 0)
        
        if is_new_store:
            print(f"Database initialized successfully! Ready for ingestion.")
        else:
            print(f"Successfully connected to database heaving {len(self.vector_store.get()['ids'])} chunks.")
        
        self.splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50, 
        separators = ["\n\n", "\n", " ", ""],
        )


    def _load_documents(self, doc_path: str):
        if str.lower(doc_path).endswith(".txt"):
            doc_content = TextLoader(doc_path).load()
        
        elif str.lower(doc_path).endswith(".pdf"):
            doc_content = PyPDFLoader(doc_path).load()
            
        elif str.lower(doc_path).endswith((".doc", ".docx")):
            doc_content = Docx2txtLoader(doc_path).load()
        
        elif str.lower(doc_path).endswith(".md"):
            doc_content = UnstructuredMarkdownLoader(doc_path).load()
        
        elif str.lower(doc_path).endswith(".csv"):
            doc_content = CSVLoader(doc_path).load()

        else:
            raise TypeError(f"Uploaded file type '{doc_path}' is not supported by RAG!")

        return doc_content
    
    
    def _calculate_file_hash(self, doc_path: str):
        hasher = hashlib.md5()
        with open(doc_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()


    def _split_documents(self, doc_content):
        return self.splitter.split_documents(doc_content)
    
    
    def _add_documents(self, docs: list[str], doc_path):
        chunk_ids = []
        file_signature = self._calculate_file_hash(doc_path)

        for doc in docs:
            doc.metadata["file_signature"] = file_signature
            source_file = doc.metadata.get("source", "unknown_file")
            text_content = doc.page_content
            
            chunk_hash = hashlib.md5(f"{source_file}_{text_content}".encode('utf-8')).hexdigest()
            chunk_ids.append(chunk_hash)

        self.vector_store.add_documents(
            documents = docs,
            ids = chunk_ids
        )
        print("Document successfully added to store.")
    
    
    def delete_documents(self, target_file_path: str):
        all_data = self.vector_store.get(where={"source": target_file_path})
        if all_data is None or len(all_data["ids"]) == 0:
            print(f"No document found with source path: {target_file_path}")
            return
        else:
            self.vector_store.delete(ids=all_data["ids"])
            print(f"Deleted document with source path: {target_file_path}")


    
    def ingest_documents(self, doc_path:str = None):
        if not doc_path:
            print("No doc path given.")
            return

        file_signature = self._calculate_file_hash(doc_path)

        exist = self.vector_store.get(where={"file_signature": file_signature})
        if exist and len(exist["ids"]) > 0:
            print((f"File {os.path.basename(doc_path)} with same content has already been ingested."))
            return

        doc_content = self._load_documents(doc_path)
        docs = self._split_documents(doc_content)
        self._add_documents(docs, doc_path)
        print("Ingestion completed successfully.")
        return 


    def retriever(self, query: str, top_k: int = 3):
        results = self.vector_store.similarity_search(
            query,
            k = top_k
        )
        return results["documents"]


my_rag = Rag()

my_rag.ingest_documents(doc_path="./test_document.pdf")

my_rag.retriever("What is the main topic of the document?", 3)[0]