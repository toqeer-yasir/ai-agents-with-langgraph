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

        self.embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_store = Chroma(
            collection_name = collection_name,
            embedding_function = self.embedding_fn,
            persist_directory = db_path
        )
        store = self.vector_store.get()
        is_new_store = (len(store["ids"]) == 0)
        
        if is_new_store:
            print(f"Database initialized successfully! Ready for ingestion.")
        else:
            print(f"Successfully connected to database heaving {len(store['ids'])} chunks.")
        
        self.splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50, 
        separators = ["\n\n", "\n", " ", ""],
        )


    async def _load_documents(self, file_path: str):
        if str.lower(file_path).endswith(".txt"):
            doc_content = await TextLoader(file_path).aload()
        
        elif str.lower(file_path).endswith(".pdf"):
            doc_content = await PyPDFLoader(file_path).aload()
            
        elif str.lower(file_path).endswith((".doc", ".docx")):
            doc_content = await Docx2txtLoader(file_path).aload()
        
        elif str.lower(file_path).endswith(".md"):
            doc_content = await UnstructuredMarkdownLoader(file_path).aload()
        
        elif str.lower(file_path).endswith(".csv"):
            doc_content = await CSVLoader(file_path).aload()

        else:
            raise TypeError(f"Uploaded file type '{file_path}' is not supported by RAG!")

        return doc_content
    
    
    def _calculate_file_hash(self, file_path: str):
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()


    def _split_documents(self, doc_content):
        return self.splitter.split_documents(doc_content)
    
    
    def _add_documents(self, docs, file_signature, document_id, user_id):
        chunk_ids = []

        for index, doc in enumerate(docs):
            doc.metadata.update({
            "user_id": user_id,
            "document_id": document_id,
            "file_signature": file_signature
        })
            chunk_id = f"{document_id}_{index}"
            chunk_ids.append(chunk_id)

        self.vector_store.add_documents(
            documents = docs,
            ids = chunk_ids
        )
    
    
    def delete_documents(self, user_id: str, document_id: str):
        
        all_data = self.vector_store.get(
    where={
        "$and": [
            {"document_id": document_id},
            {"user_id": user_id}
        ]})

        if not all_data["ids"]:
            return f"No document found with document_id: {document_id} and user_id: {user_id}"
        else:
            self.vector_store.delete(ids=all_data["ids"])
            return f"Document deleted for user: {user_id}."


    
    async def ingest_documents(self, file_path:str = None, document_id: str = None, user_id: str = None):
        if not file_path or not document_id or not user_id:
            return "Missing required parameters."

        file_signature = self._calculate_file_hash(file_path)

        exist = self.vector_store.get(where={
            "$and": [
                {"file_signature": file_signature},
                {"user_id": user_id}
            ]})

        if exist["ids"]:
            return f"File {os.path.basename(file_path)} with same content has already been ingested."
            
        doc_content = await self._load_documents(file_path)
        docs = self._split_documents(doc_content)
        self._add_documents(docs, file_signature, document_id, user_id)

        return "Ingestion completed successfully."


    def retriever(self, query: str, top_k: int = 3, user_id: str = None):
        results = self.vector_store.similarity_search(
        query,
        k=top_k,
        filter={
            "user_id": user_id
        })
        return results


my_rag = Rag()
