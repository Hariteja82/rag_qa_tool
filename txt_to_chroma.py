import os
import hashlib
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def hash_text(text: str) -> str:
    """Generate a unique hash for a given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_txts_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            txt_path = os.path.join(folder_path, filename)
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
                doc_hash = hash_text(text)
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "doc_hash": doc_hash}
                    )
                )
    return documents

def store_in_chroma(docs, persist_directory):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY,
        model="models/embedding-001"
    )

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    existing = vectordb.get(include=["metadatas"])
    existing_hashes = set()
    if existing and "metadatas" in existing:
        for metadata in existing["metadatas"]:
            if isinstance(metadata, dict) and "doc_hash" in metadata:
                existing_hashes.add(metadata["doc_hash"])

    new_docs = [doc for doc in docs if doc.metadata["doc_hash"] not in existing_hashes]

    if not new_docs:
        print("✅ No new text files found. ChromaDB is up to date.")
        return

    split_docs = splitter.split_documents(new_docs)
    vectordb.add_documents(split_docs)
    vectordb.persist()
    print(f"✅ Added {len(new_docs)} new text document(s) to ChromaDB.")

if __name__ == "__main__":
    TXT_FOLDER = "txts"
    PERSIST_DIR = "chroma_db"

    docs = load_txts_from_folder(TXT_FOLDER)
    store_in_chroma(docs, PERSIST_DIR)
