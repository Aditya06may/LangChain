import os
import sys
import warnings

# Suppress deprecation warnings from langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fix for module shadowing: The current directory is named "LangChain", which conflicts
# with the 'langchain' package on Windows (case-insensitive file system). 
# We remove it from sys.path so Python loads the actual pip package.
script_dir = os.path.dirname(os.path.abspath(__file__)).lower()
sys.path = [p for p in sys.path if p != '' and p.lower() != script_dir]

import glob
import pickle
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL_NAME = "openrouter/free"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_DIR = "faiss_index"
FAISS_PKL_PATH = "faiss_store.pkl"
SOURCE_DIR = "extracted_data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_file(file_path: str):
    ext = Path(file_path).suffix.lower()
    docs = []
    try:
        if ext == ".pdf":
            docs = PyPDFLoader(file_path).load()
        elif ext == ".csv":
            docs = CSVLoader(file_path, encoding="utf-8").load()
        elif ext in (".xlsx", ".xls"):
            docs = UnstructuredExcelLoader(file_path, mode="elements").load()
        elif ext == ".txt":
            docs = TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".docx":
            docs = Docx2txtLoader(file_path).load()
        else:
            print(f"Skipping unsupported file type: {file_path}")
            return []
            
        for d in docs:
            d.metadata["source"] = os.path.basename(file_path)
            d.metadata["file_type"] = ext.lstrip(".")
        print(f"Loaded {len(docs):3d} doc(s) from {os.path.basename(file_path)}")
        return docs
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return []

def build_vector_store(embeddings):
    print("Building vector database from source documents...")
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        print("Please extract your files into 'extracted_data' before running.")
        sys.exit(1)
        
    all_documents = []
    for file_path in sorted(glob.glob(os.path.join(SOURCE_DIR, "*"))):
        all_documents.extend(load_file(file_path))
        
    if not all_documents:
        print("Error: No documents found to build the database.")
        sys.exit(1)
        
    print(f"\nTotal raw documents loaded: {len(all_documents)}")
    print("Chunking documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_documents)
    
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        
    print(f"Split {len(all_documents)} documents into {len(chunks)} chunks")
    print("Generating embeddings and building FAISS index...")
    
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    print(f"Saving vector database to '{FAISS_INDEX_DIR}'...")
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_DIR)
    print("Build complete!")
    return vector_store

def get_vector_store(embeddings):
    """Load the vector store from local index, or build it if missing."""
    if os.path.exists(FAISS_INDEX_DIR):
        print(f"Loading FAISS index from {FAISS_INDEX_DIR}...")
        return FAISS.load_local(
            FAISS_INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    elif os.path.exists(FAISS_PKL_PATH):
        print(f"Loading FAISS index from {FAISS_PKL_PATH}...")
        with open(FAISS_PKL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        print("Vector database not found. Initiating build process...")
        return build_vector_store(embeddings)

def main():
    # 1. Initialize the embedding model
    print("Initializing embedding model...")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    # 2. Load or build the vector store
    vector_store = get_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 3. Initialize the LLM (OpenRouter)
    print("Initializing LLM...")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=MODEL_NAME
    )
    
    # 4. Set up the RAG pipeline
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("==================================================")
    print("Agent is ready! Type 'exit' or 'quit' to stop.")
    print("==================================================")
    
    # 5. Interactive loop for user input
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            print("Agent is thinking...")
            response = rag_chain.invoke({"input": user_input})
            
            print(f"\nAgent: {response['answer']}")
            print("\nSources:")
            for i, doc in enumerate(response.get('context', [])):
                source = doc.metadata.get('source', 'Unknown')
                print(f"  [{i+1}] {source}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
