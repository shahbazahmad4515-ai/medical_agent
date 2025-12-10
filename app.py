import streamlit as st
import os
import gdown
import zipfile
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
st.set_page_config(page_title="DiReCT Medical RAG", page_icon="🩺", layout="wide")
DATA_DIR = "./Finished"
DB_DIR = "./chroma_db"
DATA_ZIP_PATH = "./Finished.zip"

# !!! PASTE YOUR GOOGLE DRIVE LINK HERE !!!
GOOGLE_DRIVE_FILE_ID = "1B5aizC-NUsI72i31yjDfvx6LDDBtfCkn" # <-- REPLACE THIS with the ID from your link

# --- CLOUD SETUP: Download Data & Build DB (runs only once) ---
@st.cache_resource
def setup_rag_pipeline():
    """Downloads data, processes it, and builds the ChromaDB vector store."""

    # 1. Download data if not present
    if not os.path.exists(DATA_DIR):
        st.info("Downloading clinical notes dataset...")
        # Construct the gdown URL
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
        gdown.download(url, DATA_ZIP_PATH, quiet=False)
        
        st.info("Unzipping dataset...")
        with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(DATA_ZIP_PATH)

    # 2. Build DB if not present
    if not os.path.exists(DB_DIR):
        st.info("First-time setup: Building Vector Database. This may take a few minutes...")
        
        # --- Data Loading Logic (from your notebook) ---
        documents = []
        for dirpath, _, filenames in os.walk(DATA_DIR):
            json_files = [f for f in filenames if f.endswith(".json")]
            if not json_files: continue
            
            parts = dirpath.split(os.sep)
            subtype = parts[-1]
            disease_group = parts[-2] if len(parts) > 1 else "Unknown"
            
            for file in json_files:
                with open(os.path.join(dirpath, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                full_text = ""
                input_keys = sorted([k for k in data.keys() if k.startswith("input")])
                for key in input_keys:
                    section_text = data[key]
                    if isinstance(section_text, str) and section_text.strip() != "None":
                        full_text += f"{section_text}\n\n"
                
                if len(full_text) > 50:
                    documents.append(Document(page_content=full_text, metadata={"source": file, "disease_group": disease_group, "subtype": subtype}))
        
        # --- Chunking & Embedding ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=DB_DIR)
        st.success("Vector Database built successfully!")
    else:
        # Load existing DB
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
        
    return vectorstore

vectorstore = setup_rag_pipeline()


# --- STREAMLIT UI ---
st.title("🩺 DiReCT: AI Clinical Assistant")

with st.sidebar:
    st.header("Configuration")
    st.info("This RAG system uses Llama 3 and MIMIC-IV data to answer clinical questions.")
    # Use Streamlit Secrets for the API key
    if 'GROQ_API_KEY' not in st.secrets:
        st.error("Groq API Key not found in Streamlit Secrets!")
        st.stop()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about patient symptoms, history, or labs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # --- RAG Chain ---
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
            
            template = """You are an expert medical assistant. Answer the user's question based ONLY on the following context. If the information is not in the context, say 'Information not found in the provided patient records.'

            Context: {context}
            Question: {question}
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())
            
            response = rag_chain.invoke(prompt)
            st.markdown(response)
            
            with st.expander("🔍 Show Retrieved Documents"):
                retrieved_docs = retriever.invoke(prompt)
                for doc in retrieved_docs:
                    st.info(f"Source: {doc.metadata.get('source', 'N/A')}")
                    st.write(doc.page_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:

            st.error(f"An error occurred: {e}")

