import streamlit as st
import pickle
import os
import time
import google.generativeai as genai
import requests  # Ensure this import is included
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Set your Gemini API key
GEMINI_API_KEY = "AIzaSyB413aNdnGBpgg3dsQBKGh1w5kjP039SUU"  # Replace with your actual Gemini API key

# Configure the Gemini API using the google.generativeai library
genai.configure(api_key=GEMINI_API_KEY)

# Custom Gemini Embeddings (if Gemini provides embeddings, else, we may need to rely on another embedding provider)
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Gemini embeddings."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            "https://gemini-api-url.com/embed",  # Replace with the actual embedding endpoint URL if provided
            headers=headers,
            json={"texts": texts}
        )
        if response.status_code == 200:
            return response.json().get("embeddings", [])
        else:
            raise ValueError(f"Embedding error: {response.status_code}, {response.text}")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]

# Custom Gemini LLM using google.generativeai
class GeminiLLM(BaseModel):
    api_key: str = Field(..., description="Gemini API key")
    temperature: float = Field(0.9, description="Sampling temperature for generation")
    max_tokens: int = Field(500, description="Maximum tokens to generate")

    def generate(self, prompt: str) -> str:
        """Generate content using the Gemini model."""
        # Initialize the Gemini model using google.generativeai
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

# Streamlit app
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs from sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()

# Initialize LLM and embeddings using Gemini
llm = GeminiLLM(api_key=GEMINI_API_KEY, temperature=0.9, max_tokens=500)
embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY)

# Process URLs when button is clicked
if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings and save them to FAISS index
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_gemini, f)

# Ask for a query and process the response
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display the answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
