import os
import tempfile
import base64
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader
)
# Excel handling is done with pandas directly
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Custom QnA Maker", layout="wide")
st.title("Custom QnA Maker")

# Initialize session state variables
if "processed" not in st.session_state:
    st.session_state.processed = False
if "db" not in st.session_state:
    st.session_state.db = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # File upload section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv", "xlsx", "xls"])
    
    # Or provide a website URL
    website_url = st.text_input("Or enter a website URL")
    
    # Processing parameters
    st.subheader("Processing Parameters")
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
    
    # LLM provider selection
    st.subheader("LLM Provider")
    llm_provider = st.radio("Select LLM provider:", ["Groq", "Azure OpenAI"])
    
    # Query method selection
    st.subheader("Query Method")
    query_method = st.radio("Select query method:", ["Direct Similarity Search", "LLM-Augmented RAG"])
    
    # LLM configuration (only shown if LLM method is selected)
    if query_method == "LLM-Augmented RAG":
        st.subheader("LLM Configuration")
        
        if llm_provider == "Groq":
            llm_model = st.selectbox(
                "Select Groq model:",
                ["llama3-70b-8192", "llama3-8b-8192", "deepseek-r1-distill-llama-70b", "qwen-qwq-32b", "llama-3.1-8b-instant"]
            )
            groq_api_key = st.text_input("Groq API Key", type="password")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        else:  # Azure OpenAI
            azure_endpoint = st.text_input("Azure OpenAI Endpoint")
            azure_api_key = st.text_input("Azure OpenAI API Key", type="password")
            azure_api_version = st.text_input("Azure API Version", value="2023-05-15")
            azure_deployment = st.text_input("Azure Deployment Name")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        k_value = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=4)
    
    # Voice options
    st.subheader("Voice Options")
    enable_voice = st.checkbox("Enable Voice Features", value=False)

# Function to autoplay audio
def autoplay_audio(audio_data):
    b64 = base64.b64encode(audio_data).decode()
    md = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# Function for text-to-speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# Function for speech-to-text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now.")
        audio = r.listen(source)
        st.write("Processing speech...")
    
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError:
        st.error("Could not request results from Google Speech Recognition service")
        return None

# Function to load and process documents
def process_document(file_path=None, url=None, file_type=None):
    try:
        # Load document based on type
        if url:
            loader = WebBaseLoader(url)
            docs = loader.load()
            st.success(f"Successfully loaded content from {url}")
        elif file_path:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path)
            elif file_type == "csv":
                loader = CSVLoader(file_path)
            elif file_type in ["xlsx", "xls"]:
                # Create a custom loader for Excel files using pandas
                df = pd.read_excel(file_path)
                # Convert DataFrame to text content
                text_content = df.to_string()
                # Create a document
                from langchain_core.documents import Document
                docs = [Document(page_content=text_content, metadata={"source": file_path})]
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                final_docs = text_splitter.split_documents(docs)
                
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                # Create vector store
                db = FAISS.from_documents(final_docs, embeddings)
                
                # Save embeddings and db to session state
                st.session_state.embeddings = embeddings
                st.session_state.db = db
                st.session_state.processed = True
                
                # Display processing information
                st.info(f"Document processed into {len(final_docs)} chunks")
                
                return db, embeddings
            else:
                st.error("Unsupported file type")
                return None
            
            docs = loader.load()
            st.success(f"Successfully loaded {file_type} document")
        else:
            st.error("No document provided")
            return None
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        final_docs = text_splitter.split_documents(docs)
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        db = FAISS.from_documents(final_docs, embeddings)
        
        # Save embeddings and db to session state
        st.session_state.embeddings = embeddings
        st.session_state.db = db
        st.session_state.processed = True
        
        # Display processing information
        st.info(f"Document processed into {len(final_docs)} chunks")
        
        return db, embeddings
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# Process button
col1, col2 = st.columns([1, 1])
with col1:
    process_button = st.button("Process Document")

if process_button:
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Process the document
        file_type = uploaded_file.name.split('.')[-1].lower()
        process_document(file_path=file_path, file_type=file_type)
        
        # Clean up the temporary file
        os.unlink(file_path)
    
    elif website_url:
        process_document(url=website_url)
    
    else:
        st.warning("Please upload a file or provide a website URL")

# Voice input button
if enable_voice and st.session_state.processed:
    if st.button("🎤 Speak Your Question"):
        query = speech_to_text()
        if query:
            st.success(f"You said: {query}")
            # Store the query in a text input for possible editing
            st.session_state.voice_query = query
        else:
            st.warning("No speech detected. Please try again.")

if st.session_state.processed:
    st.header("Query Your Document")
    
    initial_query = st.session_state.get("voice_query", "") if hasattr(st.session_state, "voice_query") else ""
    query = st.text_input("Enter your question:", value=initial_query)
    
    if query:
        if query_method == "Direct Similarity Search":
            # Direct similarity search
            docs = st.session_state.db.similarity_search(query)
            
            st.subheader("Results:")
            result_text = ""
            for i, doc in enumerate(docs):
                with st.expander(f"Result {i+1}"):
                    st.write(doc.page_content)
                result_text += doc.page_content + "\n\n"
            
            # Store the response for TTS
            st.session_state.last_response = "Here are the search results: " + result_text[:500] + "..."
        # LLM-Augmented RAG
        else:  
            if llm_provider == "Groq" and not groq_api_key:
                st.error("Please provide a Groq API key")
            elif llm_provider == "Azure OpenAI" and (not azure_endpoint or not azure_api_key or not azure_deployment):
                st.error("Please provide all Azure OpenAI credentials")
            else:
                with st.spinner("Generating response with LLM..."):
                    try:
                        # Initialize LLM based on provider
                        if llm_provider == "Groq":
                            llm = ChatGroq(
                                model_name=llm_model,
                                api_key=groq_api_key,
                                temperature=temperature
                            )
                        else:  # Azure OpenAI
                            llm = AzureChatOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_key=azure_api_key,
                                api_version=azure_api_version,
                                deployment_name=azure_deployment,
                                temperature=temperature
                            )
                        
                        # Create RAG chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=st.session_state.db.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": k_value}
                            ),
                            return_source_documents=True
                        )
                        
                        # Get response
                        result = qa_chain.invoke({"query": query})
                        
                        # Display response
                        st.subheader("LLM Response:")
                        st.write(result["result"])
                        
                        st.session_state.last_response = result["result"]
                        
                        # Display source documents
                        st.subheader("Source Documents:")
                        for i, doc in enumerate(result["source_documents"]):
                            with st.expander(f"Source {i+1}"):
                                st.write(doc.page_content)
                    
                    except Exception as e:
                        st.error(f"Error generating LLM response: {str(e)}")

        # Text-to-speech for the response only if voice features are enabled
        if enable_voice and st.session_state.last_response:
            if st.button("Listen to Response"):
                audio_bytes = text_to_speech(st.session_state.last_response)
                autoplay_audio(audio_bytes.read())

if __name__ == "__main__":
    pass  