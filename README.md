# QnA Maker

A document processing and question answering application that allows users to upload various document types (PDF, Excel, text) or provide website URLs, and query the content using either direct similarity search or LLM-augmented retrieval. Now with voice capabilities!

## Features

•⁠  ⁠Support for multiple file types:
  - PDF documents
  - Text files
  - Excel spreadsheets (.xlsx, .xls)
  - CSV files
  - Website content

•⁠  ⁠Configurable text processing:
  - Adjustable chunk size and overlap
  - Document splitting for optimal retrieval

•⁠  ⁠Two query methods:
  - Direct similarity search for quick results
  - LLM-augmented RAG for more comprehensive answers

•⁠  ⁠Multiple LLM providers:
  - Groq (llama3, mixtral, gemma models)
  - Azure OpenAI

•⁠  ⁠Voice capabilities:
  - Speech-to-text for voice queries
  - Text-to-speech for listening to responses

•⁠  ⁠User-friendly interface:
  - Simple file upload
  - Website URL input
  - Configurable parameters
  - Expandable results

## Installation

1.⁠ ⁠Clone the repository
2.⁠ ⁠Install dependencies:
   
⁠    pip install -e .
    ⁠
   or
   
⁠    pip install -r requirements.txt
    ⁠

## Usage

Run the application with Streamlit:


streamlit run main.py


## Configuration

•⁠  ⁠*Processing Parameters*: Adjust chunk size and overlap to optimize for your specific documents
•⁠  ⁠*LLM Settings*: Choose between Groq and Azure OpenAI providers
•⁠  ⁠*Voice Features*: Enable speech-to-text for queries and text-to-speech for responses
•⁠  ⁠*Retrieval Settings*: Configure the number of chunks to retrieve for each query

## Requirements

•⁠  ⁠Python 3.8+
•⁠  ⁠Groq API key or Azure OpenAI credentials (for LLM-augmented queries)
•⁠  ⁠Internet connection for embedding model download
•⁠  ⁠Microphone access (for speech-to-text functionality)