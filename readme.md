# AmbedkarGPT-Intern-Task

A RAG-based Q&A system that answers questions from Dr. B.R. Ambedkar's speech using LangChain, ChromaDB, and Ollama.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that:
1. Loads text from Dr. B.R. Ambedkar's speech
2. Splits the text into manageable chunks
3. Creates embeddings using HuggingFace transformers
4. Stores embeddings in a local ChromaDB vector database
5. Retrieves relevant chunks based on user queries
6. Generates contextual answers using Ollama's Mistral 7B LLM

## Technology Stack

- **Framework**: LangChain
- **Vector Database**: ChromaDB (local, persistent)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Ollama with Mistral 7B
- **Language**: Python 3.8+

## Prerequisites

### Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

### Pull Mistral Model
```bash
ollama pull mistral
```

Verify installation:
```bash
ollama list
```

## Installation

### Clone the Repository
```bash
git clone https://github.com/wannasleepforlong/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Create Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py              # Main application code
├── speech.txt           # Dr. Ambedkar's speech text
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
└── chroma_db/           # Vector database (created on first run)
```

## Usage

### Running the Application

```bash
python main.py
```

### First Run
On the first run, the system will:
- Load `speech.txt`
- Split text into chunks
- Create embeddings 
- Store in `./chroma_db` directory

Subsequent runs load the vector database instantly.

Type `quit`, `exit`, or `q` to exit the application.


## Example Questions

1. What does Ambedkar say about the shastras?
2. What is the real enemy according to the text?
3. Can you have both caste and belief in shastras?

## How It Works

### Document Loading
Loads `speech.txt` using LangChain's `TextLoader` with UTF-8 encoding.

### Text Chunking
Uses `RecursiveCharacterTextSplitter` to create overlapping chunks (200 characters per chunk, 20-character overlap) for better context continuity.

### Embedding Generation
Each chunk is converted to a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`.

### Vector Storage
Embeddings are stored in ChromaDB with persistent local storage in `./chroma_db` for fast similarity search.

### Query Processing
User questions are embedded and compared against stored chunks using semantic similarity search.

### Answer Generation
Retrieved chunks and the question are sent to Mistral 7B LLM with a prompt that instructs it to answer only from the provided context, preventing hallucinations.



