import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

 
VECTORDB_DIR = "./chroma_db"
SPEECH_FILE = "speech.txt"

def create_vectorstore():
    # embeddings = OllamaEmbeddings(model="mistral")
    # Using HuggingFace embeddings 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(VECTORDB_DIR) and os.listdir(VECTORDB_DIR):
        vectorstore = Chroma(
            persist_directory=VECTORDB_DIR,
            embedding_function=embeddings
        )
        print("Vector store loaded.")
    else:
        print("No existing vector store found. Creating...")

        loader = TextLoader(SPEECH_FILE)
        documents = loader.load()
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, # 500 is too big and only creates 2 chunks for small .txts like this
            chunk_overlap=20
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} text chunks.")
        

        print("Creating embeddings and saving...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTORDB_DIR
        )
        print(f"Vector store created in {VECTORDB_DIR}.")
    
    return vectorstore

def query_rag(vectorstore, query):
    print(f"{'='*60}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOllama(model="mistral")
    
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} relevant chunks")
    print("Generating answer with LLM...")
    context = "\n\n".join([doc.page_content for doc in docs])
    # We use a prompt that strictly limits the LLM to the provided context. We can also give it more freedom if needed using prompt engineering
    prompt = f"""You are a question-answering assistant. 
Answer the question using ONLY the information provided in the context below. 
Do NOT add any information that is not present in the context. 
If the answer is not in the context, respond with: "The context does not provide this information."

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    print(f"\n{'─'*60}")
    print("Answer:")
    print(f"{'─'*60}")
    print(response.content)
    print(f"{'─'*60}\n")
    
    return response.content

if __name__ == "__main__":
    vectorstore = create_vectorstore()    
    print("\n") 
    print("="*60)
    print("Welcome to AmbedkarGPT")
    print("\n")
    print("Type your questions or 'quit' to exit\n")
    print("="*60)
    
    while True:
        query = input("Question: ").strip() 
        if not query:
            continue     
        if query.lower() in ['quit', 'exit', 'q']:
            break
        try:
            query_rag(vectorstore, query)
        except Exception as e:
            print(f"Error: {e}\n")