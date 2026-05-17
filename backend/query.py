import sys
import os
from dotenv import load_dotenv
import time

def main():
    load_dotenv()
    if not os.getenv("NVIDIA_API_KEY"):
        print("Error: NVIDIA_API_KEY is not set in the environment.")
        sys.exit(1)

    print("Initializing RAG Engine (loading libraries and database)...")
    start_time = time.time()
    
    from pathlib import Path
    import chromadb
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    
    BASE_DIR = Path(__file__).resolve().parent
    CHROMA_PATH = BASE_DIR / "chroma_db"
    
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = chroma_client.get_or_create_collection("hierarchical_knowledge_base")
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        sys.exit(1)
        
    NVIDIA_CHAT_MODEL = "meta/llama-3.1-8b-instruct"
    NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
    
    chat_llm = ChatNVIDIA(model=NVIDIA_CHAT_MODEL)
    embed_model = NVIDIAEmbeddings(model=NVIDIA_EMBED_MODEL)
    
    load_time = time.time() - start_time
    print(f"✅ RAG Engine Ready! (Loaded in {load_time:.2f} seconds)")
    print("Type your question below. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            # Interactive prompt for the user
            query = input("> ")
            if not query.strip():
                continue
            
            if query.lower().strip() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # Retrieval Pipeline
            query_vector = embed_model.embed_query(query)
            results = collection.query(query_embeddings=[query_vector], n_results=3)
            
            contexts = []
            if results and results["metadatas"] and results["metadatas"][0]:
                contexts = list(set([meta["parent_text"] for meta in results["metadatas"][0] if "parent_text" in meta]))
                
            if not contexts:
                print("Answer:\n" + "-"*40)
                print("I couldn't find anything relevant in your knowledge base.")
                print("-" * 40 + "\n")
                continue

            # Generation Pipeline
            context_str = "\n\n---\n\n".join(contexts)
            system_prompt = (
                "You are a helpful assistant. Answer the user's question ONLY using the provided context snippets.\n"
                "Rules:\n1. Format your response in markdown.\n"
                "2. If there is no answer in the context, reply exactly with: 'I couldn't find anything relevant in your knowledge base.'.\n"
                "3. Do not include external facts."
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Context Elements:\n{context}\n\nUser Query: {query}")
            ])
            
            chain = prompt | chat_llm
            
            print("Answer:\n" + "-"*40)
            for chunk in chain.stream({"context": context_str, "query": query}):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print("\n" + "-"*40 + "\n")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError during RAG pipeline: {e}\n")

if __name__ == "__main__":
    main()
