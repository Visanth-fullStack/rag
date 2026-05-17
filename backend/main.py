import os
import asyncio
import numpy as np
import io
import wave
import hashlib
import time
import shutil
import uuid
print("Imports start...")
import json
print("psutil...")
import psutil
print("chromadb...")
import chromadb
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import threading
import queue
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from pypdf import PdfReader
from chromadb.config import Settings
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.retrievers import BM25Retriever
print("langchain text splitters...")
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
print("langchain experimental...")
from langchain_experimental.text_splitter import SemanticChunker
print("Imports done.")

# Load API Keys
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
RIVA_SERVER = os.getenv("RIVA_SERVER", "grpc.nvcf.nvidia.com:443")
RIVA_FUNCTION_ID = os.getenv("RIVA_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081")

app = FastAPI(title="Interview Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Riva ASR Initialization (Requires nvidia-riva-client)
try:
    from riva.client import ASRService, Auth, RecognitionConfig, StreamingRecognitionConfig
    from riva.client.proto.riva_audio_pb2 import AudioEncoding
    
    auth = Auth(
        uri=RIVA_SERVER,
        use_ssl=True,
        metadata_args=[["function-id", RIVA_FUNCTION_ID], ["authorization", f"Bearer {NVIDIA_API_KEY}"]]
    )
    asr_service = ASRService(auth)
    print("NVIDIA Riva ASR Service Initialized")
except ImportError:
    print("WARNING: nvidia-riva-client not installed. Riva ASR will not work.")
    asr_service = None

# Initialize Docling Converter with OCR disabled by default to save memory
def get_converter(do_ocr=False):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    # Using PyPdfiumDocumentBackend for better memory efficiency
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

# Default converter (OCR off)
default_converter = get_converter(do_ocr=False)
# OCR converter (initialized only when needed to save memory)
ocr_converter = None

# In-memory job store
jobs = {}

def get_dynamic_chunk_size(do_ocr: bool):
    """
    Calculate a safe chunk size based on available system memory.
    """
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    # Heuristic (more conservative now):
    # Non-OCR: ~150MB per page for safe processing (layout analysis can be heavy)
    # OCR: ~500MB per page for safe processing
    if do_ocr:
        # Minimum 1 page, Maximum 3 pages for OCR
        size = int(available_mb / 500)
        return max(1, min(size, 3))
    else:
        # Minimum 2 pages, Maximum 10 pages for standard text
        size = int(available_mb / 150)
        return max(2, min(size, 10))

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB (Local Persistence)
CHROMA_PATH = BASE_DIR / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

print("Docling Converter Initialized")
print(f"ChromaDB Initialized at {CHROMA_PATH}")

# Initialize NVIDIA Embeddings and Semantic Splitter
embedding_function = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
semantic_splitter = SemanticChunker(embedding_function)
char_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

# RAG Specific Components
rag_embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
fast_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
reasoning_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

# State Ledger for Delta Analysis (In production, replace with SQLite)
state_ledger: Dict[str, Dict[str, Any]] = {}

def calculate_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_markdown_to_hierarchical_payload(filepath: str, doc_id: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_markdown = f.read()

    # Stage 1.1: Header Splitter (Parent Chunks)
    headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_docs = markdown_splitter.split_text(raw_markdown)
    
    # Stage 1.1b: Enforce maximum size before Semantic Chunking
    parent_docs = char_splitter.split_documents(header_docs)
    print(f"[{doc_id}] Splitting markdown into {len(parent_docs)} parent sections based on headers and length limits.")

    payload_batch = []
    current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for p_idx, parent_doc in enumerate(parent_docs):
        parent_text = parent_doc.page_content
        parent_id = f"{doc_id}_P{p_idx}"

        # Stage 1.2: Semantic Splitter over individual parents (Child Chunks)
        child_docs = semantic_splitter.split_documents([parent_doc])
        
        if len(parent_docs) > 0 and (p_idx + 1) % max(1, len(parent_docs) // 5) == 0:
            print(f"[{doc_id}] Processing semantic chunks: {p_idx + 1}/{len(parent_docs)} sections done ({(p_idx + 1)/len(parent_docs)*100:.1f}%)")

        for c_idx, child_doc in enumerate(child_docs):
            child_text = child_doc.page_content
            child_id = f"{doc_id}_P{p_idx}_C{c_idx}"

            # Stage 2: Cross-Reference Metadata Mapping
            metadata = {
                "doc_id": doc_id,
                "parent_id": parent_id,
                "parent_text": parent_text,  # Emplaced broad context
                "timestamp": current_time,
                "chunk_index": c_idx
            }

            payload_batch.append({
                "id": child_id,
                "text": child_text,
                "metadata": metadata
            })
            
    print(f"[{doc_id}] Payload generation complete. Total chunks generated: {len(payload_batch)}")
    return payload_batch

# ---------------------------------------------------------
# RAG GRAPH DEFINITIONS
# ---------------------------------------------------------
class RetrievalState(TypedDict):
    query: str
    route: str                 # "fast" or "advanced"
    hyde_document: str         # Generated placeholder text
    retrieved_chunks: List[str] # Final contexts
    generation: str            # Raw output
    hallucination_checks: int  # Circuit breaker
    validated: bool            # Self-RAG flag

class QueryRouterEvaluation(BaseModel):
    path: Literal["fast", "advanced"] = Field(
        description="Choose 'fast' for definitions/facts. 'advanced' for complex code/architecture."
    )

class HallucinationEvaluator(BaseModel):
    is_hallucinating: bool = Field(description="True if generation references outside facts.")
    rationalization: str = Field(description="Reason for deviations.")

def route_query_node(state: RetrievalState):
    print(f"Routing query: {state['query']}")
    structured_router = fast_llm.with_structured_output(QueryRouterEvaluation)
    system_prompt = (
        "Analyze the user query. Classify it as:\n"
        "1. 'fast': Basic definition or direct fact.\n"
        "2. 'advanced': Code generation, logic reasoning, or deep debugging."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")])
    decision = (prompt | structured_router).invoke({"query": state["query"]})
    print(f"Decision: {decision.path}")
    return {"route": decision.path, "hallucination_checks": 0}

def fast_retrieval_node(state: RetrievalState):
    print("Executing fast retrieval...")
    query_vector = rag_embeddings.embed_query(state["query"])
    collection = chroma_client.get_or_create_collection("hierarchical_knowledge_base")
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    
    if not results or not results["metadatas"] or not results["metadatas"][0]:
        print("No results found in fast retrieval.")
        return {"retrieved_chunks": []}
        
    parent_contexts = list(set([meta["parent_text"] for meta in results["metadatas"][0] if "parent_text" in meta]))
    print(f"Retrieved {len(parent_contexts)} contexts.")
    return {"retrieved_chunks": parent_contexts}

def generate_hyde_node(state: RetrievalState):
    hyde_prompt = ChatPromptTemplate.from_template(
        "Write a technical paragraph that answers this question perfectly. No preamble.\nQuestion: {query}"
    )
    chain = hyde_prompt | fast_llm
    hypothetical_doc = chain.invoke({"query": state["query"]})
    return {"hyde_document": hypothetical_doc.content}

def hybrid_rerank_retrieval_node(state: RetrievalState):
    search_target = state["hyde_document"]
    query_vector = rag_embeddings.embed_query(search_target)
    collection = chroma_client.get_collection("hierarchical_knowledge_base")
    dense_results = collection.query(query_embeddings=[query_vector], n_results=15)
    
    chunks_pool = dense_results["documents"][0]
    metadata_pool = dense_results["metadatas"][0]
    
    # Simple BM25 on the retrieved candidates
    if not chunks_pool:
        return {"retrieved_chunks": []}
        
    bm25 = BM25Retriever.from_texts(chunks_pool)
    # Rerank logic simulation
    combined_parents = [meta["parent_text"] for meta in metadata_pool]
    deduplicated_contexts = list(set(combined_parents))[:5]
    return {"retrieved_chunks": deduplicated_contexts}

def structured_generation_node(state: RetrievalState):
    print(f"Generating response using {len(state['retrieved_chunks'])} chunks...")
    context_str = "\n\n---\n\n".join(state["retrieved_chunks"])
    system_prompt = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided context snippets.\n"
        "Rules:\n1. Format your response in markdown.\n"
        "2. If there is no answer in the context, reply exactly with: 'no answer'.\n"
        "3. Do not include external facts."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context Elements:\n{context}\n\nUser Query: {query}")
    ])
    chain = prompt | reasoning_llm
    response = chain.invoke({"context": context_str, "query": state["query"]})
    print(f"Generation complete. Length: {len(response.content)}")
    return {"generation": response.content}

def self_rag_guardrail_node(state: RetrievalState):
    print(f"Running hallucination check (Attempt {state['hallucination_checks'] + 1})...")
    if state["hallucination_checks"] >= 2:
        print("Max hallucination checks reached. Approving.")
        return {"validated": True}
    
    structured_checker = fast_llm.with_structured_output(HallucinationEvaluator)
    system_prompt = "Auditor: Identify if the response introduces external knowledge not in the context."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Raw Reference Data:\n{context}\n\nGenerated Response:\n{generation}")
    ])
    context_str = "\n\n---\n\n".join(state["retrieved_chunks"])
    
    try:
        audit_result = (prompt | structured_checker).invoke({"context": context_str, "generation": state["generation"]})
        print(f"Hallucination check result: {audit_result.is_hallucinating}")
        return {"validated": not audit_result.is_hallucinating, "hallucination_checks": state["hallucination_checks"] + 1}
    except Exception as e:
        print(f"Guardrail error: {e}. Defaulting to validated.")
        return {"validated": True, "hallucination_checks": state["hallucination_checks"] + 1}

# Assemble LangGraph
workflow = StateGraph(RetrievalState)
workflow.add_node("router", route_query_node)
workflow.add_node("fast_retrieval", fast_retrieval_node)
workflow.add_node("generate_hyde", generate_hyde_node)
workflow.add_node("hybrid_rerank_retrieval", hybrid_rerank_retrieval_node)
workflow.add_node("generation_layer", structured_generation_node)
workflow.add_node("self_rag_guardrail", self_rag_guardrail_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["route"], {"fast": "fast_retrieval", "advanced": "generate_hyde"})
workflow.add_edge("fast_retrieval", "generation_layer")
workflow.add_edge("generate_hyde", "hybrid_rerank_retrieval")
workflow.add_edge("hybrid_rerank_retrieval", "generation_layer")
workflow.add_edge("generation_layer", "self_rag_guardrail")
workflow.add_conditional_edges(
    "self_rag_guardrail", 
    lambda x: "approved" if x["validated"] else "rejected",
    {"approved": END, "rejected": "generate_hyde"}
)
rag_app = workflow.compile()

@app.get("/")
async def root():
    return {"message": "Interview Agent API is running"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll the status and progress of a conversion job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/health/chroma")
async def health_chroma():
    """
    Check the health of the ChromaDB connection.
    """
    try:
        # chroma_client.heartbeat() returns a nanosecond timestamp if healthy
        heartbeat = chroma_client.heartbeat()
        return {
            "status": "healthy",
            "db_path": str(CHROMA_PATH),
            "heartbeat": heartbeat
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """
    Trigger the orchestrated ingestion workflow for all markdown files in the outputs directory.
    """
    background_tasks.add_task(run_orchestrated_ingestion)
    return {"status": "accepted", "message": "Ingestion workflow started in the background."}

def run_orchestrated_ingestion():
    if not OUTPUT_DIR.exists():
        print(f"Directory {OUTPUT_DIR} not found.")
        return

    detected_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".md")]
    active_doc_ids = []
    total_files = len(detected_files)
    
    print(f"\n========================================")
    print(f"Starting Ingestion Workflow")
    print(f"Detected {total_files} markdown files in {OUTPUT_DIR}")
    print(f"========================================")
    
    # Ensure collection exists with correct metric
    collection = chroma_client.get_or_create_collection(
        name="hierarchical_knowledge_base", 
        metadata={"hnsw:space": "cosine"}
    )

    for f_idx, filename in enumerate(detected_files):
        filepath = OUTPUT_DIR / filename
        doc_id = filename
        active_doc_ids.append(doc_id)

        print(f"\n[File {f_idx + 1}/{total_files}] Processing {doc_id} ...")

        file_hash = calculate_md5(str(filepath))
        ledger_record = state_ledger.get(doc_id)

        # Delta Analysis Rule
        if ledger_record is None or ledger_record["hash"] != file_hash:
            if ledger_record is None:
                print(f"[{doc_id}] Status: New file detected.")
            else:
                print(f"[{doc_id}] Status: Content changes detected.")

            # Stage 3.2: Atomic Invalidation
            if ledger_record is not None:
                print(f"[{doc_id}] Invalidating stale tracking context from ChromaDB.")
                collection.delete(where={"doc_id": doc_id})

            # Execute Stage 1 and Stage 2 Tasks
            print(f"[{doc_id}] Initiating markdown parsing and hierarchical payload generation...")
            payloads = process_markdown_to_hierarchical_payload(str(filepath), doc_id)

            if not payloads:
                print(f"[{doc_id}] Warning: No payloads generated. Skipping.")
                continue

            # Stage 3.3: Vector Populations
            texts = [item["text"] for item in payloads]
            metadatas = [item["metadata"] for item in payloads]
            ids = [item["id"] for item in payloads]
            
            # Generate embeddings with progress
            print(f"[{doc_id}] Generating embeddings for {len(texts)} semantic nodes...")
            embeddings = []
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = rag_embeddings.embed_documents(batch_texts)
                embeddings.extend(batch_embeddings)
                progress = min(100, int((i + len(batch_texts)) / len(texts) * 100))
                print(f"[{doc_id}] Embedding progress: {progress}% ({i + len(batch_texts)}/{len(texts)})")

            print(f"[{doc_id}] Embeddings complete. Ingesting into ChromaDB...")
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            # Commit current state info back to the Ledger
            state_ledger[doc_id] = {
                "hash": file_hash,
                "last_sync": os.path.getmtime(str(filepath))
            }
            print(f"[{doc_id}] SUCCESS: Ingested {len(payloads)} semantic nodes.")
        else:
            print(f"[{doc_id}] Status: No changes. Skipping ingestion.")

    # Garbage Collection Rule
    for ledger_doc_id in list(state_ledger.keys()):
        if ledger_doc_id not in active_doc_ids:
            print(f"\n[Garbage Collection] File missing: {ledger_doc_id}. Dropping indices from ChromaDB.")
            collection.delete(where={"doc_id": ledger_doc_id})
            del state_ledger[ledger_doc_id]
            
    print(f"\n========================================")
    print(f"Ingestion Workflow Completed Successfully")
    print(f"========================================")

async def process_conversion_task(
    job_id: str,
    temp_path: Path,
    file_name: str,
    output_format: str,
    do_ocr: bool
):
    """
    Background task to process conversion with chunks, retries, and progress tracking.
    """
    try:
        jobs[job_id]["status"] = "processing"
        
        # Use appropriate converter
        global ocr_converter
        if do_ocr:
            if ocr_converter is None:
                ocr_converter = get_converter(do_ocr=True)
            conv = ocr_converter
        else:
            conv = default_converter

        # Get total page count
        total_pages = 1
        if file_name.lower().endswith(".pdf"):
            try:
                reader = PdfReader(temp_path)
                total_pages = len(reader.pages)
            except Exception as e:
                print(f"Error reading PDF page count: {e}")

        # Calculate dynamic chunk size
        current_chunk_size = get_dynamic_chunk_size(do_ocr)
        
        all_results = []
        output_filename = f"{job_id}_{Path(file_name).stem}"
        
        for start_page in range(1, total_pages + 1, current_chunk_size):
            end_page = min(start_page + current_chunk_size - 1, total_pages)
            
            # Resilience: Retry logic for each chunk
            max_retries = 2
            chunk_success = False
            
            for attempt in range(max_retries):
                try:
                    result = conv.convert(str(temp_path), page_range=(start_page, end_page))
                    all_results.append(result)
                    chunk_success = True
                    break
                except Exception as e:
                    print(f"Chunk error (pages {start_page}-{end_page}), attempt {attempt+1}: {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to process chunk {start_page}-{end_page} after {max_retries} attempts.")
            
            # Update progress
            progress = int((end_page / total_pages) * 100)
            jobs[job_id]["progress"] = progress

        # Merge and Save
        if output_format == "markdown":
            md_contents = [res.document.export_to_markdown() for res in all_results]
            full_md = "\n\n---\n\n".join(md_contents)
            output_file = OUTPUT_DIR / f"{output_filename}.md"
            with output_file.open("w", encoding="utf-8") as f:
                f.write(full_md)
            result_data = {"output_file": str(output_file.name), "pages": total_pages}
        else:
            json_results = [res.document.export_to_dict() for res in all_results]
            output_file = OUTPUT_DIR / f"{output_filename}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2)
            result_data = {"output_file": str(output_file.name), "pages": total_pages}

        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": result_data
        })

    except Exception as e:
        print(f"Background task failed for job {job_id}: {e}")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })
    finally:
        if temp_path.exists():
            temp_path.unlink()

@app.post("/convert")
async def convert_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    output_format: str = "markdown",
    do_ocr: bool = Query(False, description="Enable OCR for PDF files (can be memory intensive)")
):
    """
    Submit a document for conversion. Returns a job ID to track progress.
    """
    if output_format not in ["markdown", "json"]:
        raise HTTPException(status_code=400, detail="Invalid output format. Use 'markdown' or 'json'.")

    job_id = str(uuid.uuid4())
    temp_path = OUTPUT_DIR / f"{job_id}_{file.filename}"
    
    # Initialize job status
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": file.filename,
        "format": output_format,
        "result": None,
        "error": None
    }

    try:
        # Save uploaded file temporarily
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start background task
        background_tasks.add_task(
            process_conversion_task, 
            job_id, 
            temp_path, 
            file.filename, 
            output_format, 
            do_ocr
        )

        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Conversion started in the background. Use /jobs/{job_id} to track progress."
        }

    except Exception as e:
        # Cleanup on immediate failure
        if temp_path.exists():
            temp_path.unlink()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

async def transcribe_with_groq(audio_bytes):
    # Convert raw PCM bytes to a virtual WAV file in memory
    # Groq API requires a standard audio format (WAV, MP3, etc.)
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(16000) # 16kHz
            wav_file.writeframes(audio_bytes)
        
        wav_io.seek(0)
        # Send to Groq
        translation = client.audio.transcriptions.create(
            file=("chunk.wav", wav_io.read()),
            model="whisper-large-v3",
            response_format="json",
        )
        return translation.text

class ChatRequest(BaseModel):
    query: str

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    NVIDIA_CHAT_MODEL = "meta/llama-3.1-8b-instruct"
    NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
    
    chat_llm = ChatNVIDIA(model=NVIDIA_CHAT_MODEL)
    embed_model = NVIDIAEmbeddings(model=NVIDIA_EMBED_MODEL)
    
    async def event_generator():
        print(f"Running simple RAG for: {query}")
        
        def retrieve_chroma(q_vector):
            collection = chroma_client.get_or_create_collection("hierarchical_knowledge_base")
            results = collection.query(query_embeddings=[q_vector], n_results=3)
            
            contexts = []
            if results and results["metadatas"] and results["metadatas"][0]:
                contexts = list(set([meta["parent_text"] for meta in results["metadatas"][0] if "parent_text" in meta]))
            return contexts
        
        try:
            query_vector = await embed_model.aembed_query(query)
            contexts = await asyncio.to_thread(retrieve_chroma, query_vector)
        except Exception as e:
            print(f"Retrieval error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
            
        if not contexts:
            response_data = {"generation": "I couldn't find anything relevant in your knowledge base.", "query": query}
            yield f"data: {json.dumps(response_data)}\n\n"
            return
            
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
        
        try:
            async for chunk in chain.astream({"context": context_str, "query": query}):
                if chunk.content:
                    yield f"data: {json.dumps({'generation': chunk.content, 'query': query})}\n\n"
        except Exception as e:
            print(f"Generation error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.websocket("/ws/transcribe/{source}")
async def transcribe(websocket: WebSocket, source: str):
    if not NVIDIA_API_KEY or asr_service is None:
        print("ERROR: NVIDIA_API_KEY is not set or nvidia-riva-client is not installed!")
        await websocket.close(code=4001)
        return

    await websocket.accept()
    print(f"Client connected to Riva backend for source: {source}")
    
    audio_queue = queue.Queue()
    result_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    riva_config = StreamingRecognitionConfig(
        config=RecognitionConfig(
            encoding=AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code="en-US",
            max_alternatives=1,
            audio_channel_count=1,
            enable_automatic_punctuation=True,
        ),
        interim_results=True
    )
    
    def riva_worker():
        try:
            def audio_generator():
                while True:
                    chunk = audio_queue.get()
                    if chunk is None: break
                    yield chunk
            
            responses = asr_service.streaming_response_generator(audio_generator(), riva_config)
            for response in responses:
                if response.results:
                    result = response.results[0]
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final
                    
                    if transcript.strip():
                        loop.call_soon_threadsafe(
                            result_queue.put_nowait, 
                            {
                                "text": transcript,
                                "source": source,
                                "language": "en",
                                "is_final": is_final
                            }
                        )
        except Exception as e:
            print(f"[Riva Worker Error] {e}")
            
    riva_thread = threading.Thread(target=riva_worker, daemon=True)
    riva_thread.start()
    
    async def result_sender():
        while True:
            res = await result_queue.get()
            if res is None: break
            try:
                await websocket.send_json(res)
            except: break

    sender_task = asyncio.create_task(result_sender())
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_queue.put(data)
                
    except WebSocketDisconnect:
        print(f"Client disconnected ({source})")
    except Exception as e:
        print(f"Connection error ({source}): {e}")
        try:
            await websocket.close()
        except: pass
    finally:
        audio_queue.put(None)
        sender_task.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
