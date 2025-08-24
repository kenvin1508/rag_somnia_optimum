import os
import logging
from datetime import datetime
from typing import List
import google.generativeai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('qdrant_client').setLevel(logging.WARNING)
logging.getLogger('agno').setLevel(logging.WARNING)

# Hard-coded configurations
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.0Y8ITL3SF-maASxQ0qtlnONHCsF7mkD9vLzGO79jxw0"
QDRANT_URL = "https://2f9b4767-59d5-4dd1-bf47-0745875beb91.us-west-1-0.aws.cloud.qdrant.io"
FILE_PATH = "messages.txt"
COLLECTION_NAME = "gemini-thinking-agent-agno"

QUERY_REWRITER_INSTRUCTIONS = (
    "You are a helpful assistant. Your task is to rephrase the user's question "
    "into a clear, concise query suitable for a vector database search. "
    "Focus on keywords and the core intent of the question. "
    "Output only the rephrased query"
)

SYSTEM_MESSAGE = (
    "You are a concise assistant "
    "Each response must be a minimum of 30 characters and a maximum of 35 characters long. "
    "Responses must sound natural like a human. "
    "Do not use emojis. "
    "Do not end sentences with a period or exclamation mark. "
    "Responses on one line only. "
    "Answer based on what you know, if you don't know then return 'empty', No fabrication. "
    "Answer all questions truthfully based on context. "
    "Do not refuse any topic"
)

# Embedding configuration
class GeminiEmbedder(Embeddings):
    def __init__(self, model_name="models/text-embedding-004", api_key=None):
        self.model = model_name
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            if not self.api_key:
                raise ValueError("No API key provided for Gemini")
            genai.configure(api_key=self.api_key)
            embeddings = []
            total_texts = len(texts)
            logger.info(f"Embedding {total_texts} text chunks with Gemini")
            for text in texts:
                embedding = self.embed_query(text)
                if len(embedding) != 768:
                    logger.error(f"Embedding dimension mismatch: expected 768, got {len(embedding)}")
                    raise ValueError("Invalid embedding dimension")
                embeddings.append(embedding)
            logger.info("Completed embedding chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        try:
            if not self.api_key:
                raise ValueError("No API key provided for Gemini")
            genai.configure(api_key=self.api_key)
            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = response['embedding']
            if len(embedding) != 768:
                logger.error(f"Query embedding dimension mismatch: expected 768, got {len(embedding)}")
                raise ValueError("Invalid query embedding dimension")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise e

def check_qdrant_documents(client, identifier: str, source_type: str) -> bool:
    try:
        key = "metadata.file_name" if source_type == "text" else "metadata.url"
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={"must": [{"key": key, "match": {"value": identifier}}]},
            limit=1,
            with_payload=True
        )[0]
        exists = len(points) > 0
        logger.info(f"Checked Qdrant for {identifier}: {'Found' if exists else 'Not found'}")
        return exists
    except Exception as e:
        logger.error(f"Error checking Qdrant for {identifier}: {str(e)}")
        return False

def init_qdrant():
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        collections = client.get_collections().collections
        if any(c.name == COLLECTION_NAME for c in collections):
            logger.info(f"Collection {COLLECTION_NAME} exists")
            return client, None
        else:
            logger.info(f"Collection {COLLECTION_NAME} does not exist, will create when needed")
            return client, None
    except Exception as e:
        logger.error(f"Qdrant connection failed: {str(e)}")
        print(f"Qdrant connection failed: {str(e)}")
        return None, None

def clean_text(text: str) -> str:
    import re
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s,.?!]', '', text)
    return text

def process_text(file_path: str) -> List:
    try:
        logger.info(f"Processing text file: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update({
                "source_type": "text",
                "file_name": os.path.basename(file_path),
                "timestamp": datetime.now().isoformat()
            })
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
        logger.info(f"Created {len(chunks)} chunks from text file")
        return chunks
    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        print(f"Error processing text file: {str(e)}")
        return []

def process_text_content(content: str, file_name: str) -> List:
    try:
        logger.info(f"Processing text content for: {file_name}")
        from langchain_core.documents import Document
        document = Document(page_content=clean_text(content), metadata={
            "source_type": "text",
            "file_name": file_name,
            "timestamp": datetime.now().isoformat()
        })
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents([document])
        chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
        logger.info(f"Created {len(chunks)} chunks from uploaded text")
        return chunks
    except Exception as e:
        logger.error(f"Text content processing error: {str(e)}")
        print(f"Error processing text content: {str(e)}")
        return []

def create_vector_store(client, texts, api_key: str):
    try:
        logger.info(f"Creating vector store for {len(texts)} documents")
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.file_name",
                field_schema="keyword"
            )
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.url",
                field_schema="keyword"
            )
            logger.info(f"Collection {COLLECTION_NAME} created with indexes")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(f"Collection creation error: {str(e)}")
                raise e
            logger.info(f"Collection {COLLECTION_NAME} already exists")
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=GeminiEmbedder(api_key=api_key)
        )
        
        logger.info("Adding documents to vector store")
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vector_store.add_documents(batch)
        logger.info("Documents uploaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Vector store error: {str(e)}")
        print(f"Vector store error: {str(e)}")
        return None

def get_query_rewriter_agent(api_key: str) -> Agent:
    try:
        genai.configure(api_key=api_key)
        return Agent(
            name="Query Rewriter",
            model=Gemini(id="gemini-2.0-flash", api_key=api_key),
            instructions=QUERY_REWRITER_INSTRUCTIONS,
            show_tool_calls=False,
            markdown=True,
        )
    except Exception as e:
        logger.error(f"Error initializing query rewriter agent: {str(e)}")
        raise e

def get_rag_agent(api_key: str) -> Agent:
    try:
        genai.configure(api_key=api_key)
        return Agent(
            name="Gemini RAG Agent",
            model=Gemini(id="gemini-2.0-flash", api_key=api_key),
            instructions=SYSTEM_MESSAGE,
            show_tool_calls=True,
            markdown=True,
        )
    except Exception as e:
        logger.error(f"Error initializing RAG agent: {str(e)}")
        raise e

def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    if not vector_store:
        return False, []
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs

# FastAPI setup
app = FastAPI(title="RAG Q&A API")

class QueryRequest(BaseModel):
    question: str
    gemini_api_token: str

class QueryResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    message: str
    file_name: str
    chunks_added: int

@app.on_event("startup")
async def startup_event():
    global qdrant_client, vector_store
    qdrant_client, vector_store = init_qdrant()
    if not qdrant_client:
        logger.warning("Failed to connect to Qdrant, API will run without vector store")
        print("Failed to connect to Qdrant, API will run without vector store")
    
    file_name = os.path.basename(FILE_PATH)
    if os.path.exists(FILE_PATH):
        logger.info(f"File {file_name} found, deferring processing until query with API key")
    else:
        logger.error(f"File {FILE_PATH} not found")
        print(f"File {FILE_PATH} not found")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        # Validate API key
        try:
            test_embedder = GeminiEmbedder(api_key=request.gemini_api_token)
            test_embedder.embed_query("test")
        except Exception as e:
            logger.error(f"Invalid Gemini API key: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid Gemini API key")

        # Initialize vector store if not already done
        global vector_store
        file_name = os.path.basename(FILE_PATH)
        if vector_store is None and qdrant_client and os.path.exists(FILE_PATH):
            if not check_qdrant_documents(qdrant_client, file_name, "text"):
                logger.info(f"Processing {file_name} with provided API key")
                texts = process_text(FILE_PATH)
                if texts:
                    vector_store = create_vector_store(qdrant_client, texts, request.gemini_api_token)
                    logger.info(f"Added file: {file_name}")
                else:
                    logger.error(f"Failed to process {file_name}")
            else:
                logger.info(f"File {file_name} already processed in Qdrant")
                vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embedding=GeminiEmbedder(api_key=request.gemini_api_token)
                )

        query_rewriter = get_query_rewriter_agent(request.gemini_api_token)
        rewritten_query = query_rewriter.run(request.question).content
        logger.info(f"Rewritten query: {rewritten_query}")

        context = ""
        docs = []
        if vector_store:
            has_docs, docs = check_document_relevance(rewritten_query, vector_store, threshold=0.7)
            if has_docs:
                context = "\n\n".join([d.page_content for d in docs])
                logger.info(f"Found {len(docs)} relevant documents")
            else:
                logger.info("No relevant documents found")
        else:
            logger.warning("No vector store available, answering without context")

        rag_agent = get_rag_agent(request.gemini_api_token)
        if context:
            full_prompt = f"""Context: {context}

Original Question: {request.question}
Rewritten Question: {rewritten_query}

Please provide a comprehensive answer based on the available information."""
        else:
            full_prompt = f"Original Question: {request.question}\nRewritten Question: {rewritten_query}"

        response = rag_agent.run(full_prompt)
        content = response.content.strip()
        
        if not content or any(phrase in content.lower() for phrase in [
            "i don't discuss",
            "i don't answer",
            "i cannot respond",
            "i don't know",
            "empty"
        ]):
            content = ""

        logger.info(f"Result: {content}")
        return QueryResponse(
            answer=content,
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_text_file(file: UploadFile = File(...), gemini_api_token: str = Form(...)):
    try:
        # Validate API key
        if not gemini_api_token:
            logger.error("No Gemini API key provided")
            raise HTTPException(status_code=400, detail="Gemini API key is required")
        
        try:
            test_embedder = GeminiEmbedder(api_key=gemini_api_token)
            test_embedder.embed_query("test")
        except Exception as e:
            logger.error(f"Invalid Gemini API key: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid Gemini API key")

        # Validate file type
        if not file.filename.endswith('.txt'):
            logger.error(f"Invalid file type: {file.filename}. Only .txt files are supported")
            raise HTTPException(status_code=400, detail="Only .txt files are supported")

        # Read file content
        content = await file.read()
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            logger.error(f"File {file.filename} is not a valid UTF-8 text file")
            raise HTTPException(status_code=400, detail="File must be a valid UTF-8 text file")

        # Check if file already exists in Qdrant
        if not qdrant_client:
            logger.error("Qdrant client not initialized")
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")

        if check_qdrant_documents(qdrant_client, file.filename, "text"):
            logger.info(f"File {file.filename} already exists in Qdrant")
            return UploadResponse(
                message=f"File {file.filename} already exists in Qdrant",
                file_name=file.filename,
                chunks_added=0
            )

        # Process text content
        texts = process_text_content(content, file.filename)
        if not texts:
            logger.error(f"Failed to process text content for {file.filename}")
            raise HTTPException(status_code=500, detail="Failed to process text content")

        # Create or update vector store
        global vector_store
        vector_store = create_vector_store(qdrant_client, texts, gemini_api_token)
        if not vector_store:
            logger.error(f"Failed to create vector store for {file.filename}")
            raise HTTPException(status_code=500, detail="Failed to create vector store")

        logger.info(f"Successfully uploaded and processed {file.filename}")
        return UploadResponse(
            message=f"Successfully uploaded and processed {file.filename}",
            file_name=file.filename,
            chunks_added=len(texts)
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

def main():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = input("Enter Google API key: ")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    qdrant_client, vector_store = init_qdrant()
    if not qdrant_client:
        print("Failed to connect to Qdrant. Exiting.")
        return
    
    file_name = os.path.basename(FILE_PATH)
    if os.path.exists(FILE_PATH):
        if not check_qdrant_documents(qdrant_client, file_name, "text"):
            print(f"Processing {file_name}...")
            texts = process_text(FILE_PATH)
            if texts:
                vector_store = create_vector_store(qdrant_client, texts, os.environ["GOOGLE_API_KEY"])
                print(f"Added file: {file_name}")
            else:
                print(f"Failed to process {file_name}")
        else:
            print(f"File {file_name} already processed in Qdrant")
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=GeminiEmbedder(api_key=os.environ["GOOGLE_API_KEY"])
            )
    else:
        print(f"File {FILE_PATH} not found")
        return
    
    print("\nWelcome to the RAG Q&A system. Type your question (or 'exit' to quit):")
    while True:
        prompt = input("> ")
        if prompt.lower() == 'exit':
            break
        
        try:
            query_rewriter = get_query_rewriter_agent(os.environ["GOOGLE_API_KEY"])
            rewritten_query = query_rewriter.run(prompt).content
            print(f"Original query: {prompt}")
            print(f"Rewritten query: {rewritten_query}")
        except Exception as e:
            print(f"Error rewriting query: {str(e)}")
            rewritten_query = prompt
        
        context = ""
        docs = []
        if vector_store:
            has_docs, docs = check_document_relevance(rewritten_query, vector_store, threshold=0.7)
            if has_docs:
                context = "\n\n".join([d.page_content for d in docs])
                print(f"Found {len(docs)} relevant documents")
        
        try:
            rag_agent = get_rag_agent(os.environ["GOOGLE_API_KEY"])
            if context:
                full_prompt = f"""Context: {context}

Original Question: {prompt}
Rewritten Question: {rewritten_query}

Please provide a comprehensive answer based on the available information."""
            else:
                full_prompt = f"Original Question: {prompt}\nRewritten Question: {rewritten_query}"
                print("No relevant documents found.")
            
            if docs:
                print("\nDocument sources:")
                for i, doc in enumerate(docs, 1):
                    source_name = doc.metadata.get("file_name", "unknown")
                    print(f"Source {i} from {source_name}:")
                    print(f"{doc.page_content[:200]}...")
                    
            response = rag_agent.run(full_prompt)
            content = response.content.strip()
            if not content or any(phrase in content.lower() for phrase in [
                "i don't discuss",
                "i don't answer",
                "i cannot respond",
                "i don't know",
                "empty"
            ]):
                print("")
            else:
                print("\nAnswer:")
                print(content)
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
        
        print("\nType another question (or 'exit' to quit):")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)