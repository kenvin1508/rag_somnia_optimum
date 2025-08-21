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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Chỉ log INFO trở lên cho ứng dụng
logging.getLogger('urllib3').setLevel(logging.WARNING)  # Tắt log chi tiết từ urllib3
logging.getLogger('langchain').setLevel(logging.WARNING)  # Tắt log chi tiết từ langchain
logging.getLogger('qdrant_client').setLevel(logging.WARNING)  # Tắt log chi tiết từ qdrant_client
logging.getLogger('agno').setLevel(logging.WARNING)  # Tắt log chi tiết từ agno

# Hard-coded configurations
GOOGLE_API_KEY = "AIzaSyBawZkl-ndb38sxze7uye_NLjDhuS3zFLk"  # Thay bằng key thật
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.0Y8ITL3SF-maASxQ0qtlnONHCsF7mkD9vLzGO79jxw0"  # Thay bằng key thật
QDRANT_URL = "https://2f9b4767-59d5-4dd1-bf47-0745875beb91.us-west-1-0.aws.cloud.qdrant.io"  # Thay bằng URL thật
FILE_PATH = "messages.txt"  # Thay bằng đường dẫn thực tế đến messages.txt
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
    def __init__(self, model_name="models/text-embedding-004"):
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = model_name
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
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

EMBEDDING_MODEL = GeminiEmbedder()
EMBEDDING_DIM = 768  # Kích thước vector cho Gemini

def check_qdrant_documents(client, identifier: str, source_type: str) -> bool:
    """Check if documents exist in Qdrant."""
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
    """Initialize Qdrant client and vector store."""
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        collections = client.get_collections().collections
        if any(c.name == COLLECTION_NAME for c in collections):
            logger.info(f"Collection {COLLECTION_NAME} exists, initializing vector store")
            try:
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    embedding=EMBEDDING_MODEL
                )
                # Ensure indexes for metadata
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
                logger.info("Created payload indexes for metadata")
                return client, vector_store
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {str(e)}")
                raise e
        else:
            logger.info(f"Collection {COLLECTION_NAME} does not exist, will create when needed")
            return client, None
    except Exception as e:
        logger.error(f"Qdrant connection failed: {str(e)}")
        print(f"Qdrant connection failed: {str(e)}")
        return None, None

def clean_text(text: str) -> str:
    import re
    # Xóa khoảng trắng thừa và ký tự đặc biệt không mong muốn
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s,.?!]', '', text)  # Giữ chữ, số, dấu câu cơ bản
    return text

def process_text(file_path: str) -> List:
    """Process and clean text file, add source metadata."""
    try:
        logger.info(f"Processing text file: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # Clean text content
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
    
def add_new_data_to_qdrant(file_path: str, collection_name: str = "gemini-thinking-agent-agno") -> bool:
    """Add new data to existing Qdrant collection."""
    try:
        # Initialize Qdrant client
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        logger.info(f"Connected to Qdrant for adding data to {collection_name}")

        # Process new text file
        file_name = os.path.basename(file_path)
        if not check_qdrant_documents(client, file_name, "text"):
            texts = process_text(file_path)
            if not texts:
                logger.error(f"No valid chunks created from {file_path}")
                return False

            # Initialize vector store for existing collection
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=EMBEDDING_MODEL
            )

            # Add new documents in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                vector_store.add_documents(batch)
                logger.info(f"Added batch {i // batch_size + 1} of {len(texts)} documents")
            
            logger.info(f"Successfully added {len(texts)} chunks from {file_name} to {collection_name}")
            return True
        else:
            logger.info(f"File {file_name} already exists in {collection_name}")
            return False
    except Exception as e:
        logger.error(f"Error adding new data to Qdrant: {str(e)}")
        print(f"Error adding new data: {str(e)}")
        return False
    
def create_vector_store(client, texts):
    """Create and initialize vector store with documents."""
    try:
        logger.info(f"Creating vector store for {len(texts)} documents")
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
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
            embedding=EMBEDDING_MODEL
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

def get_query_rewriter_agent() -> Agent:
    return Agent(
        name="Query Rewriter",
        model=Gemini(id="gemini-2.0-flash"),
        instructions=QUERY_REWRITER_INSTRUCTIONS,
        show_tool_calls=False,
        markdown=True,
    )

def get_rag_agent() -> Agent:
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash"),
        instructions=SYSTEM_MESSAGE,
        show_tool_calls=True,
        markdown=True,
    )

def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    """Check if documents in vector store are relevant to the query."""
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

class QueryResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    """Initialize Qdrant and vector store on FastAPI startup."""
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
    
    global qdrant_client, vector_store
    qdrant_client, vector_store = init_qdrant()
    if not qdrant_client:
        logger.error("Failed to connect to Qdrant during startup")
        raise HTTPException(status_code=500, detail="Failed to connect to Qdrant")
    
    file_name = os.path.basename(FILE_PATH)
    if os.path.exists(FILE_PATH):
        if not check_qdrant_documents(qdrant_client, file_name, "text"):
            logger.info(f"Processing {file_name} during startup")
            texts = process_text(FILE_PATH)
            if texts:
                if vector_store:
                    logger.info(f"Adding {len(texts)} chunks to vector store")
                    batch_size = 100
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        vector_store.add_documents(batch)
                else:
                    vector_store = create_vector_store(qdrant_client, texts)
                logger.info(f"Added file: {file_name}")
            else:
                logger.error(f"Failed to process {file_name}")
        else:
            logger.info(f"File {file_name} already processed in Qdrant")
    else:
        logger.error(f"File {FILE_PATH} not found")
        raise HTTPException(status_code=400, detail=f"File {FILE_PATH} not found")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Handle query requests via API."""
    try:
        query_rewriter = get_query_rewriter_agent()
        rewritten_query = query_rewriter.run(request.question).content
        logger.info(f"Rewritten query: {rewritten_query}")

        context = ""
        docs = []
        if vector_store:
            has_docs, docs = check_document_relevance(rewritten_query, vector_store, threshold=0.7)
            if has_docs:
                context = "\n\n".join([d.page_content for d in docs])
                logger.info(f"Found {len(docs)} relevant documents")

        rag_agent = get_rag_agent()
        if context:
            full_prompt = f"""Context: {context}

Original Question: {request.question}
Rewritten Question: {rewritten_query}

Please provide a comprehensive answer based on the available information."""
        else:
            full_prompt = f"Original Question: {request.question}\nRewritten Question: {rewritten_query}"
            logger.info("No relevant documents found")

        response = rag_agent.run(full_prompt)
        content = response.content.strip()
        
        if not content or any(phrase in content.lower() for phrase in [
            "i don't discuss",
            "i don't answer",
            "i cannot respond",
            "i don't know",
            "empty"
        ]):
            content = "empty"

        documents = [
            {"source": doc.metadata.get("file_name", "unknown"), "content": doc.page_content[:200] + "..."}
            for doc in docs
        ]

        return QueryResponse(
            answer=content,
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def main():
    """Main function to run the console-based RAG application."""
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)
    
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
                if vector_store:
                    logger.info(f"Adding {len(texts)} chunks to vector store")
                    batch_size = 100
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        vector_store.add_documents(batch)
                else:
                    vector_store = create_vector_store(qdrant_client, texts)
                print(f"Added file: {file_name}")
            else:
                print(f"Failed to process {file_name}")
        else:
            print(f"File {file_name} already processed in Qdrant")
    else:
        print(f"File {FILE_PATH} not found")
        return
    
    print("\nWelcome to the RAG Q&A system. Type your question (or 'exit' to quit):")
    while True:
        prompt = input("> ")
        if prompt.lower() == 'exit':
            break
        
        try:
            query_rewriter = get_query_rewriter_agent()
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
            rag_agent = get_rag_agent()
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