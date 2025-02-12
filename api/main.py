import os
from dotenv import load_dotenv
from typing import List
import json

# De utils
from .utils.prompt import prompt_template

# FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# LangChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document

from pinecone import Pinecone

# Cargar variables de entorno
load_dotenv(".env") 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Modelo de Respuestas
class Request(BaseModel):
    messages: List[dict]

app = FastAPI(
    title="RAG API",
    description="API para chatbot con RAG usando OpenAI y Pinecone",
    version="1.0.2"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   
###
try:
    # Modelo de Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    raise RuntimeError(f"Error al inicializar los embeddings: {str(e)}")

try:
    # Conexión a Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "chatbot"  # Se está usando un solo índice pero con múltiples namespaces
    index = pc.Index(index_name)
    namespace = "TestPersonal"  # Importante direccionar correctamente el namespace , namespace=namespace
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
except Exception as e:
    raise RuntimeError(f"Error al conectar con Pinecone o inicializar el vector store: {str(e)}")

try:
    # Configurar el modelo de GPT usando LangChain
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # Modelo equivalente a Gemini 1.5 Flash
        api_key=OPENAI_API_KEY,
        streaming=True,
        temperature=0.4,
        max_tokens=512  # Máximo de tokens de salida en las respuestas del LLM
    )
except Exception as e:
    raise RuntimeError(f"Error al inicializar el modelo LLM: {str(e)}")

docs_md = []
for doc_path in [
    "./api/documents/Preguntas.pdf",
    # "./api/documents/Introduccion.pdf",
    "./api/documents/CALENDARIO.pdf",
]:
    md_pages = pymupdf4llm.to_markdown(doc=doc_path, page_chunks=True)  # Extrae por página
    docs_md.append(md_pages)

if docs_md != None:
    print("Documentos correctamente cargados")

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# # The storage layer for the parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore = vectorstore,
    docstore = store,
    child_splitter = child_splitter,
    parent_splitter = parent_splitter,
)

# Flatten the list of lists into a list of Documents
docs_to_add = []
for doc_list in docs_md:
    for doc_page in doc_list:
        # create a langchain Document object for each page in the markdown
        docs_to_add.append(Document(page_content=doc_page["text"])) # Corrected line

# Now add the documents
retriever.add_documents(docs_to_add)

if retriever != None:
    print("Parent in Memory Creado")

def stream_rag_response(messages: List[dict]):
    try:
        """Devuelve respuestas generadas con RAG como un stream compatible con Vercel AI SDK."""
        # Extraer última pregunta del usuario
        question = messages[-1]['content']
        docs = retriever.invoke(question)

        docs_text = "".join(d.page_content for d in docs)
        # Formatear como una sola cadena de texto
        context = "\n".join(docs_text)
        # Construir el mensaje con contexto y pregunta
        input_data = {"context": context, "question": question}

        # Generar la respuesta como streaming desde el LLM
        prompt = ChatPromptTemplate.from_template(prompt_template).format_prompt(**input_data)
        result_stream = llm.stream(prompt.to_messages())

        # Iterar sobre los fragmentos generados y transmitirlos en formato Vercel AI SDK
        for chunk in result_stream:
            # Cada chunk será un JSON válido en el formato esperado
            yield '0:{text}\n'.format(text=json.dumps(chunk.content))

        # Mensaje de cierre del stream
        yield '2:[{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0},"isContinued":false}]\n'
    except Exception as e:
        yield json.dumps({"error": "Error inesperado durante el procesamiento.", "detail": str(e)})


#API
@app.post("/api/chat", response_class=StreamingResponse)
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    try:
        messages = request.messages
        response = StreamingResponse(stream_rag_response(messages))
        response.headers["x-vercel-ai-data-stream"] = "v1"
        return response
    except Exception as e:
        # Manejo genérico de errores
        return JSONResponse(
            status_code=500,
            content={
                "error": "Ocurrió un error inesperado durante el procesamiento.",
                "detail": str(e)
            }
        )
    
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}