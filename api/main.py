import os
import json
from dotenv import load_dotenv
from typing import List
from pinecone import Pinecone

# FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# RAG
from pymupdf4llm import to_markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document

# Environment variables
load_dotenv(".env") 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class Request(BaseModel):
    messages: List[dict]

app = FastAPI(
    title="RAG API",
    description="API para chatbot con RAG utilizando OpenAI y Pinecone",
    version="1.0.3"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

# OpenAI
client = OpenAI()

# Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "chatbot"
    index = pc.Index(index_name)
    namespace = "md_docs"
    print("Index created successfully!")
except Exception as error:
    print("Error al conectar con Pinecone:", error)

# Embeddings
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    print("Error al crear el modelo de embeddings:", e)

# Vectorstore y Docstore
vectorstore = PineconeVectorStore(embedding=embeddings, index=index, namespace=namespace)
store = InMemoryStore()

# Splitters para documentos "padre" e "hijo"
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Definir el ParentDocumentRetriever antes de usarlo en la indexación
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Carga e indexación de documentos (se elimina el contenido actual y se reindexan)
try:
    # Intentar eliminar todos los vectores en el namespace
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Existing vectors in namespace '{namespace}' have been deleted.")
    except Exception as e:
        if "Namespace not found" in str(e):
            print(f"Namespace '{namespace}' not found, skipping deletion.")
        else:
            raise e

    docs_md = []
    for doc_path in [
        "api/assets/calendario-academico-mad-abril-agosto-2025.pdf",
        "api/assets/introduccion-mad.pdf",
        "api/assets/preguntas-frecuentes-mad.pdf",
        "api/assets/preguntas-frecuentes-eva.pdf"
    ]:
        md_pages = to_markdown(doc=doc_path, page_chunks=True)  # Page_Chunks -> Extrae por página
        for page_data in md_pages:
            filtered_metadata = {
                "source": os.path.basename(doc_path),
                "page": page_data["metadata"]["page"],
                "page_count": page_data["metadata"]["page_count"]
            }
            docs_md.append({
                "metadata": filtered_metadata,
                "text": page_data["text"]
            })
    docs = [
            Document(
                page_content=doc["text"],
                metadata=doc["metadata"]
            )
            for doc in docs_md
        ]

    # Añadir documentos al Docstore y reindexar
    retriever.add_documents(docs)
    print("Documents uploaded successfully!")
except Exception as error:
    print("Error during re-indexing of documents:", error)


def stream_data_with_rag(messages: List[ChatCompletionMessageParam], protocol: str = 'data'):
    # Extraer la última pregunta enviada por el usuario (opcionalmente, para recuperar contexto de documentos)
    last_query = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            last_query = message.get("content", "")
            break
    if not last_query:
        last_query = " "

    # Recuperar documentos relevantes para la última consulta utilizando el retriever.
    docs = retriever.invoke(last_query)
    docs_text = "".join([doc.page_content for doc in docs])

    # Prompt para el chatbot que incluye el contexto recuperado
    system_prompt = ("""
        # Instrucciones para el Sistema: 
        Genera respuestas para las preguntas del usuario a partir del contexto proporcionado.
        *FINGE que la información proporcionada en 'CONTEXTO' es de tu conocimiento general para que la interacción sea más agradable*
        EVITA FRASES como 'segun la información', 'según los documentos' 'de acuerdo a la información' etc.
        Responde con explicaciones claras y detalladas. 
        *Asegúrante de proporcionar los LINKS que vienen dentro del contexto proporcionalo, como recomendación para el usuario y su aprendizaje;*
        Al final de la respuesta menciona de que 'páginas' se obtuvo la información con el nombre del documento correspondientes  
        (ejemplo: '(Información obtenida de páginas: 5, 10 del Plan Docente)')reemplaza por los valores reales y si son varios documentos indica específicamente a que documento corresponde a la página.
        *Responde las siguientes preguntas basándote únicamente en el siguiente contexto*
        Si la pregunta está fuera de contexto no la respondas y menciona que solo posees información del curso de introducción y provee alguna recomendación de donde investigar.
        A las palabras más importantes de tu respuesta resaltalas con negrita
        # Contexto: {context}
    """)

    system_prompt_formatted = system_prompt.format(context=docs_text)

    # Construir la lista de mensajes que se enviará a OpenAI:
    # Se añade un mensaje del sistema (con el prompt que incluye el contexto) y se concatenan todos los mensajes anteriores.
    new_messages = [{"role": "system", "content": system_prompt_formatted}] + messages

    if protocol == 'data':
        stream_result = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=new_messages,
            stream=True
        )

        for chunk in stream_result:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    continue
                else:
                    yield '0:{text}\n'.format(text=json.dumps(choice.delta.content))
            # Si el chunk no contiene choices, se envía un mensaje de finalización
            if chunk.choices == []:
                usage = chunk.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                yield 'e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
                    reason="stop",
                    prompt=prompt_tokens,
                    completion=completion_tokens
                )

        # Mensaje de cierre del stream
        yield '2:[{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0},"isContinued":false}]\n'

# API
@app.post("/api/chat", response_class=StreamingResponse)
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    try:
        messages = request.messages
        response = StreamingResponse(stream_data_with_rag(messages, protocol))
        response.headers["x-vercel-ai-data-stream"] = "v1"
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "An error occurred during processing the request. Please check the API and try again.",
                "detail": str(e)
            }
        )
    
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}