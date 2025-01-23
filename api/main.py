import os
from dotenv import load_dotenv
from typing import List
import json

#De utils
from .utils.prompt import prompt_template

#FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

#LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Cargar variables de entorno
load_dotenv(".env") 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

#Modelo de Respuestas
class Request(BaseModel):
    messages: List[dict]

app = FastAPI(
    title="RAG API",
    description="API para chatbot con RAG usando Gemini y Pinecone",
    version="1.0.0"
)

# CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

try:
    # Modelo de Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
except Exception as e:
    raise RuntimeError(f"Error al inicializar los embeddings: {str(e)}")

try:
    # Conexión a Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "tutormad"  # se está usando un solo índice pero con múltiples namespaces
    index = pc.Index(index_name)
    namespace = "curso_introduccion_MAD"  # Importante direccionar correctamente el namespace
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
except Exception as e:
    raise RuntimeError(f"Error al conectar con Pinecone o inicializar el vector store: {str(e)}")

try:
    # Configurar el modelo de Gemini usando Langchain
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Modelo especializado en resúmenes
        google_api_key=GOOGLE_API_KEY,
        streaming=True,
        temperature=0.4,
        max_tokens=1024  # Máximo de tokens de salida en las respuestas del LLM
    )
except Exception as e:
    raise RuntimeError(f"Error al inicializar el modelo LLM: {str(e)}")

def stream_rag_response(messages: List[dict]):
    try:
        """Devuelve respuestas generadas con RAG como un stream compatible con Vercel AI SDK."""
        #Extraer última pregunta del usuario
        question = messages[-1]['content']
        
        # Configuración del recuperador
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.6},
        )

        # Recuperar contexto desde Pinecone
        docs = retriever.invoke(question)
        context_items = [
            #'Documento: 'Guía Didáctica' en la Página 45: '
            f"Documento: {doc.metadata.get('doc_name', 'desconocido')} en la Página: {int(doc.metadata.get('page'))}: {doc.page_content}: "
            for doc in docs   
        ]
        #Formatear como una sola cadena de texto
        context = "\n".join(context_items)
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