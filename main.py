from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from rich.padding import Padding
from rich.markdown import Markdown

import signal
import sys
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai

#region config
load_dotenv()

console = Console()
print = console.print

MODEL_PARAMETER = os.getenv('MODEL_PARAMETER')
#endregion config

# Create an panels to differentiate in the console
def ai_panel(text: str) -> Panel: 
    return Panel(
        Markdown(text), 
        title="Gemini", 
        border_style='blue'
    )

def tool_panel(text: str) -> Panel: 
    return Panel(
        text, 
        title="Tool Calls - RAG", 
        border_style='yellow'
    )

def sig_interrupt(_a, _b):
    print(Padding(" "))
    print(ai_panel("Goodbye..."))
    sys.exit(0)
    
signal.signal(signal.SIGINT, sig_interrupt)


# Get context from local vector store
def get_context(query: str) -> str:
    context = ""

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        show_progress=True
    )

    vectordb = Chroma(
        persist_directory='./chroma_db_vectors',
        embedding_function=embedding
    )

    documents = vectordb.similarity_search(query)

    print(tool_panel(f"Found {len(documents)} documents that match the query"))
          
    for page in documents:
        context += page.page_content + "\n"
    
    return context

# Get response from GenAI
def generate_response(query: str, context: str = "") -> str:
    normalized_context = context.strip("'").strip('"').replace('\n', ' ')
    prompt = (
        f"""
        - You are Gemini, {MODEL_PARAMETER}. you can answer questions using the context included below\n
        - You are to answer in a detailed manner stating all facts strictly as per the context provided\n
        - If a query lies outside the scope of the context, inform the user that it is irrelevant.\n
        - You need to analyze the context and break down your responses in a friendly and understanding manner.\n
        - The query is provided below in the format "QUERY: '<query>'"\n
        - The context is provided below in the format "CONTEXT: '<context>'"\n\n

        QUERY: '{query}'
        CONTEXT: '{normalized_context}'
        """
    )

    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents=prompt
    )
    return response.text

print(ai_panel(
    generate_response(
        "Introduce yourself in short", 
        "Provide a brief introduction based on instructions"
    )
))

while True:
    query = Prompt.ask("Query")
    context = get_context(query)
    print(ai_panel(generate_response(query, context)))