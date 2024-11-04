import os
import sys
import time
import signal
import warnings
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import google.generativeai as genai
from dotenv import load_dotenv
_ = load_dotenv()


def signal_handler(sig, frame):
    print('You pressed Ctrl+C! to exit')
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

def get_relevant_context_from_db(query):
    pass
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',model_kwargs = {'device':'cpu'})
    vector_db = Chroma(persist_directory='./Chroma_db',embedding_function=embedding_function,collection_name='test')
    search_results = vector_db.similarity_search(query,k=2)
    for results in search_results:
        context += results.page_content + "\n"
    return context

def generate_rag_prompt(query,context):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""
              You are a helpful and informative bot that answers questions using text from the reference context included below.\
              Be sure to respond in a complete sentence, being comprehensive and informative.\
              How ever, you are talking to a non-technical person, so keep your answers simple and easy to understand.\
              If the context is irrelevant, you can ignore it.
                    QUESTION: '{query}'
                    CONTEXT: '{context}'
    """).format(query=query, context=context)   
    return context
    
def generate_answer(prompt):
    genai.configure(api_key=os.environ["Gemini_API_KEY"])
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

welcome_message = generate_answer("Can you introduce yourself?")
print(welcome_message)

while True:
    print("-----------------------------------------------------------")
    print("What would you like to ask?")
    query = input("Query: ")
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query= query,context=context)
    answer = generate_answer(prompt)
    print("Answer: ",answer)
    
    