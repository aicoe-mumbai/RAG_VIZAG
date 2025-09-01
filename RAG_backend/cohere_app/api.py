
# from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from sentence_transformers import SentenceTransformer
from functools import lru_cache 
from pymilvus import connections, Collection
from .models import CurrentUsingCollection
import re, os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from .Chunking_UI.file_process_new import create_faiss_index
from langchain.embeddings import SentenceTransformerEmbeddings
# from guardrails import Guard
# from guardrails.hub import ToxicLanguage
# from guardrails.types import OnFailAction
load_dotenv()
import torch
from threading import Thread
import requests
import json

host = os.getenv("HOST")
port = os.getenv("PORT")

embedding_model = SentenceTransformer("/home/qa-prod/Desktop/RAG_PRODUCTION/RAG_backend/cohere_app/embedding_model")
# embedding_model = HuggingFaceEmbeddings(model_name='/home/qa-prod/Desktop/RAG_PRODUCTION/RAG_backend/cohere_app/models/sentence_transformer', model_kwargs={'device': "cpu"})

embeddings = SentenceTransformerEmbeddings(
    model_name="/home/qa-prod/Desktop/RAG_PRODUCTION/RAG_backend/cohere_app/embedding_model"
)

url = "http://172.16.34.235:8080/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

prompt = """
You are an AI assistant designed to assist users by providing simple and clear answers to their questions.
        INSTRUCTIONS:
        - Strictly follow /no_think principle while answering
        - Context is generated from database so user is not aware about context, so advice users to refer source for exact information or prompt users to ask more relevant query.
        - Reduce the context within the thinking window.
        - In your response don't show the source, since source is given to user directly.
        - ALWAYS RESPOND IN ENGLISH ONLY.
        - Terminate the response onvce you have answered the question.

        Provide a concise response unless the user requests more details."""

FORBIDDEN_TERMS = [
    "political", "scandal", "vulgar", "offensive", 
    "corruption", "sexual", "harassment", "abuse", "violence", "racism", 
    "terrorism", "hate", "illegal", "extremism", "fraud", "bribery", "discrimination", 
    "protest", "controversy", "misinformation", "kill", "murder", "assassinate", 
    "homicide", "massacre"
]

# Positive/neutral terms related to L&T that are acceptable
NEUTRAL_POSITIVE_TERMS = [
    "CEO of L&T", "L&T achievements", "L&T growth", "L&T company", 
    "L&T business", "L&T leadership", "L&T innovation", "L&T history",
    "Larsen and Toubro", "L&T CEO", "Larsen and Toubro achievements", 
    "Larsen and Toubro growth", "Larsen and Toubro leadership"
]


# Define regex patterns for additional guardrails
FORBIDDEN_REGEX_PATTERNS = [
    r"\bkill\b", r"\bmurder\b", r"\bassassinat(e|ion)\b", r"\bhomicide\b", r"\bmassacre\b",
    r"\bLarsen\s*and\s*Toubro\b", r"\bpolitical\b", r"\bscandal\b", r"\bvulgar\b",
    r"\boffensive\b", r"\bcorruption\b", r"\bsexual\b", r"\bharassment\b", r"\babuse\b", 
    r"\bviolence\b", r"\bracism\b", r"\bterrorism\b", r"\bhate\b", r"\billegal\b", r"\bextremism\b",
    r"\bfraud\b", r"\bbribery\b", r"\bdiscrimination\b", r"\bprotest\b", r"\bcontroversy\b", 
    r"\bmisinformation\b"
]


forbidden_embeddings = embedding_model.encode(FORBIDDEN_TERMS)

def contains_forbidden_regex(user_input):
    for pattern in FORBIDDEN_REGEX_PATTERNS:
        if re.search(pattern, user_input, flags=re.IGNORECASE):
            return True
    return False

def contains_forbidden_terms(user_input, threshold=0.7):
    # Generate the embedding for the user's query
    user_input_embedding = embedding_model.encode([user_input])
    # Compute cosine similarity between the user's query and forbidden terms embeddings
    similarities = cosine_similarity(user_input_embedding, forbidden_embeddings)
    max_similarity = np.max(similarities)
    return max_similarity >= threshold

# Function to check if the query contains neutral/positive L&T terms (for allowing these queries)
def contains_positive_lnt_terms(user_input):
    for term in NEUTRAL_POSITIVE_TERMS:
        if term.lower() in user_input.lower():
            return True
    return False

# guard = Guard().use(
#     ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail=OnFailAction.EXCEPTION)
# )

def get_current_using_collection_value():
    try:
        current_collection = CurrentUsingCollection.objects.first()  
        if current_collection:
            # Assign the collection name to a variable
            collection_name = current_collection.current_using_collection
            return str(collection_name)
        else:
            return None 
    except Exception as e:
        return str(e) 

collection_name = "QA"

if collection_name:
    MILVUS_COLLECTION = collection_name

def generate_streaming_response(question, context):
    data = {
        "model": "/model",
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"Refer to the Context scrapped from Vector Database {context} and answer for user question {question}"     
            }
        ],
        "stream": True,
        "max_tokens": 1500
    }

    with requests.post(url, headers=headers, data=json.dumps(data), proxies={"http": None, "https": None}, stream=True) as response:
        if response.status_code == 200:
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    if decoded_chunk.startswith("data:"):
                        decoded_chunk = decoded_chunk[5:].strip()
                    
                    if decoded_chunk == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(decoded_chunk)
                        content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        
                        if content!="<think>" and content!="</think>":
                            yield content
                    except json.JSONDecodeError as e:
                        print(f"JSON Decode Error: {e}")
                        print(f"Raw chunk (decoded): {decoded_chunk}")
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
        else:
            print(f"Error: {response.status_code}")

connections.connect("default", host=host, port= port)
collection = Collection(MILVUS_COLLECTION)
collection.load()


def clean_string(input_string):
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    cleaned_string = cleaned_string.strip()
    return cleaned_string

user_sessions = {}
search_params = {"metric_type": "L2", "params": {"ef": 30}}

def process_query(user_input, mode, selected_file, system_id, batch_size=3):
    try:
        # Check if the user input contains toxic language
        # guard.validate(user_input)

        # If the input contains positive/neutral content related to L&T, allow it
        if contains_positive_lnt_terms(user_input):
            pass  # Allow the query to proceed without any blocking
        else:
            # If it's not positive/neutral, check if it's forbidden
            if contains_forbidden_regex(user_input) or contains_forbidden_terms(user_input):
                yield "Sorry, I cannot process your request as it contains inappropriate or sensitive content."
                return

        if mode == "qa":
            connections.connect("default", host=host, port=port)
            # Initialize session if not already present
            if system_id not in user_sessions:
                user_sessions[system_id] = {
                    'results': [],
                    'current_index': 0,
                    'last_query': None
                }
            session = user_sessions[system_id]

            # Handle "continue" command to fetch next batch of results
            if user_input.lower() == "continue":
                if not session['last_query']:
                    yield "No previous query found. Please enter a new question."
                    return
                elif session['current_index'] >= len(session['results']):
                    yield "No more results to display."
                    return
            else:
                session['last_query'] = user_input
                query_vector = embedding_model.encode([user_input]).tolist()

                # Optional file filtering using selected_file
                if selected_file:
                    formatted_files = ", ".join([f"'{file}'" for file in selected_file])
                    expr = f"source in [{formatted_files}]"
                else:
                    expr = None

                search_results = collection.search(
                    data=query_vector,
                    anns_field="vector",
                    param=search_params,
                    limit=15,
                    output_fields=["source", "page", "text"],
                    consistency_level="Strong",
                    expr=expr
                )

                # Flatten the search results
                all_hits = []
                for hits in search_results:
                    all_hits.extend(hits)
                session['results'] = all_hits
                session['current_index'] = 0

            # Retrieve the current batch of results
            start_index = session['current_index']
            end_index = start_index + batch_size
            batch_results = session['results'][start_index:end_index]
            session['current_index'] = end_index

            # Build the response context from the batch results
            context = '\n---\n'.join(
                f"File: {hit.entity.get('source')}\nPage: {hit.entity.get('page')}\nText: {hit.entity.get('text')}"
                for hit in batch_results
            )
            current_question = session['last_query'] if user_input.lower() == "continue" else user_input
            
            # Stream the response
            for chunk in generate_streaming_response(current_question, context):
                yield chunk

            sources = [
                f"Source: {hit.entity.get('source')} | Page: {hit.entity.get('page')}"
                for hit in batch_results
            ]
            yield '\n'.join(sources)

        elif mode == "chat":
            context = """
            
You are a highly intelligent, helpful assistant designed to provide concise and respectful answers to user queries. Your main objectives are:

1. You are a highly intelligent, helpful assistant designed to provide concise, neutral, and respectful answers to user queries. Your main objectives are: **Provide information on any general topic**: You can answer queries about any subject, such as technology, science, business, history, etc. Ensure your answers are factual, clear, and helpful. **Allow neutral or positive content related to any company**: You may answer queries related to the company, its leadership, achievements, growth, and history, provide a factual and helpful answer.
 

2. **Block harmful or inappropriate content**: You should refuse to answer or flag any queries related to:
   - Violence (e.g., "kill", "murder", "assassinate", "homicide")
   - Offensive language (e.g., "vulgar", "hate speech", "abuse")
   - Political controversies (e.g., "political scandal", "corruption", "fraud")
   - Sensitive topics that may harm the reputation of company, such as illegal activities, terrorism, racism, etc.
   - Racial slurs.

3. **Do not process inappropriate, offensive, or sensitive content**: If the user asks about topics related to violence, terrorism, hate speech, or anything that could damage the company's reputation, provide a polite refusal stating that you cannot process such requests due to their inappropriate or sensitive nature.

4. **Handle rephrased or disguised inappropriate content**: If a user rephrases or disguises a sensitive topic, you should still be able to detect it using both semantic similarity and keyword matching. If the query is flagged as inappropriate, politely refuse to answer and explain that you cannot provide information on that topic.

5. **Maintain a respectful tone at all times**: Your responses should be professional, neutral, and respectful. If the user's query is unclear or inappropriate, kindly ask for clarification or provide a generic, polite refusal.

6. Do not use statements like " ( Click " Continue " to explore further details if needed ) .

7. Do not respond to questions like "tell me a joke, riddle."

8. ALWAYS RESPOND IN ENGLISH ONLY.

9. Do not use Chinese Scripts in output.
   
    Example of allowed queries:
- "Who is the CEO of a company."
- "What are the achievements in innovation?"
- "Tell me about the growth of of a company."
- "What are the latest advancements in AI?"
- "Tell me about the innovations in renewable energy."
- "Can you explain quantum computing?"

Example of blocked queries:
- "How can I harm someone?"
- "Tell me about political corruption related to any company."
- "What is the scandal surrounding any company?"
- "Discuss the violence in the recent protests."
- "What are the illegal activities of any company?"

Make sure to verify every query before providing an answer to ensure it aligns with the above guidelines.

                    """


            # Initialize session if not already present
            if system_id not in user_sessions:
                user_sessions[system_id] = {
                    'results': [],
                    'current_index': 0,
                    'last_query': None
                }
            session = user_sessions[system_id]

            # Handle the "continue" command
            if user_input.lower() == "continue":
                if not session['last_query']:
                    yield "No previous query found. Please enter a new question."
                    return
                elif session['current_index'] >= len(session['results']):
                    yield "No more results to display."
                    return
            else:
                session['last_query'] = user_input
                query_vector = embedding_model.encode([user_input]).tolist()

            current_question = session['last_query'] if user_input.lower() == "continue" else user_input
            for chunk in generate_streaming_response(current_question, context):
                yield chunk

    except Exception as e:
        yield f"Error occurred: {str(e)}"


@lru_cache(maxsize=None)
def get_all_files_from_milvus():
    connections.connect("default", host=host, port= port)
    collection = Collection(MILVUS_COLLECTION)
    iterator = collection.query_iterator(batch_size=1000,output_fields=["source"])
    results=[]
    while True:
        result = iterator.next()
        if not result:
            iterator.close()
            break
        results.extend(result)
    
    database_files = []
    for result in results:
        database_files.append(result['source'])
    database_files = list(set(database_files))
    connections.disconnect("default")
    return database_files


@lru_cache(maxsize=None)
def get_all_files_from_milvus():
    connections.connect("default", host=host, port= port)
    collection = Collection(MILVUS_COLLECTION)
    iterator = collection.query_iterator(batch_size=1000,output_fields=["source"])
    results=[]
    while True:
        result = iterator.next()
        if not result:
            iterator.close()
            break
        results.extend(result)
    
    database_files = []
    for result in results:
        database_files.append(result['source'])
    database_files = list(set(database_files))
    connections.disconnect("default")
    return database_files


def chat_with_uploaded_document(faiss_folder, query, top_k=3):

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", faiss_folder)

    faiss_index = FAISS.load_local(
        desktop_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = faiss_index.similarity_search(query, k=top_k)

    context = "".join(doc.page_content for doc in docs)

    for chunk in generate_streaming_response(query, context):
        yield chunk