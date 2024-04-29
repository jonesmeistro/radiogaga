import streamlit as st
import requests
import json
from pinecone import Pinecone

# Retrieve API keys from Streamlit secrets
embed_api_key = st.secrets["EMBED_API_KEY"]
generate_response_api_key = st.secrets["GENERATE_RESPONSE_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "index-llm-yt"
    index = pc.Index(name=index_name) if index_name in pc.list_indexes().names() else None
    if index is None:
        st.error(f"Index {index_name} does not exist.")
except Exception as e:
    st.error(f"Failed to initialize Pinecone index: {e}")

# Dropdown options for system messages
system_messages = {
    "I am a consumer who wants to stay ahead of the crowd": "I am a consumer who wants to be ahead of the curve in terms of the latest fashions and trends...",
    "You are a helpful assistant": "It's your job to help me find trends, insights and hot topics...",
    "I am a PR specialist": "I am a public relations professional and I want to coin new trends..."
}

st.title('Trend Analysis Tool')
selected_system_message = st.selectbox("Choose your role:", list(system_messages.keys()))
top_k = st.slider("Select the number of top responses (TOPK):", min_value=1, max_value=20, value=5)
user_input = st.text_input("What would you like to know?")

def fetch_vectors(vector_ids):
    response = index.fetch(ids=vector_ids)
    return {item['id']: item['metadata'] for item in response.get('results', [])}

def generate_response_with_gpt3(responses):
    url = "https://ai-api-dev.dentsu.com/openai/deployments/GPT4-8K/chat/completions?api-version=2024-02-01"
    headers = {
        'x-service-line': 'Creative',
        'x-brand': 'carat',
        'x-project': 'data-journalist-radio',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': generate_response_api_key,
    }
    messages = [{"role": "system", "content": system_messages[selected_system_message]}] + \
               [{"role": "user", "content": text} for text in responses]
    data = {"model": "gpt-4-8k", "messages": messages, "max_tokens": 1000}
    response = requests.post(url, headers=headers, json=data)
    return response.json().get('choices')[0].get('message', {}).get('content', "Failed to generate response.")

def process_user_query(user_query, top_k):
    url = "https://ai-api-dev.dentsu.com/openai/deployments/TextEmbeddingAda2/embeddings?api-version=2024-02-01"
    headers = {
        'x-service-line': 'creative',
        'x-brand': 'carat',
        'x-project': 'CraigJonesProject',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': embed_api_key,
    }
    data = {"input": user_query, "user": "streamlit_user", "input_type": "query"}
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        response_data = response.json()
        embedding_vector = response_data['data'][0].get('embedding') if 'data' in response_data else None
        if embedding_vector:
            query_results = index.query(vector=embedding_vector, top_k=top_k, include_metadata=True)
            text_responses = [match['metadata']['text_chunk'] for match in query_results['matches'] if 'text_chunk' in match['metadata']]
            return generate_response_with_gpt3(text_responses)
        st.error("Failed to retrieve valid embedding vector.")
    else:
        st.error(f"Failed to fetch embedding: {response.status_code} - {response.text}")
    return "Failed to process your query."

if st.button('Analyze'):
    if user_input:
        response_text = process_user_query(user_input, top_k)
        st.write(response_text)
    else:
        st.write("Please enter a question to analyze.")

