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
    "I am a consumer who wants to stay ahead of the crowd by keeping in the loop about the latest trends":
        "I am a consumer who wants to be ahead of the curve in terms of the latest fashions and trends, you are an assistant who is going to help me with tips and tricks to be trendy by reviewing some information I am sending to you from interesting sources",
    "You are a helpful assistant who is helping me uncover interesting insights, hot-topics and trends":
        "It's your job to help me find trends, insights and hot topics that people may find interesting regarding particular topics.  I am going to feed you some information and your response will be based on this",
    "I am a PR specialist and I want to coin new trends to become an authoritative industry voice":
        "I am a public relations professional and I want to coin new trends, to create new trend terms. For example 'quiet luxury' is a new trend, and it is catchy and it basically means minimalist luxury. I want to come up with new trend terms based on information that I am sending, help me do this by reviewing the information explaining why it could be considered a trend."
}

st.title('Trend Analysis Tool')
selected_system_message = st.selectbox("Choose your role:", list(system_messages.keys()))
top_k = st.slider("Select the number of top responses (TOPK):", min_value=1, max_value=20, value=5)
user_input = st.text_input("What would you like to know?")

def fetch_vectors(vector_ids):
    response = index.fetch(ids=vector_ids)
    if 'results' in response:
        return {item['id']: {
            'text_chunk': item['metadata'].get('text_chunk', ''),
            'transcript_text': item['metadata'].get('transcript_text', ''),
            'values': item['values']
        } for item in response['results']}
    return {}

def generate_response_with_gpt3(responses):
    url = "https://ai-api-dev.dentsu.com/openai/deployments/GPT4-8K/chat/completions?api-version=2024-02-01"
    headers = {
        'Content-Type': 'application/json',
        'x-service-line': 'Creative',
        'x-brand': 'carat',
        'x-project': 'data-journalist-radio',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': generate_response_api_key,
    }
    data = {
        "model": "gpt-4-8k",
        "messages": [{"role": "system", "content": system_messages[selected_system_message]}] + 
                   [{"role": "user", "content": text} for text in responses],
        "max_tokens": 1000
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "Failed to generate response.") if response.ok else "Failed to generate response."

def process_user_query(user_query, top_k):
    url = "https://ai-api-dev.dentsu.com/openai/deployments/TextEmbeddingAda2/embeddings?api-version=2024-02-01"
    headers = {
        'Content-Type': 'application/json',
        'x-service-line': 'creative',
        'x-brand': 'carat',
        'x-project': 'CraigJonesProject',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': embed_api_key,
    }
    data = {"input": user_query, "user": "streamlit_user", "input_type": "query"}
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        response_data = response.json()
        embedding_vector = response_data['data'][0]['embedding'] if 'data' in response_data and 'embedding' in response_data['data'][0] else None
        if embedding_vector:
            query_results = index.query(vector=embedding_vector, top_k=top_k, include_metadata=True)
            responses = [match['metadata'].get('text_chunk', '') + " " + match['metadata'].get('transcript_text', '') for match in query_results['matches']]
            return generate_response_with_gpt3(responses)
        else:
            st.error("Failed to retrieve valid embedding vector.")
    else:
        st.error(f"Failed to fetch embedding: {response.status_code} - {response.text}")
    return None

if st.button('Analyze'):
    response_text = process_user_query(user_input, top_k)
    st.write(response_text if response_text else "No response generated.")

