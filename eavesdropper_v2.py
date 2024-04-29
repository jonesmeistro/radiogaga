import os
import requests
import streamlit as st
from pinecone import Pinecone
import json

# Environment variables
embed_api_key = os.getenv("EMBED_API_KEY")
generate_response_api_key = os.getenv("GENERATE_RESPONSE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "index-llm-yt"
    if index_name not in pc.list_indexes().names():
        st.error(f"Index {index_name} does not exist.")
    else:
        index = pc.Index(name=index_name)
except Exception as e:
    st.error(f"Failed to initialize Pinecone index: {e}")

# Dropdown options for system messages
system_messages = {
    "I am a consumer who wants to stay ahead of the crowd by keeping in the loop about the latest trends":
        "I am a consumer who wants to be ahead of the curve in terms of the latest fashions and trends, you are an assistant who is going to help me with tips and tricks to be trendy by reviewing some information I am sending to you from interesting sources",
    "You are a helpful assistant who is helping me uncover interesting insights, hot-topics and trends":
        "It's your job to help me find trends, insights and hot topics that people may find interesting regarding particular topics.  I am going to feed you some information and your response will be based on this",
    "I am a PR specialist and I want to coin new trends to become an authoritative industry voice":
        "I am a public relations professional and I want to coin new trends, to create new trend terms.  For example 'quiet luxury' is a new trend, and it is catchy and it basically means minimalist luxury.  I want to come up with new trend terms based on information that I am sending, help me do this by reviewing the information explaining why it could be considered a trend."
}

# Setup Streamlit UI
st.title('Welcome to the Eavesdropper')
selected_system_message = st.selectbox("Choose your role:", list(system_messages.keys()))
top_k = st.slider("Select the number of top responses you want:", min_value=1, max_value=20, value=5)
comments_only = st.checkbox("Filter for comments only")
user_input = st.text_input("What would you like to know?", "")

@st.cache
def fetch_vectors(vector_ids):
    response = index.fetch(ids=vector_ids)
    if 'results' in response:
        return {item['id']: {
            'text_chunk': item['metadata'].get('text_chunk', ''),
            'transcript_text': item['metadata'].get('transcript_text', ''),
            'values': item['values']
        } for item in response['results']}
    else:
        return {}

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
    messages = [{"role": "system", "content": system_messages[selected_system_message]}] + [
        {"role": "user", "content": text} for text in responses
    ]
    data = {"model": "gpt-4-8k", "messages": messages, "max_tokens": 1000}
    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content'] if 'choices' in response.json() else "Failed to generate response."

def process_user_query(user_query, top_k, comments_only):
    # Fetch embeddings and query Pinecone
    url = "https://ai-api-dev.dentsu.com/openai/deployments/TextEmbeddingAda2/embeddings?api-version=2024-02-01"
    headers = {
        'x-service-line': 'creative',
        'x-brand': 'carat',
        'x-project': 'CraigJonesProject',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': 'v8',
        'OcpApim-Subscription-Key': embed_api_key,
    }
    data = json.dumps({
        "input": user_query,
        "user": "streamlit_user",
        "input_type": "query"
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(req)
        response_data = json.loads(response.read().decode('utf-8'))
        
        if 'data' in response_data and len(response_data['data']) > 0 and 'embedding' in response_data['data'][0]:
            embedding_vector = response_data['data'][0]['embedding']
        else:
            st.error("Failed to retrieve embedding vector from response.")
            return None

        # Query Pinecone with the embedding
        query_results = index.query(vector=embedding_vector, top_k=top_k, include_metadata=True)
        if 'matches' in query_results and query_results['matches']:
            vector_ids = [match['id'] for match in query_results['matches']]
            vectors = fetch_vectors(vector_ids)
            responses = [vec['text_chunk'] + " " + vec['transcript_text'] for vec in vectors.values()]
            return generate_response_with_gpt3(responses)
        else:
            st.error("No matches found or invalid query results.")
            return "No valid data found based on the query."
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Interaction button
if st.button('Ask'):
    if user_input:
        response_text = process_user_query(user_input, top_k, comments_only)
        st.write(response_text)
    else:
        st.write("Please enter a question.")

