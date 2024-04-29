import os 
import requests
import streamlit as st
from pinecone import Pinecone
import numpy as np
from sklearn.cluster import KMeans
import json

# Initialize Pinecone
embed_api_key = os.getenv("EMBED_API_KEY")
generate_response_api_key = os.getenv("GENERATE_RESPONSE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "index-llm-yt"

try:
    if index_name not in pc.list_indexes().names():
        st.error(f"Index {index_name} does not exist.")
    else:
        index = pc.Index(name=index_name)
except Exception as e:
    st.error(f"Failed to initialize Pinecone index: {e}")

def get_query_embedding(user_query):
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
    data = {
        "input": user_query,
        "user": "streamlit_user",
        "input_type": "query"
    }
    response = requests.post(url, json=data, headers=headers)
    if response.ok:
        embedding_data = response.json()
        if 'data' in embedding_data and len(embedding_data['data']) > 0:
            return np.array(embedding_data['data'][0]['embedding']).reshape(1, -1)
        else:
            st.error("No embedding data found in the response.")
            return None
    else:
        st.error(f"Failed to fetch embedding: {response.status_code} - {response.text}")
        return None

def perform_clustering(embeddings, num_clusters):
    if embeddings is None or len(embeddings) == 0 or len(embeddings[0]) == 0:
        st.error("No valid embeddings provided for clustering.")
        return []
    if len(embeddings) < num_clusters:
        st.error("Not enough data points for the number of requested clusters.")
        return []
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    cluster_indices = [np.argmin(np.linalg.norm(embeddings - kmeans.cluster_centers_[i], axis=1)) for i in range(num_clusters)]
    return cluster_indices

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
    messages = [
        {"role": "system", "content": "This conversation is aimed at generating insights from YouTube data."},
        {"role": "user", "content": "Based on the following information, provide a detailed analysis:"}
    ] + [{"role": "user", "content": text} for text in responses]
    data = {
        "model": "gpt-4-8k",
        "messages": messages,
        "max_tokens": 1000
    }
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        return response.json().get('choices')[0]['message']['content']
    else:
        st.error(f"Failed to generate response: {response.status_code} - {response.text}")
        return None

def process_user_query_with_clustering(user_query, top_k, num_clusters):
    try:
        embedding_vector = get_query_embedding(user_query)
        # Ensure the embedding vector is not None and has the correct shape
        if embedding_vector is None or embedding_vector.size == 0:
            st.error("Failed to generate embedding vector for the query.")
            return None

        query_results = index.query(vector=embedding_vector, top_k=top_k, include_metadata=True)
        if not query_results['matches']:
            st.error("No matches found. Adjust your query or top_k value.")
            return None
        
        texts = [match['metadata']['text_chunk'] for match in query_results['matches'] if 'text_chunk' in match['metadata']]
        embeddings = [match['metadata'].get('embedding') for match in query_results['matches']]
        
        # Check if embeddings are valid
        if not embeddings or all(embed is None for embed in embeddings):
            st.error("Insufficient data for clustering. Reduce the number of clusters or increase top_k.")
            return None

        # Convert list of embeddings to numpy array for clustering
        embeddings_array = np.array([embed for embed in embeddings if embed is not None])
        if len(embeddings_array) < num_clusters:
            st.error(f"Insufficient embeddings for clustering. Required: {num_clusters}, Available: {len(embeddings_array)}")
            return None

        # Clustering
        representative_texts = perform_clustering(embeddings_array, num_clusters)
        selected_texts = [texts[idx] for idx, _ in representative_texts]
        return generate_response_with_gpt3(selected_texts)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# UI Components
st.title('YouTube Data Insight Generator')
top_k = st.slider("Select the number of top responses (TOPK):", min_value=1, max_value=20, value=10, help="The higher the number, the more results are used in reference.")
num_clusters = st.slider("Select the number of clusters:", min_value=1, max_value=10, value=5, help="The higher the number, the more diverse the response.")
user_input = st.text_input("What would you like to know?", "")

if st.button('Analyze'):
    if user_input:
        response_text = process_user_query_with_clustering(user_input, top_k, num_clusters)
        if response_text:
            st.write(response_text)
        else:
            st.write("Please try a different query or adjust the settings.")
    else:
        st.write("Please enter a question to analyze.")

# Additional handling for possible user interaction errors
if st.session_state.get('error', False):
    st.error("An error occurred. Please check the inputs and try again.")


