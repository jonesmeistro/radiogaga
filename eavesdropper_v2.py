import os 
import requests
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from sklearn.cluster import KMeans
import json

embed_api_key = os.getenv("EMBED_API_KEY")
generate_response_api_key = os.getenv("GENERATE_RESPONSE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

try:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "index-llm-yt"
    if index_name not in pc.list_indexes().names():
        st.error(f"Index {index_name} does not exist.")
    else:
        index = pc.Index(name=index_name)
except Exception as e:
    st.error(f"Failed to initialize Pinecone index: {e}")

# Supporting Functions
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
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        response_data = response.json()
        if 'data' in response_data and 'embedding' in response_data['data'][0]:
            return response_data['data'][0]['embedding']
        else:
            st.error("Embedding not found in response data.")
            return None
    else:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return None

def perform_clustering(embeddings, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(embeddings))
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    representative_texts = []
    for i in range(num_clusters):
        indices_in_cluster = np.where(cluster_labels == i)[0]
        cluster_embeddings = np.array([embeddings[idx] for idx in indices_in_cluster])
        distances = np.linalg.norm(cluster_embeddings - centroids[i], axis=1)
        closest_point_index = indices_in_cluster[np.argmin(distances)]
        representative_texts.append((closest_point_index, distances.min()))

    return representative_texts

# Function to generate response with GPT-3
def generate_response_with_gpt3(responses):
    url = "https://ai-api-dev.dentsu.com/openai/deployments/GPT4-8K/chat/completions?api-version=2024-02-01"
    headers = {
        'x-service-line': 'Creative',
        'x-brand': 'carat',
        'x-project': 'data-journalist-radio',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': os.getenv("generate_response_api_key"),
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
    return response.json().get('choices')[0]['message']['content'] if response.status_code == 200 else "Failed to generate response."

# Function to fetch and process data
def process_user_query_with_clustering(user_query, top_k, num_clusters):
    try:
        embedding_vector = get_query_embedding(user_query)
        if not embedding_vector:
            st.error("Failed to generate embedding vector for the query.")
            return None

        query_results = index.query(vector=embedding_vector, top_k=top_k, include_metadata=True)
        vector_ids = [match['id'] for match in query_results['matches']]
        texts = [match['metadata']['text_chunk'] for match in query_results['matches'] if 'text_chunk' in match['metadata']]
        embeddings = [match['values'] for match in query_results['matches']]

        # Clustering
        representative_texts = perform_clustering(embeddings, num_clusters)
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
        st.write(response_text)
    else:
        st.write("Please enter a question to analyze.")


