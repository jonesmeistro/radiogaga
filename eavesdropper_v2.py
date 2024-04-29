import os 
import requests
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from sklearn.cluster import KMeans
import json

# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index_name = "index-llm-yt"
index = pc.Index(name=index_name)

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
        'Ocp-Apim-Subscription-Key': os.getenv("GENERATE_RESPONSE_API_KEY"),
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

# Supporting Functions
def get_query_embedding(user_query):
    url = "https://api.example.com/get_embeddings"
    headers = {'Authorization': 'Bearer your_api_key'}
    response = requests.post(url, json={"query": user_query}, headers=headers)
    return response.json()['embedding'] if response.ok else None

def perform_clustering(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters).fit(np.array(embeddings))
    indices = np.argmin(kmeans.transform(np.array(embeddings)), axis=1)
    return [(index, 0) for index in indices]

