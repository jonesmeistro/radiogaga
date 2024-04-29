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

def fetch_vectors(vector_ids):
    # Fetch vectors by IDs from the initialized index
    response = index.fetch(ids=vector_ids)
    return response


def get_embeddings_and_texts(user_query, top_k):
    try:
        embedding_vector = get_query_embedding(user_query)
        if not embedding_vector:
            print("Failed to generate embedding vector for the query.")
            return [], []

        query_results = index.query(vector=embedding_vector, top_k=top_k)
        vector_ids = [match['id'] for match in query_results['matches']]

        fetch_response = fetch_vectors(vector_ids)
        if 'vectors' in fetch_response:
            embeddings = [vec['values'] for vec_id, vec in fetch_response['vectors'].items()]
            texts = [vec['metadata']['text_chunk'] for vec_id, vec in fetch_response['vectors'].items() if 'text_chunk' in vec['metadata']]
        else:
            print("Fetch response did not contain 'vectors'.")
            return [], []

        return embeddings, texts
    except Exception as e:
        print(f"Failed to fetch data: {str(e)}")
        return [], []


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

def perform_clustering(embeddings, num_clusters=num_clusters):
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
def generate_response_with_gpt3(responses):
    url = "https://ai-api-dev.dentsu.com/openai/deployments/GPT4-8K/chat/completions?api-version=2024-02-01"
    headers = {
        'x-service-line': 'Creative',
        'x-brand': 'carat',
        'x-project': 'data-journalist-radio',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': generate_response_api_key,  # Use your actual subscription key
    }

    # Create messages for the chat history context
    messages = [
        {"role": "system", "content": "It's your job to help me find trends, insights and hot topics that people may find interesting regarding particular topics.  I am going to feed you some information and your response will be based on this"},
        {"role": "user", "content": "Based on the following information, please unpick some hot topics or trends which are hinted at while thoroughly explaining the information given:"}
    ] + [{"role": "user", "content": text} for text in responses]

    data = {
        "model": "gpt-4-8k",  # Use appropriate model name if different
        "messages": messages,
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


    response = requests.post(url, headers=headers, json=data)
    return response.json()

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


