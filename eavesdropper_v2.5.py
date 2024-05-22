import streamlit as st
import requests
import json
from datetime import datetime
from pinecone import Pinecone


embed_api_key = "717a41bf61fc4198b9e7c8cc668bddfb"
generate_response_api_key = "c4f949461bee42fa968fcd58f1ec2f41"
pinecone_api_key = "2af4a324-c766-4bd9-a523-b5457187fcc1"




# Initialize Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "interior-design-index"
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
        "It's your job to help me find trends, insights and hot topics that people may find interesting regarding particular topics. I am going to feed you some information and your response will be based on this",
    "I am a PR specialist and I want to coin new trends to become an authoritative industry voice":
        "I am a public relations professional and I want to coin new trends, to create new trend terms. For example 'quiet luxury' is a new trend, and it is catchy and it basically means minimalist luxury. I want to come up with new trend terms based on information that I am sending, help me do this by reviewing the information explaining why it could be considered a trend."
}

st.title('Trend Analysis Tool')
selected_system_message = st.selectbox("Choose your role:", list(system_messages.keys()))
top_k = st.slider("Select the number of top responses (TOPK):", min_value=1, max_value=20, value=5)
user_input = st.text_input("What would you like to know?")

start_date = st.date_input("Start Date:")
end_date = st.date_input("End Date:")

def generate_embedding(user_query):
    headers = {
        'x-service-line': 'creative',
        'x-brand': 'carat',
        'x-project': 'CraigJonesProject',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': embed_api_key
    }
    data = {
        "input": user_query,
        "user": "streamlit_user",
        "input_type": "query"
    }
    response = requests.post("https://ai-api-dev.dentsu.com/openai/deployments/TextEmbeddingAda3Large/embeddings?api-version=2024-02-01", headers=headers, json=data)
    response_data = response.json()

    if 'data' in response_data and 'embedding' in response_data['data'][0]:
        return response_data['data'][0]['embedding']
    else:
        return None

def generate_response_with_gpt3(responses, metadata):
    url = "https://ai-api-dev.dentsu.com/openai/deployments/GPT4o128k/chat/completions?api-version=2024-02-01"
    headers = {
        'x-service-line': 'Creative',
        'x-brand': 'carat',
        'x-project': 'data-journalist-radio',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': 'v8',
        'Ocp-Apim-Subscription-Key': generate_response_api_key,
    }

    # Prepare the detailed content with metadata for each transcript
    content = "\n\n".join([
        f"Transcript: {resp['transcript_text']}\nVideo Title: {resp['video_title']}\nChannel Name: {resp['channel_name']}\nDate: {resp['date']}"
        for resp in responses
    ])

    messages = [
        {"role": "system", "content": system_messages[selected_system_message]},
        {"role": "user", "content": "Based on the following information, please unpick some hot topics or trends which are hinted at while thoroughly explaining the information given. Also, please list all sources that have been used to construct this answer, noting the video title, channel name, and date for each content you are fed:\n\n" + content}
    ]

    data = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1000  # Adjust this based on your needs
    }

    response = requests.post(url, headers=headers, json=data)
    gpt_response = response.json()
    response_text = gpt_response.get('choices', [{}])[0].get('message', {}).get('content', "Failed to generate response.")

    source_info = "Sources used: " + ", ".join([f"Video ID: {meta['id']}, Video Title: {meta['video_title']}, Channel Name: {meta['channel_name']}, Date: {meta['date']}" for meta in metadata])
    return response_text + "\n\n" + source_info

def process_user_query(user_query, top_k, start_date, end_date):
    embedding_vector = generate_embedding(user_query)
    if embedding_vector is None:
        return "Failed to retrieve embedding vector."

    # Convert dates to numerical format
    start_date_num = int(start_date.strftime('%Y%m%d'))
    end_date_num = int(end_date.strftime('%Y%m%d'))

    # Create a filter for the date range
    date_filter = {
        "date": {
            "$gte": start_date_num,
            "$lte": end_date_num
        }
    }

    try:
        query_results = index.query(vector=embedding_vector, top_k=top_k, include_metadata=True, filter=date_filter)
        responses = [
            {
                'transcript_text': match['metadata']['transcript_text'],
                'video_title': match['metadata']['video_title'],
                'channel_name': match['metadata']['channel_name'],
                'date': match['metadata']['date']
            }
            for match in query_results['matches'] if 'transcript_text' in match['metadata']
        ]

        if not responses:
            return "No valid transcript_text metadata found in query results."

        metadata = [
            {
                'id': match['id'],
                'video_title': match['metadata']['video_title'],
                'channel_name': match['metadata']['channel_name'],
                'date': match['metadata']['date']
            }
            for match in query_results['matches']
        ]
        return generate_response_with_gpt3(responses, metadata)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

if st.button('Analyze'):
    response_text = process_user_query(user_input, top_k, start_date, end_date)
    st.write(response_text if response_text else "No response generated.")
