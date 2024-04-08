# -*- coding: utf-8 -*-
"""2024 Steamlit App.ipynb

Automatically generated by Colaboratory.

Original file is located at
"""

import os 
import requests
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec


# Initialize API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
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


# Function to generate response with GPT-3
def generate_response_with_gpt3(responses):
    messages = [
        {"role": "system", "content": "I am feeding you context from YouTube videos and Comments.  This data is helping you to give me the best possible answer, I am interested in obtaining new and deep insights from this data"},
        {"role": "user", "content": "Based on the following information, provide a summary:"}
    ]

    for text in responses:
        messages.append({"role": "user", "content": text})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    if isinstance(response, openai.openai_object.OpenAIObject):
        return response.get("choices")[0].get("message").get("content")
    else:
        return "Response format unknown: {}".format(response)

def process_user_query(user_query):
    # Generate an embedding for the query
    embedding_response = openai.Embedding.create(model="text-embedding-ada-002", input=user_query)
    embedding_vector = embedding_response['data'][0]['embedding']

    responses = [] # Define responses here to ensure it's accessible later

    try:
        query_results = index.query(vector=embedding_vector, top_k=5, include_metadata=True)
        responses = [match['metadata']['text_chunk'] for match in query_results['matches'] if 'metadata' in match and 'text_chunk' in match['metadata']]
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None

    # Generate a response using GPT-3 with the retrieved texts
    return generate_response_with_gpt3(responses)


# Streamlit app layout
st.title('Welcome to the Eavesdropper')

user_input = st.text_input("What would you like to know?", "")

if st.button('Ask'):
    if user_input:
        response_text = process_user_query(user_input)
        st.write(response_text)
    else:
        st.write("Please enter a question.")
