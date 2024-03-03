# -*- coding: utf-8 -*-
"""2024 Steamlit App.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bSlZMbG0yAgu3nJATf9GNoQWu9t8xhD4
"""

import openai
import os 
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

# Initialize API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Pinecone setup
pc = Pinecone(api_key=pinecone_api_key)
index_name = "index-llm-yt"

# Function to generate response with GPT-3
def generate_response_with_gpt3(responses):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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

# Function to process user query
def process_user_query(user_query):
    # Generate an embedding for the query
    embedding_response = openai.Embedding.create(model="text-embedding-ada-002", input=user_query)
    embedding_vector = embedding_response['data'][0]['embedding']

    # Query the Pinecone index
    query_results = index.query(vector=embedding_vector, top_k=5, include_metadata=True)
    responses = [match['metadata']['text_chunk'] for match in query_results['matches'] if 'metadata' in match and 'text_chunk' in match['metadata']]

    # Generate a response using GPT-3 with the retrieved texts
    return generate_response_with_gpt3(responses)

# Streamlit app layout
st.title('Chat Model App')

user_input = st.text_input("What would you like to know?", "")

if st.button('Ask'):
    if user_input:
        response_text = process_user_query(user_input)
        st.write(response_text)
    else:
        st.write("Please enter a question.")
