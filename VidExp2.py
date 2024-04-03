
from langchain_community.document_loaders import YoutubeLoader
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

genai.configure(
    api_key="AIzaSyDwrLsUBYvq78CrymNFLMthcWqhz2FJdIo"
)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

st.title('VidExp- Video Expert Chatbot')
st.write('Video explanation bot to answer questions regarding any youtube video with a good transcript record')
st.write('Primarily built for E-learning platforms to reduce repeated playbacks of teaching videos')
st.image("pexels-negative-space-97077.jpg")

st.sidebar.info('The bot will activate once you provide the URL of the video')
video = st.sidebar.text_input("Enter the video URL here...")
check = st.sidebar.button("Ask VidExp")

if video:
    loader = YoutubeLoader.from_youtube_url(video, add_video_info=True)
    result = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    texts = text_splitter.split_documents((result))

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your Prompt here")
    prompt_template = f"with this data : {result[0]}, answer the questions from the user : {prompt}"
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        response = chat.send_message(prompt_template)
        response = response.text
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
# except:
#     st.error("An error occured. Sorry for the inconvenience.")
#     st.header("We will get back to you ASAP! :sparkles:")

# # while(True):
#     prompt = input("You: ")
#     if(prompt.strip()==''):
#         break
#
#     # response = chat.send_message(f"analyze this document {result[0]} and answer the question - {prompt}")
#     print(f"Bot: {response.text}")

