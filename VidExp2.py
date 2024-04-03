import dotenv
from langchain_community.document_loaders import YoutubeLoader
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from secret_key import GOOGLE_API_KEY
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

import textwrap
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai


def videxp():
    model = genai.GenerativeModel('gemini-pro')

    st.sidebar.header("VidExp Bot")
    st.sidebar.info("Only provide youtube videos or Videos with proper Transcript to get Accurate results")
    st.sidebar.warning(
        "This is purely built for integrating with videos in E-learning websites without and not in any other intentions")
    st.header("VidExp:sparkles::parrot:")

    age = st.sidebar.slider('How do rate your experience here?', 0, 100, 5)
    st.sidebar.write(f"Thank you for you review => {age} || We will try to improvise")

    st.write("No need to replay the Video again - Plug in Video URL and shoot your questions to get accurate answers")
    st.image("pexels-negative-space-97077.jpg")
    st.write("AI generated Image :point_up_2: || Learn and Grow with AI")
    prompt = st.text_input("Enter the video url")
    query = st.text_input("Enter your question")
    # os.environ["OPENAI_API_KEY"] = os.getenv('openai_api_key')
    embeddings = OpenAIEmbeddings()
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        input="Your text string goes here",
        model="text-embedding-3-small"
    )

    print(response.data[0].embedding)

    def create_db_from_youtube_video_url(video_url):
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        db = FAISS.from_documents(docs, response)
        return db

    def get_response_from_query(db, query, k=4):
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])
        question = query
        chat = genai.GenerativeModel('gemini-pro')

        template = """
                You are a helpful assistant that can answer questions about youtube videos
                based on the video's transcripts: {docs}
                Only use the factual information from the transcript to answer the questions
                If you feel like you dont have enough information to answer the question, just say "I dont know"
                Your answers should be verbose and detailed
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        human_template = "Answer the following questions: {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template((human_template))

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        chain = LLMChain(llm=chat, prompt=chat_prompt)

        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        return response, docs

    def transcript_generate(video_url, query):
        db = create_db_from_youtube_video_url(video_url)
        response, docs = get_response_from_query(db, query)
        return response

    if prompt and query:
        video_url = prompt
        db = create_db_from_youtube_video_url(video_url)
        response, docs = get_response_from_query(db, query)
        st.write(textwrap.fill(response, width=85))


genai.configure(
    api_key= GOOGLE_API_KEY
)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

st.title('VidExp- Video Expert Chatbot')
st.write('Video explanation bot to answer questions regarding any youtube video with a good transcript record')
st.write('Primarily built for E-learning platforms to reduce repeated playbacks of teaching videos')

st.sidebar.info('The bot will activate once you provide the URL of the video')
video = st.sidebar.text_input("Enter the video URL here...")
check = st.sidebar.button("Ask VidExp")

if video:
    st.video(video)
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


