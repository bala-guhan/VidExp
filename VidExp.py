from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from secret_key import openai_api_key
import textwrap
import os
import streamlit as st

st.sidebar.info("Only provide youtube videos or Videos with proper Transcript to get Accurate results")
st.sidebar.warning("This is purely built for integrating with videos in E-learning websites without and not in any other intentions")
st.header("Transcript AI:sparkles::parrot:")

age = st.sidebar.slider('How do rate your experience here?', 0, 100, 5)
st.sidebar.write(f"Thank you for you review.{age}. We will try to improvise")


st.write("No need to replay the Video again - Plug in Video URL and shoot your questions to get accurate answers")
st.video("pexels-google-deepmind-18069166 (2160p).mp4")
st.write("AI generated Video :point_up_2: || Learn and Grow with AI")
prompt = st.text_input("Enter the video url")
query = st.text_input("Enter your question")
os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings()



def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    question = query
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

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
