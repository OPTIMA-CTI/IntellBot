import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download
# from langchain.llms import Llamacpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.docstore.document import Document
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# import torchvision

def get_vectorstore():
    embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore1=FAISS.load_local("newfaiss",embeddings)
    vectorstore2=FAISS.load_local("tryparu",embeddings)
    vectorstore3=FAISS.load_local("binu",embeddings)
    vectorstore4=FAISS.load_local("navyabiju",embeddings)
    vectorstore5=FAISS.load_local("arunima",embeddings)
    vectorstore1.merge_from(vectorstore2)
    vectorstore1.merge_from(vectorstore3)
    vectorstore1.merge_from(vectorstore4)
    vectorstore1.merge_from(vectorstore5)
    return vectorstore1

def get_conversation_chain(vectorstore,TEMP):
    llm = ChatOpenAI(openai_api_key="PASTE YOUR API KEY",temperature=TEMP,model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        chain_type="stuff",
        # return_source_document=True,
        get_chat_history=lambda h: h
        )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # with st.container():
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

            
def sidebar():
    global TEMP
    # global MODEL
    with st.sidebar:
        # MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','mistralai/Mistral-7B-v0.1'])
        TEMP = st.slider("Temperature",0.0,1.0,0.5)
        if TEMP:
            vectorstore = get_vectorstore()
            st.session_state.conversation = get_conversation_chain(vectorstore,TEMP)

def main():
    load_dotenv()
    st.set_page_config(page_title=chr(0x1F916) +"Bot")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    sidebar()

    st.header("🤖 SECURITY CHATBOT")
    user_question = st.text_input("Ask your Questions:")
    if user_question:
         handle_userinput(user_question)
if __name__ == '__main__':
    main()
