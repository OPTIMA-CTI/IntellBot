import streamlit as st
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from htmlTemplates import css, bot_template, user_template
# from textFunctions import get_pdf_text, get_pdfs_text, get_text_chunks
# from vizFunctions import roberta_barchat, vaders_barchart
# from prompts import set_prompt


def init_ses_states():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

def get_vectorstore():
    embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore=FAISS.load_local("newfaiss", embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, temp, model):
    llm = ChatOpenAI(openai_api_key="PASTE_YOUR_KEY",temperature=temp, model_name=model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': (user_question)})
    st.session_state.chat_history = response['chat_history']
    with st.spinner('Generating response...'):
        display_convo()
        

def display_convo():
    with st.container():
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def sidebar():
    global MODEL
    global PERSONALITY
    global TEMP
    global pdf_docs
    with st.sidebar:
        with st.expander("Chat Bot Settings", expanded=True):
            MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo'])
            PERSONALITY = st.selectbox(label='Personality', options=['general assistant','academic','witty'])
            TEMP = st.slider("Temperature",0.0,1.0,0.5)
            st.button("enter")
            if st.button:
                vectorstore=get_vectorstore()
                st.session_state.conversation = get_conversation_chain(vectorstore, temp=TEMP, model=MODEL)
        # pdf_analytics_settings()
        # with st.expander("Your Documents", expanded=True):
        #     pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        #     if st.button("Process Files + New Chat"):
        #         if pdf_docs:
        #             with st.spinner("Processing"):
        #                 process_docs(pdf_docs, TEMP, MODEL)
        #         else: 
        #             st.caption("Please Upload At Least 1 PDF")
        #             st.session_state.pdf_processed = False


def main():
    load_dotenv()
    st.set_page_config(page_title="🤖BOT", page_icon=":bot:")
    st.write(css, unsafe_allow_html=True)
    init_ses_states()
    sidebar()
    st.header("🤖 SECURITY CHATBOT")
    user_question = st.chat_input("Ask your Questions:")
    if user_question:
         handle_userinput(user_question)

if __name__ == '__main__':
    main()

