import time
import evaluate
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.docstore.document import Document
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
from bert_score import BERTScorer
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import LlamaCpp
from bert_score import score

def get_vectorstore():
    embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore1=FAISS.load_local("newfaiss",embeddings,allow_dangerous_deserialization=True)
    vectorstore2=FAISS.load_local("parvathi",embeddings,allow_dangerous_deserialization=True)
    vectorstore3=FAISS.load_local("faiss",embeddings,allow_dangerous_deserialization=True)
    vectorstore4=FAISS.load_local("navyabiju",embeddings,allow_dangerous_deserialization=True)
    vectorstore5=FAISS.load_local("arunima",embeddings,allow_dangerous_deserialization=True)
    vectorstore6=FAISS.load_local("blog-vector",embeddings,allow_dangerous_deserialization=True)
    vectorstore7=FAISS.load_local("navyabiju2",embeddings,allow_dangerous_deserialization=True)
    vectorstore1.merge_from(vectorstore2)
    vectorstore1.merge_from(vectorstore3)
    vectorstore1.merge_from(vectorstore4)
    vectorstore1.merge_from(vectorstore5)
    vectorstore1.merge_from(vectorstore6)
    vectorstore1.merge_from(vectorstore7)
    return vectorstore1


def rqa(vectorstore):
    model_path="model\mistral-7b-instruct-v0.1.Q4_0.gguf"
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

    rqa = RetrievalQA.from_chain_type(llm= LlamaCpp(
            model_path=model_path,
            temperature=0,
            max_tokens=10240,
            top_p=1,
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=4096,
            stop=["[INST]"],
            verbose=False,
            streaming=True,
            ),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
    return rqa
    
def question(ques,rqa):
    ans=rqa.invoke(ques)
    data1=ans['source_documents'][0].page_content
    data2=ans['source_documents'][1].page_content
    data3=ans['source_documents'][2].page_content
    data4=ans['source_documents'][0].metadata['source']
    data5=ans['source_documents'][1].metadata['source']
    data6=ans['source_documents'][2].metadata['source']
    return data1,data2,data3,data4,data5,data6

def templates(data1,data2,data3,query):
    template = """[INST]You are a cyber security expert.Provide the accurate responses for the Question considering the context below."
    Context: {page_content1},{page_content2},{page_content3}.
    Question: {query}
    Answer: [INST]"""
    prompt_template = PromptTemplate(
    input_variables=["page_content1","page_content2","page_content3","query"],
    template=template
    )
    openai=OpenAI(openai_api_key="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")
    accurate_result=openai(prompt_template.format(
        query=query,
        page_content1=data1,
        page_content2=data2,
        page_content3=data3

    ))
    return accurate_result

def sidebar(data1,data2,data3):
    with st.sidebar:
        st.write("source:",data1,",",data2,",",data3)

def evaluate1(result,manualanswer):
  scorer = BERTScorer(lang='en')
  reference = [result]
  candidate = [manualanswer]
  P, R, F1 = scorer.score(candidate, reference)
  print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(P.item(), R.item(), F1.item()))

    
def your_code_here():
      load_dotenv()
      st.set_page_config(
        page_title="Security Chatbot!"
    )

      st.header("Security Chatbot🤖")
      vectorstore=get_vectorstore()
      rqa1=rqa(vectorstore)
      if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "How may I help you today?"}
            ]

      if "current_response" not in st.session_state:
        st.session_state.current_response = ""

      for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

      if user_prompt := st.chat_input("Enter your message here", key="user_input"):
        data1,data2,data3,data4,data5,data6=question(user_prompt,rqa1)
        acc_ans=templates(data1,data2,data3,user_prompt)
        response = acc_ans
        if response:
            sidebar(data4,data5,data6)
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant"):
            st.markdown(response)
        result=response
        manualanswer=input("Enter manual answer:")
        evaluate2=evaluate1(result,manualanswer)
        print(evaluate2)
        # result=f1score(evaluate2)
        # print(result)
      for i in range(1000):
       pass  # Simulate some work

def main():
    
    start_time = time.perf_counter()
    your_code_here()
    end_time = time.perf_counter()

    latency = end_time - start_time

    # print(f"The code took {latency}seconds to run.")

    
if __name__ == '__main__':
    main()