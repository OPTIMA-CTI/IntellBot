import time
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.retrievers import  EnsembleRetriever

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
    # vectorstore1=FAISS.load_local("CVE_DATA",embeddings,allow_dangerous_deserialization=True)
    vectorstore2=FAISS.load_local("HTML_DATA",embeddings,allow_dangerous_deserialization=True)
    vectorstore3=FAISS.load_local("APT_DATA",embeddings,allow_dangerous_deserialization=True)
    vectorstore4=FAISS.load_local("URL_DATA",embeddings,allow_dangerous_deserialization=True)
    vectorstore5=FAISS.load_local("JSON_DATA",embeddings,allow_dangerous_deserialization=True)
    # vectorstore6=FAISS.load_local("blog-vector",embeddings,allow_dangerous_deserialization=True) #.pkl was unable to download
    vectorstore7=FAISS.load_local("Malwarebytes",embeddings,allow_dangerous_deserialization=True)
    # retriever1=vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retriever2=vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retriever3=vectorstore3.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retriever4=vectorstore4.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retriever5=vectorstore5.as_retriever(search_type="similarity", search_kwargs={"k":3})
    # retriever6=vectorstore6.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retriever7=vectorstore7.as_retriever(search_type="similarity", search_kwargs={"k":3})
    ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever2,retriever3,retriever4,retriever5,retriever7],weights=[0.8] * 5  #the weights schould be assigned according to the no of retrevers
    )
    return ensemble_retriever

if 'retriever' not in st.session_state:
    st.session_state.retriever = get_vectorstore()

def rqa(ensemble_retriever):
    retriever = ensemble_retriever
    llms = ChatOpenAI(openai_api_key="PASTE YOUR API KEY")
    rqa=RetrievalQA.from_chain_type(llm=llms,retriever=retriever,return_source_documents=True)
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
    template = """You are a cyber security expert.Provide the accurate responses for the Question considering the context below."
    Context: {page_content1},{page_content2},{page_content3}.
    Question: {query}
    Answer: """
    prompt_template = PromptTemplate(
    input_variables=["page_content1","page_content2","page_content3","query"],
    template=template
    )
    openai=OpenAI(openai_api_key="PASTE YOUR API KEY")
    accurate_result=openai(prompt_template.format(
        query=query,
        page_content1=data1,
        page_content2=data2,
        page_content3=data3

    ))
    return accurate_result

def source(data1,data2,data3):
    with st.expander("Source"):
        st.write("source:",data1)

def load_vector():
    retriever1=st.session_state.retriever
    rqa1=rqa(retriever1)
    return rqa1

    
def your_code_here(rq1):
      load_dotenv()
      st.header("Security Chatbot🤖")
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
        data1,data2,data3,data4,data5,data6=question(user_prompt,rq1)
        acc_ans=templates(data1,data2,data3,user_prompt)
        response = acc_ans
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
        if response:
            source(data4,data5,data6)

def main():
    rqa3=load_vector()
    while rqa3:
        start_time = time.perf_counter()
        your_code_here(rqa3)
        end_time = time.perf_counter()
        break
    latency = end_time - start_time
    print(f"The code took {latency}seconds to run.")

if __name__ == '__main__':
    main()