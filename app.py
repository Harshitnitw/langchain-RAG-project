from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
import streamlit as st

st.title("RAG app demo")

persist_directory='db'


if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=MistralAIEmbeddings())
    # print("loaded vectorstore from disk")
else:
    urls = ['https://www.victoriaonmove.com.au/local-removalists.html','https://victoriaonmove.com.au/index.html']
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000)
    docs=text_splitter.split_documents(data)
    import chromadb

    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore=Chroma.from_documents(documents=docs,embedding=MistralAIEmbeddings(),persist_directory=persist_directory)
    # print("created fresh vectorstore")


retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm = ChatMistralAI(
model="mistral-small-latest",
temperature=0.4,
max_tokens=500
)

query=st.chat_input("Ask me anything:")

system_prompt=(
    """
    You are an assistant for  question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that you don't know.
    Use 3 sentences maximum and keep answer concise.
    \n\n
    {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)
if query:
    question_answer_chain=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(retriever,question_answer_chain)
    response=rag_chain.invoke({"input":query})
    st.write(response["answer"])