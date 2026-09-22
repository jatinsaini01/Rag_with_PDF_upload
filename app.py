import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain , create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()
#-------------------------------------------------------------------------------------------------------
st.header("RAG with PDF uploads and Chat Hisitory")
api_key = st.text_input("Enter API Key :",type="password")
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")
st.write("Upload PDF and ask question about their content")
if api_key:
    llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=api_key , temperature=0.1)
    session_id = st.text_input("Enter Session ID : ",value="default session")
    if "store" not in st.session_state:
        st.session_state.store = {}
    # loading PDF files
    uploaded_pdfs = st.file_uploader("Upload PDF here : ",type="pdf",accept_multiple_files=True)
    if uploaded_pdfs:
        document = []
        for uploaded_pdf in uploaded_pdfs:
            temppdf = f"./temp.pdf"
            with open(temppdf , "wb") as file:
                file.write(uploaded_pdf.getvalue())
            loader = PyPDFLoader(temppdf)
            pdf = loader.load()
            document.extend(pdf)
        # text splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
        chunked_document = splitter.split_documents(document)
        # embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # vector store
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = Chroma.from_documents(chunked_document, embedding=embeddings)
        # retriever
        retriever = st.session_state.vector_store.as_retriever()
        context_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        context_prompt = ChatPromptTemplate.from_messages([
            ("system",context_system_prompt),
            ("human","{input}"),
            MessagesPlaceholder("chat_history")
        ])
        # retriever with history
        history_aware_retriever = create_history_aware_retriever(llm , retriever , context_prompt)
        # query answer promt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
            "{context}"
        )
        query_answer_prompt = ChatPromptTemplate([
            ("system",system_prompt),
            ("human","{input}"),
            MessagesPlaceholder("chat_history")
        ])
        # chains
        question_answer_chain = create_stuff_documents_chain(llm , query_answer_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.success(f"Assistant : {response['answer']}")
            st.write("Chat History : ",session_history.messages)
else:
    st.warning("please enter API key!")
