import streamlit as st 
from sidebar import chat_sidebar
from chat_interface import chat_interface

st.title("Langchain RAG Chatbot")

if "message" not in st.session_state:
    st.session_state.messages=[]

if "session_id" not in st.session_state:
    st.session_state.session_id=None
    

chat_sidebar()

chat_interface()