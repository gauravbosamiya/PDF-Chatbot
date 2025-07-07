from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import BaseRetriever
from chroma_utils import vectorstore
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

retriever = vectorstore.as_retriever(search_kwargs={"k":2})
output_parser = StrOutputParser()

contextualize_q_system_prompt =(
    """
    Given a chat history and the latest user question
    which might reference context in the chat history
    formulate a standalone question which can be understood
    without chat history. DO NOT answer the question
    just reformulate it if needed and otherwise return I do not have permission to give any senseless answer.
    """
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system","Context : {context}"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{input}')
])

from langchain_core.runnables import RunnableLambda

def get_rag_chain(model="llama3-70b-8192"):
    llm = ChatGroq(model=model)
    
    contextualize_chain = contextualize_q_prompt | llm | output_parser
    qa_chain = qa_prompt | llm | output_parser
    
    def rag_chain_func(inputs: dict):
        chat_history = inputs.get("chat_history", [])
        user_input = inputs["input"]

        standalone_question = contextualize_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        docs = retriever.get_relevant_documents(standalone_question)

        answer = qa_chain.invoke({
            "input": user_input,
            "context": "\n\n".join([doc.page_content for doc in docs]),
            "chat_history": chat_history
        })

        return {"answer": answer}
    
    return RunnableLambda(rag_chain_func)


    # history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    # qa_chain = create_stuff_documents_chain(
    #     llm=llm,
    #     prompt=qa_prompt
    # )
    # rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    # return rag_chain