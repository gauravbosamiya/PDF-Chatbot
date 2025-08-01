import streamlit as st 
from api_utils import upload_document, list_document, delete_document

def chat_sidebar():
    model_selection = ["compound-beta-mini","llama3-70b-8192"]
    st.sidebar.selectbox("Select Model", options=model_selection, key="model")
    
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choos a file", type=["pdf","docx","html"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully with ID {upload_response['file_id']}.")
                    st.session_state.documents = list_document()
                
                
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document list"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_document()
            
    if "documents" not in st.session_state:
        st.session_state["documents"] = []

    documents = st.session_state.documents
    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']}, Uploaded: {doc['upload_timestamp']})")
            
        selected_file_id = st.sidebar.selectbox("Select a document to delete", options=[doc['id'] for doc in documents],
                                                format_func=lambda x: next(doc['filename'] for doc in documents if doc['id'] == x))

        if st.sidebar.button("Delete selected document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Document with ID {selected_file_id} deleted successfully")
                    st.session_state.documents = list_document()
                else:
                    st.sidebar.error(f"Failed to delete document with ID {selected_file_id}")
                    