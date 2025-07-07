import streamlit as st 
from api_utils import get_api_response

def chat_interface():
    # ✅ Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if "model" not in st.session_state:
        st.session_state.model = "llama3-70b-8192"  # Default model

    # ✅ Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    # ✅ Handle new user input
    if prompt := st.chat_input("Query..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating Response..."):
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

            if response:
                # Update session ID
                st.session_state.session_id = response.get('session_id')

                # ✅ Append assistant response
                st.session_state.messages.append({'role': 'assistant', 'content': response['answer']})

                # ✅ Display assistant message
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])

                    # Optional expandable details
                    with st.expander("Details"):
                        st.subheader("Generated answer")
                        st.code(response['answer'])
                        st.subheader("Model Used")
                        st.code(response['model'])
                        st.subheader("Session ID")
                        st.code(response['session_id'])

            else:
                st.error("❌ Failed to get a response from the API. Please try again.")
