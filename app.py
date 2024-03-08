import subprocess
import sys

# Install required packages using pip
required_packages = [
    'streamlit',
    'streamlit_chat',
    'langchain',
    'openai',
    'faiss-cpu',
    'tiktoken'
]
for package in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import os

def main():
    st.markdown(
            """
            <h2 style='text-align: center;'> Take Flight with Precision and Revolutionize Your Aircraft Assignments! ✈️</h2>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
            """
            <div style='text-align: center;'>
                <h4 style='text-align: center;'>Streamline Aircraft Assignments and Elevate Efficiency with our Next-Generation Aviation Management Solution</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown(
            """
            <div style='text-align: center;'>
                <h6>Enter the <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API key</a> to start chatting</h6>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="sk-",
    type="password")

    uploaded_file = st.sidebar.file_uploader("upload", type="csv")
    #Contact
    with st.sidebar.expander("📬 Contact"):

        st.write("**LinkedIn:**")
        st.write("[Rama-Alyoubi](https://www.linkedin.com/in/rama-alyoubi/)")
        st.write("**Twitter:**")
        st.write("[Rama-Alyoubi](https://twitter.com/Rama_Alyoubi)")
        st.write("**Mail** :")
        st.write("*Rama Alyoubi* : Rama.mohammed.alyoubi@gmail.com")

    if uploaded_file :
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()
        os.environ["OPENAI_API_KEY"] = "sk-"+ user_api_key
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
                                                                        retriever=vectors.as_retriever())

        def conversational_chat(query):
            
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            
            return result["answer"]
        
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]
            
        #container for the chat history
        response_container = st.container()
        #container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                    

if __name__ == "__main__":
    main()
