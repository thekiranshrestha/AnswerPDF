import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

with st.sidebar:
    st.title('AnswerPDF: Get any answer from pdf document')
    st.markdown('''
    ## Chat with your PDF document
    LLM Powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchained.com)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    ''')

    add_vertical_space(5)
    st.write('Created by: [Kiran Shrestha](https://www.linkedin.com/in/thekiranstha/)')
