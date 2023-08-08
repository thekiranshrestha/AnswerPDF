import os
import pickle

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain import FAISS
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


def main():
    st.header("AnswerPDF: Get any answer from pdf document")

    # loading openai api key
    load_dotenv()

    # upload pdf file
    pdf = st.file_uploader("Upload your PDF document", type="pdf")
    # st.write(pdf.name)
    # reading pdf document
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        # extracting text from pdf document
        for page in pdf_reader.pages:
            text += page.extract_text()

        # setup chunks sizes and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # splitting text into chunks
        chunks = text_splitter.split_text(text=text)

        # getting pdf name
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        # checking if the file already exist in the directory
        if os.path.exists(f'{store_name}.pkl'):
            # loading previously pickled data from the local directory
            with open(f"{store_name}.pkl", "rb") as file:
                VectorStore = pickle.load(file)
            # st.write("Embeddings loaded from local directory!")
        else:
            # embedding file using OpenAIEmbeddings
            embedding = OpenAIEmbeddings()

            # creating a vectore store using a embedding text of the file using FAISS model
            VectorStore = FAISS.from_texts(chunks, embedding=embedding)
            # storing embedding data into the local directory
            with open(f"{store_name}.pkl", "wb") as file:
                pickle.dump(VectorStore, file)
        # st.write("Embedding completed successfully!")

        # input user query
        query = st.text_input("Ask question for your PDF documents: ")

        if query:
            result = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(temperature=0, )
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=result, question=query)
            st.write(response)


if __name__ == '__main__':
    main()
