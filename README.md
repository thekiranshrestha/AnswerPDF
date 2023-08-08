# AnswerPDF: Get Answers from PDF Documents
### AnswerPDF is a chatbot application that allows users to get answers from PDF documents using a large language model. 
### This project utilizes Streamlit for the user interface, Langchain for processing text data, and OpenAI LLM model for generating responses.

## Features
- Upload a PDF document to extract text.
- Split the extracted text into smaller chunks for processing.
- Create embeddings using OpenAIEmbeddings and store them using FAISS.
- Search for answers to user queries in the PDF document.
- Display responses generated by an OpenAI language model.

## Technologies Used
+ **Streamlit**: UI framework for creating interactive web applications.
+ **Langchain**: Text processing library for text splitting and embedding.
+ **OpenAI**: Language model for generating responses.
+ **PyPDF2**: Library for working with PDF files.
+ **FAISS**: Library for similarity search and indexing.
+ **dotenv**: Library for loading environment variables.
+ **streamlit_extras**: Additional Streamlit utilities.
