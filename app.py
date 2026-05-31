import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
# Step 1: Set user agent (important)
os.environ["USER_AGENT"] = "Mozilla/5.0"

with st.sidebar:
    st.title("📄 PDF Chatbot")
    st.write("Upload a PDF and ask questions.")

st.title("📄 PDF Chatbot")


uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing PDF...")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM (stable)
    llm = ChatGroq(
        model='llama-3.1-8b-instant',
        api_key=os.getenv("GROQ_API_KEY")
    )

    from langchain.prompts import PromptTemplate
    
    prompt_template = """
    You are a helpful assistant.Answer ONLY from the given context.If answer is not found, say "I don't know".Context:{context}\n Question:{question}"""
    
    PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
    )

    # QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt':PROMPT}
    )
   

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
query = st.chat_input("Ask something about your PDF...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Get response
    response = qa.invoke(query)

    answer = response['result']

    with st.chat_message("assistant"):
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.expander("📚 Sources"):
         for doc in response["source_documents"]:
             st.write(doc.page_content[:300])
