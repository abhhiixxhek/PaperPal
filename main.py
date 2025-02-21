import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 🚀 Move this line to the top before any Streamlit commands
st.set_page_config(page_title="PaperPal 📄🤖", page_icon="📄", layout="wide")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create conversational chain for question answering."""
    prompt_template = """
    You are an intelligent AI assistant designed to answer questions based on the provided PDF content.  
    Your responses should be **accurate, informative, and strictly based on the given context**.  

    - **If the answer is fully available in the context, provide a well-structured, detailed, and clear response.**  
    - **If the answer is partially available, state what is known and mention any missing details.**  
    - **If the question is entirely out of scope and cannot be answered based on the given context, respond with: "The answer is not available in the provided context." Do not make up information.**  
    - **Keep responses concise yet comprehensive, avoiding unnecessary speculation.**  
    - **Use bullet points or numbered lists for clarity when needed.**   

    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process user question against uploaded PDF context."""
    try:
        # Verify embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)

        # Perform similarity search
        docs = new_db.similarity_search(user_question)

        # Get conversational chain
        chain = get_conversational_chain()

        # Get response
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )

        # Display response
        st.write("Reply: ", response["output_text"])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please upload PDFs and process them first.")

def main():
    # Custom header with emoji
    st.markdown("<h1 style='text-align: center;'>📄 PaperPal – Chat with Your PDFs 💡</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Your AI-powered research assistant to extract insights effortlessly! 🚀</h4>", unsafe_allow_html=True)

    # Sidebar for PDF upload
    with st.sidebar:
        st.markdown("## 📂 Upload PDFs")
        pdf_docs = st.file_uploader("Drop your PDFs here!", type=['pdf'], accept_multiple_files=True)

        if st.button("🚀 Process PDFs"):
            if pdf_docs:
                with st.spinner("🔍 Analyzing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)  # Extract text
                    text_chunks = get_text_chunks(raw_text)  # Chunk text
                    get_vector_store(text_chunks)  # Store in vector DB
                    st.success("✅ PDFs processed successfully! Ask away. 🎯")
            else:
                st.warning("⚠️ Please upload at least one PDF first.")

    # Main chat interface
    st.markdown("---")
    st.markdown("### 🤖 Ask PaperPal Anything About Your PDFs!")
    user_question = st.text_input("🔎 Type your question here...")

    if user_question:
        st.markdown("📝 **Your Question:** " + user_question)
        user_input(user_question)

    # Footer with credits
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>🚀 Made with ❤️ by <b>Abhishek Kumar</b></p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
