import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Core Functions ---

def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore):
    """Create a conversational retrieval chain."""
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')
    
    template = """You are a helpful AI assistant for querying PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and relevant.
    
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={'prompt': prompt}
    )

def process_documents(pdf_docs):
    """Process uploaded PDF documents."""
    try:
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.warning("Could not extract text from the PDF(s). They might be image-based or empty.")
            return

        text_chunks = get_text_chunks(raw_text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True
        st.success("PDFs processed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📚")
    st.title("Chat with your PDF 📚")

    # Check for API key in environment
    api_key_loaded = os.getenv("OPENAI_API_KEY") is not None

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not api_key_loaded:
            st.error("OpenAI API Key not found. Please add it to your .env file.")
        else:
            st.success("API Key loaded successfully.")

        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            type="pdf",
            accept_multiple_files=True
        )
        
        if st.button("Process"):
            if not api_key_loaded:
                st.error("Cannot process without an OpenAI API key.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing your PDFs..."):
                    process_documents(pdf_docs)

    # Display chat history
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(message.content)

    # Main chat interface
    if st.session_state.processComplete:
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Pass the chat history to the chain
                    response = st.session_state.conversation.invoke(
                        {'question': prompt, 'chat_history': st.session_state.messages}
                    )
                    answer = response["answer"]
                    st.write(answer)
                    st.session_state.messages.append(AIMessage(content=answer))
    else:
        st.info("👈 Upload your PDFs in the sidebar to get started!")

if __name__ == '__main__':
    main()
