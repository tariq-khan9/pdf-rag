import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# RAG specific imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global vector store instance
vectorstore = None
rag_chain = None

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('vectorstore', exist_ok=True)


def allowed_file(filename):
    """Checks if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_vectorstore():
    """
    Processes all PDF documents in the 'uploads' folder and creates
    or updates a Chroma vector store.
    """
    global vectorstore
    if vectorstore is not None:
        print("Vector store already exists. Not rebuilding.")
        return vectorstore

    print("Building vector store from documents in the 'uploads' directory...")

    # Check for existing PDFs in the uploads folder
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the 'uploads' directory. Skipping vector store creation.")
        return None

    # Load documents from the uploads folder
    documents = []
    for file in pdf_files:
        file_path = os.path.join(UPLOAD_FOLDER, file)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    if not documents:
        print("No content could be extracted from the PDF files.")
        return None

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and the vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    print("Vector store created successfully.")
    return vectorstore


@app.route('/')
def index():
    """Renders the main page with the file upload form."""
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    return render_template('index.html', files=files)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload and text extraction."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # After a new file is uploaded, reset the vector store to rebuild it
        global vectorstore
        vectorstore = None

        flash(f'File {filename} uploaded successfully! The document will now be processed for RAG.')
        return redirect(url_for('index'))
    else:
        flash('Please upload a PDF file')
        return redirect(url_for('index'))


@app.route('/chat')
def chat():
    """Renders the chat interface."""
    return render_template('chat.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Handles the user's question and provides a RAG-based answer.
    """
    global rag_chain, vectorstore
    question = request.json.get('question', '')

    if not question:
        return jsonify({'response': "Please enter a question."})

    # Ensure the vector store is built before trying to use it
    if vectorstore is None:
        vectorstore = get_vectorstore()

    if vectorstore is None:
        return jsonify({'response': "Please upload and process at least one PDF file first."})

    # Create the RAG chain if it doesn't exist
    if rag_chain is None:
        llm = ChatDeepSeek(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)

        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever()

        # Define the prompt template for the LLM (use {input})
        prompt_template = PromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}

            Question: {input}
            """
        )

        # Create the retrieval chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Use the chain to get the final answer
    try:
        response = rag_chain.invoke({"input": question})
        final_response = response.get("answer", "Sorry, I could not find a relevant answer in your documents.")
        return jsonify({'response': final_response})
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return jsonify({'response': "Sorry, an error occurred while processing your request."})


if __name__ == '__main__':
    # Initialize the vector store on startup
    get_vectorstore()
    app.run(debug=True, port=5050)
