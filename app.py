import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from collections import deque
import uuid

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
DOWNLOAD_FOLDER = 'downloads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Global vector store instance
vectorstore = None
rag_chain = None

# Chat memory storage - stores conversation history for each session
chat_memory = {}
MAX_MEMORY_SIZE = 20

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs('vectorstore', exist_ok=True)


def get_session_id():
    """Get or create a session ID for chat memory."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def add_to_memory(session_id, user_message, ai_response):
    """Add conversation to memory with a limit of MAX_MEMORY_SIZE."""
    if session_id not in chat_memory:
        chat_memory[session_id] = deque(maxlen=MAX_MEMORY_SIZE)
    
    chat_memory[session_id].append({
        'user': user_message,
        'ai': ai_response,
        'timestamp': str(os.times())
    })


def get_conversation_context(session_id):
    """Get conversation history for context."""
    if session_id not in chat_memory:
        return ""
    
    context = "PREVIOUS CONVERSATION:\n"
    for entry in chat_memory[session_id]:
        context += f"User: {entry['user']}\n"
        context += f"AI: {entry['ai']}\n\n"
    
    return context


def allowed_file(filename):
    """Checks if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_available_files():
    """Get list of available PDF files with metadata."""
    pdf_files = []
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.pdf'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file_size = os.path.getsize(file_path)
                pdf_files.append({
                    'filename': filename,
                    'size': file_size,
                    'path': file_path
                })
    return pdf_files


def create_pdf_summary(content, filename):
    """Create a PDF file with summary content."""
    output_path = os.path.join(DOWNLOAD_FOLDER, f"summary_{filename}")
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(f"Summary of {filename}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Content
    content_para = Paragraph(content, styles['Normal'])
    story.append(content_para)
    
    doc.build(story)
    return output_path


def extract_file_operation(response_text):
    """Extract file operations from AI response."""
    operations = {
        'download_original': False,
        'download_summary': False,
        'filename': None,
        'summary_content': None
    }
    
    lines = response_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('DOWNLOAD_ORIGINAL:'):
            operations['download_original'] = line.split(':', 1)[1].strip().lower() == 'true'
        elif line.startswith('DOWNLOAD_SUMMARY:'):
            operations['download_summary'] = line.split(':', 1)[1].strip().lower() == 'true'
        elif line.startswith('FILENAME:'):
            operations['filename'] = line.split(':', 1)[1].strip()
        elif line.startswith('SUMMARY_CONTENT:'):
            current_section = 'summary'
            operations['summary_content'] = line.split(':', 1)[1].strip()
        elif current_section == 'summary' and line and not line.startswith('ANSWER:'):
            operations['summary_content'] += ' ' + line
        elif line.startswith('ANSWER:'):
            current_section = 'answer'
            
    return operations


def format_response_with_links(clean_answer, download_links):
    """Add download links to the AI response for display in chat."""
    formatted_response = clean_answer
    
    if download_links:
        formatted_response += "\n\n📎 **Download Links:**\n"
        
        if 'original' in download_links:
            filename = download_links['original'].split('/')[-1]
            formatted_response += f"• [📄 Download Original PDF: {filename}]({download_links['original']})\n"
        
        if 'summary' in download_links:
            filename = download_links['summary'].split('/')[-1]
            formatted_response += f"• [📋 Download Summary PDF: {filename}]({download_links['summary']})\n"
    
    return formatted_response


def get_vectorstore():
    """
    Processes all PDF documents in the 'uploads' folder and creates
    or updates a Chroma vector store with file metadata.
    """
    global vectorstore
    if vectorstore is not None:
        print("Vector store already exists. Not rebuilding.")
        return vectorstore

    print("Building vector store from documents in the 'uploads' directory...")

    # Check for existing PDFs in the uploads folder
    pdf_files = get_available_files()
    if not pdf_files:
        print("No PDF files found in the 'uploads' directory. Skipping vector store creation.")
        return None

    # Load documents from the uploads folder with metadata
    documents = []
    for file_info in pdf_files:
        file_path = file_info['path']
        filename = file_info['filename']
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Add filename metadata to each document chunk
        for doc in docs:
            doc.metadata['source_filename'] = filename
            doc.metadata['file_path'] = file_path
        
        documents.extend(docs)

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
    files = get_available_files()
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
    files = get_available_files()
    session_id = get_session_id()
    return render_template('chat.html', files=files, session_id=session_id)


@app.route('/download/<filename>')
def download_file(filename):
    """Download original PDF file."""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-summary/<filename>')
def download_summary(filename):
    """Download generated summary PDF."""
    try:
        summary_path = os.path.join(DOWNLOAD_FOLDER, filename)
        if os.path.exists(summary_path):
            return send_file(summary_path, as_attachment=True)
        else:
            return jsonify({'error': 'Summary file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear-memory', methods=['POST'])
def clear_memory():
    """Clear chat memory for current session."""
    session_id = get_session_id()
    if session_id in chat_memory:
        del chat_memory[session_id]
    return jsonify({'status': 'Memory cleared successfully'})


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Handles the user's question and provides a RAG-based answer with file operations.
    """
    global rag_chain, vectorstore
    question = request.json.get('question', '')
    session_id = get_session_id()

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

        # Get available files for the prompt
        available_files = get_available_files()
        file_list = [f"- {f['filename']}" for f in available_files]
        files_context = "\n".join(file_list)

        # Enhanced prompt template with file operations and memory
        prompt_template = PromptTemplate.from_template(
            f"""You are an intelligent document assistant with access to multiple PDF files and conversation memory.

AVAILABLE FILES:
{files_context}

CAPABILITIES:
1. Answer questions about specific files or all files
2. Provide file downloads when requested
3. Generate and provide PDF summaries when requested
4. Remember previous conversation context

INSTRUCTIONS:
- Use conversation history to understand context and references
- Always identify which specific file(s) contain the relevant information
- If user asks about a specific file, focus your search on that file
- If user wants to download a file, set DOWNLOAD_ORIGINAL: true
- If user wants a summary as PDF, set DOWNLOAD_SUMMARY: true and provide summary content
- Always mention the source filename in your answer
- Refer to previous conversation when relevant

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

DOWNLOAD_ORIGINAL: [true/false]
DOWNLOAD_SUMMARY: [true/false] 
FILENAME: [exact filename if download/summary requested]
SUMMARY_CONTENT: [detailed summary content if DOWNLOAD_SUMMARY is true]

ANSWER: [Your detailed answer here, always mentioning which file(s) the information comes from]

{{conversation_context}}

CONTEXT FROM DOCUMENTS:
{{context}}

USER QUESTION: {{input}}
"""
        )

        # Create the retrieval chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Get conversation context
    conversation_context = get_conversation_context(session_id)
    
    # Use the chain to get the final answer
    try:
        # Include conversation context in the input
        full_input = {
            "input": question,
            "conversation_context": conversation_context
        }
        
        response = rag_chain.invoke(full_input)
        ai_response = response.get("answer", "Sorry, I could not find a relevant answer in your documents.")
        
        # Extract file operations from response
        operations = extract_file_operation(ai_response)
        
        # Extract the actual answer (everything after ANSWER:)
        answer_parts = ai_response.split('ANSWER:', 1)
        clean_answer = answer_parts[1].strip() if len(answer_parts) > 1 else ai_response
        
        # Handle file operations
        download_links = {}
        
        if operations['download_original'] and operations['filename']:
            file_path = os.path.join(UPLOAD_FOLDER, operations['filename'])
            if os.path.exists(file_path):
                download_links['original'] = f"/download/{operations['filename']}"
        
        if operations['download_summary'] and operations['filename'] and operations['summary_content']:
            try:
                summary_path = create_pdf_summary(
                    operations['summary_content'], 
                    operations['filename']
                )
                summary_filename = os.path.basename(summary_path)
                download_links['summary'] = f"/download-summary/{summary_filename}"
            except Exception as e:
                print(f"Error creating summary PDF: {e}")
        
        # Format response with download links for display
        formatted_response = format_response_with_links(clean_answer, download_links)
        
        # Add to conversation memory (store clean answer without links for context)
        add_to_memory(session_id, question, clean_answer)
        
        return jsonify({
            'response': formatted_response,  # This includes download links
            'downloads': download_links,     # For programmatic access
            'operations': operations,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        error_response = "Sorry, an error occurred while processing your request."
        add_to_memory(session_id, question, error_response)
        return jsonify({'response': error_response})


if __name__ == '__main__':
    # Initialize the vector store on startup
    get_vectorstore()
    app.run(debug=True, port=5050)