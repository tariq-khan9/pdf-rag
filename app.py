import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid

# Import our custom modules
from config import Config
from rag_handler import RAGHandler
from pdf_formatter import create_enhanced_pdf_summary
from utils import allowed_file, get_available_files, get_downloads_files, extract_file_operation, format_response_with_links
from memory_manager import MemoryManager

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components
memory_manager = MemoryManager()
rag_handler = RAGHandler()


def get_session_id():
    """Get or create a session ID for chat memory."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


@app.route('/')
def index():
    """Renders the main page with the file upload form."""
    files = get_available_files()
    return render_template('index.html', files=files)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload and text extraction."""
    print(f"Upload request received. Files: {request.files}")
    print(f"Form data: {request.form}")
    
    if 'file' not in request.files:
        print("No file in request.files")
        flash('No file selected')
        return redirect(url_for('index'))

    file = request.files['file']
    print(f"File object: {file}")
    print(f"File filename: {file.filename}")
    print(f"File content type: {file.content_type}")
    
    if file.filename == '':
        print("Empty filename")
        flash('No file selected')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {file_path}")
        file.save(file_path)

        # Reset the vector store to rebuild it with new file
        rag_handler.reset_vectorstore()

        flash(f'File {filename} uploaded successfully! The document will now be processed for RAG.')
        return redirect(url_for('index'))
    else:
        print(f"File validation failed. File: {file}, allowed_file: {allowed_file(file.filename) if file else 'No file'}")
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
        summary_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
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
    memory_manager.clear_memory(session_id)
    return jsonify({'status': 'Memory cleared successfully'})


@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == Config.ADMIN_USERNAME and password == Config.ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('admin_login.html')


@app.route('/admin')
def admin():
    """Admin page for file management."""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    uploads_files = get_available_files()
    downloads_files = get_downloads_files()
    return render_template('admin.html', uploads_files=uploads_files, downloads_files=downloads_files)


@app.route('/admin-logout')
def admin_logout():
    """Admin logout."""
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))


@app.route('/delete-file', methods=['POST'])
def delete_file():
    """Delete a file from uploads or downloads folder."""
    data = request.json
    folder = data.get('folder')  # 'uploads' or 'downloads'
    filename = data.get('filename')
    
    if not folder or not filename:
        return jsonify({'error': 'Missing folder or filename'}), 400
    
    if folder not in ['uploads', 'downloads']:
        return jsonify({'error': 'Invalid folder'}), 400
    
    try:
        if folder == 'uploads':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Reset vector store when deleting from uploads
            rag_handler.reset_vectorstore()
        else:
            file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'status': 'File deleted successfully'})
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Handles the user's question and provides a RAG-based answer with file operations.
    """
    question = request.json.get('question', '')
    session_id = get_session_id()

    if not question:
        return jsonify({'response': "Please enter a question."})

    # Ensure the vector store is built
    if not rag_handler.initialize():
        return jsonify({'response': "Please upload and process at least one PDF file first."})

    # Get conversation context
    conversation_context = memory_manager.get_conversation_context(session_id)
    
    try:
        # Get response from RAG handler
        ai_response = rag_handler.get_response(question, conversation_context)
        
        # Extract file operations from response
        operations = extract_file_operation(ai_response)
        
        # Extract the actual answer (everything after ANSWER:)
        answer_parts = ai_response.split('ANSWER:', 1)
        clean_answer = answer_parts[1].strip() if len(answer_parts) > 1 else ai_response
        
        # Handle file operations
        download_links = {}
        
        if operations['download_original'] and operations['filename']:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], operations['filename'])
            if os.path.exists(file_path):
                download_links['original'] = f"/download/{operations['filename']}"
        
        if operations['download_summary'] and operations['filename'] and operations['summary_content']:
            try:
                summary_path = create_enhanced_pdf_summary(
                    operations['summary_content'], 
                    operations['filename']
                )
                summary_filename = os.path.basename(summary_path)
                download_links['summary'] = f"/download-summary/{summary_filename}"
            except Exception as e:
                print(f"Error creating summary PDF: {e}")
        
        # Format response with download links for display
        formatted_response = format_response_with_links(clean_answer, download_links)
        
        # Add to conversation memory
        memory_manager.add_to_memory(session_id, question, clean_answer)
        
        return jsonify({
            'response': formatted_response,
            'downloads': download_links,
            'operations': operations,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error during processing: {e}")
        error_response = "Sorry, an error occurred while processing your request."
        memory_manager.add_to_memory(session_id, question, error_response)
        return jsonify({'response': error_response})


if __name__ == '__main__':
    # Initialize the RAG handler on startup (optional - will be initialized when needed)
    try:
        rag_handler.initialize()
        print("RAG handler initialized successfully on startup.")
    except Exception as e:
        print(f"RAG handler initialization failed on startup: {e}")
        print("RAG handler will be initialized when first needed.")
    
    app.run(debug=True, port=5050)