import os
from config import Config


def allowed_file(filename):
    """Checks if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_available_files():
    """Get list of available PDF files with metadata."""
    pdf_files = []
    if os.path.exists(Config.UPLOAD_FOLDER):
        for filename in os.listdir(Config.UPLOAD_FOLDER):
            if filename.endswith('.pdf'):
                file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                file_size = os.path.getsize(file_path)
                pdf_files.append({
                    'filename': filename,
                    'size': file_size,
                    'path': file_path
                })
    return pdf_files


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
    summary_lines = []
    
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
            # Get content after the colon
            content_after_colon = line.split(':', 1)[1].strip()
            if content_after_colon:
                summary_lines.append(content_after_colon)
        elif current_section == 'summary' and line and not line.startswith('ANSWER:'):
            summary_lines.append(line)
        elif line.startswith('ANSWER:'):
            current_section = 'answer'
            break
    
    # Join summary lines with newlines to preserve formatting
    if summary_lines:
        operations['summary_content'] = '\n'.join(summary_lines)
            
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


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"