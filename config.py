import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Flask application."""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # API Keys
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Directory settings
    UPLOAD_FOLDER = 'uploads'
    DOWNLOAD_FOLDER = 'downloads'
    VECTORSTORE_FOLDER = 'vectorstore'
    
    # File settings
    ALLOWED_EXTENSIONS = {'pdf'}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    
    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_MEMORY_SIZE = 20
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "deepseek-chat"
    
    @staticmethod
    def init_directories():
        """Create necessary directories if they don't exist."""
        directories = [
            Config.UPLOAD_FOLDER,
            Config.DOWNLOAD_FOLDER,
            Config.VECTORSTORE_FOLDER
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Initialize directories when config is imported
Config.init_directories()