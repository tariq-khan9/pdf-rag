#!/usr/bin/python3.4
"""
WSGI configuration for PDF-IQ on PythonAnywhere
"""

import sys
import os

# Add your project directory to the Python path
project_home = '/home/yourusername/pdf-rag'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import your Flask application
from app import app as application

if __name__ == "__main__":
    application.run()

