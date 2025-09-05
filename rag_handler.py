from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from config import Config
from utils import get_available_files


class RAGHandler:
    """Handles RAG (Retrieval-Augmented Generation) operations."""
    
    def __init__(self):
        self.vectorstore = None
        self.rag_chain = None
        self.embeddings = None
        self.llm = None
    
    def reset_vectorstore(self):
        """Reset the vector store to force rebuild."""
        self.vectorstore = None
        self.rag_chain = None
    
    def get_vectorstore(self):
        """
        Processes all PDF documents in the 'uploads' folder and creates
        or updates a Chroma vector store with file metadata.
        """
        if self.vectorstore is not None:
            print("Vector store already exists. Not rebuilding.")
            return self.vectorstore

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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)

        # Create embeddings and the vector store
        if not self.embeddings:
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        self.vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings)

        print("Vector store created successfully.")
        return self.vectorstore
    
    def create_rag_chain(self):
        """Create the RAG chain with prompt template."""
        if self.rag_chain is not None:
            return self.rag_chain
        
        if not self.llm:
            self.llm = ChatDeepSeek(model=Config.LLM_MODEL, api_key=Config.DEEPSEEK_API_KEY)

        # Create a retriever from the vector store
        retriever = self.vectorstore.as_retriever()

        # Get available files for the prompt
        available_files = get_available_files()
        file_list = [f"- {f['filename']}" for f in available_files]
        files_context = "\n".join(file_list)

        # Enhanced prompt template with file operations and memory
        prompt_template = PromptTemplate.from_template(
            f"""You are PDF-IQ, an intelligent document assistant with access to multiple PDF files and conversation memory.

AVAILABLE FILES:
{files_context}

CAPABILITIES:
1. Answer questions about specific files or all files
2. Provide file downloads when requested
3. Generate and provide well-structured PDF summaries when requested
4. Remember previous conversation context

SUMMARY FORMATTING GUIDELINES (when creating summaries):
- Use clear headings for main sections (e.g., "# Overview", "# Key Points", "# Conclusion")
- Use subheadings for subsections (e.g., "## Main Findings", "## Methodology")
- Use bullet points for lists (start with - or •)
- Use numbered lists for sequential items (1. First item, 2. Second item)
- Use **bold text** for emphasis
- Use *italic text* for definitions or important terms
- Keep paragraphs concise and well-structured
- Add blank lines between sections for better readability

INSTRUCTIONS:
- Use conversation history to understand context and references
- Always identify which specific file(s) contain the relevant information
- If user asks about a specific file, focus your search on that file
- If user wants to download a file, set DOWNLOAD_ORIGINAL: true
- If user wants a summary as PDF, set DOWNLOAD_SUMMARY: true and provide well-structured summary content
- Always mention the source filename in your answer
- Refer to previous conversation when relevant

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

DOWNLOAD_ORIGINAL: [true/false]
DOWNLOAD_SUMMARY: [true/false] 
FILENAME: [exact filename if download/summary requested]
SUMMARY_CONTENT: [well-structured summary with proper headings, bullet points, and formatting when DOWNLOAD_SUMMARY is true]

ANSWER: [Your detailed answer here, always mentioning which file(s) the information comes from]

{{conversation_context}}

CONTEXT FROM DOCUMENTS:
{{context}}

USER QUESTION: {{input}}
"""
        )

        # Create the retrieval chain
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt_template)
        self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        return self.rag_chain
    
    def initialize(self):
        """Initialize the RAG system."""
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            return False
        
        self.create_rag_chain()
        return True
    
    def get_response(self, question, conversation_context=""):
        """Get response from the RAG system."""
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")
        
        # Include conversation context in the input
        full_input = {
            "input": question,
            "conversation_context": conversation_context
        }
        
        response = self.rag_chain.invoke(full_input)
        return response.get("answer", "Sorry, I could not find a relevant answer in your documents.")