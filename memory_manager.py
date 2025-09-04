import os
from collections import deque
from config import Config


class MemoryManager:
    """Manages chat memory for different sessions."""
    
    def __init__(self):
        self.chat_memory = {}
        self.max_memory_size = Config.MAX_MEMORY_SIZE
    
    def add_to_memory(self, session_id, user_message, ai_response):
        """Add conversation to memory with a limit of MAX_MEMORY_SIZE."""
        if session_id not in self.chat_memory:
            self.chat_memory[session_id] = deque(maxlen=self.max_memory_size)
        
        self.chat_memory[session_id].append({
            'user': user_message,
            'ai': ai_response,
            'timestamp': str(os.times())
        })
    
    def get_conversation_context(self, session_id):
        """Get conversation history for context."""
        if session_id not in self.chat_memory:
            return ""
        
        context = "PREVIOUS CONVERSATION:\n"
        for entry in self.chat_memory[session_id]:
            context += f"User: {entry['user']}\n"
            context += f"AI: {entry['ai']}\n\n"
        
        return context
    
    def clear_memory(self, session_id):
        """Clear chat memory for a specific session."""
        if session_id in self.chat_memory:
            del self.chat_memory[session_id]
    
    def get_memory_stats(self, session_id):
        """Get memory statistics for a session."""
        if session_id not in self.chat_memory:
            return {'count': 0, 'max_size': self.max_memory_size}
        
        return {
            'count': len(self.chat_memory[session_id]),
            'max_size': self.max_memory_size
        }
    
    def get_all_sessions(self):
        """Get all active session IDs."""
        return list(self.chat_memory.keys())