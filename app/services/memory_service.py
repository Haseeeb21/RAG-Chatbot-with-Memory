import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from app.config import settings

class MemoryService:
    """
    Manages conversation history for each user.
    
    Features:
    - Per-user conversation storage
    - Sliding window memory (keeps last N messages)
    - Persistent storage to disk
    - Context maintenance across sessions
    
    Why Memory?
    - Enables follow-up questions
    - Maintains conversation context
    - Personalizes responses
    """
    
    def __init__(self):
        self.memory_path = Path(settings.memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.max_history = settings.max_conversation_history
    
    def _get_user_file(self, user_id: str) -> Path:
        """Get the file path for a user's conversation history."""
        return self.memory_path / f"{user_id}.json"
    
    def add_interaction(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        retrieved_context: Optional[List[Dict]] = None
    ):
        """
        Add a new interaction to user's conversation history.
        
        Args:
            user_id: Unique user identifier
            user_message: User's query
            assistant_message: AI's response
            retrieved_context: Documents retrieved from RAG
        """
        history = self.get_history(user_id)
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'assistant': assistant_message,
            'retrieved_docs': len(retrieved_context) if retrieved_context else 0
        }
        
        history['messages'].append(interaction)
        
        # Keep only last N messages (sliding window)
        if len(history['messages']) > self.max_history:
            history['messages'] = history['messages'][-self.max_history:]
        
        history['updated_at'] = datetime.now().isoformat()
        
        # Save to disk
        self._save_history(user_id, history)
    
    def get_history(self, user_id: str) -> Dict:
        """
        Retrieve conversation history for a user.
        
        Returns:
            Dictionary with messages and metadata
        """
        user_file = self._get_user_file(user_id)
        
        if user_file.exists():
            with open(user_file, 'r') as f:
                return json.load(f)
        else:
            # Create new history
            return {
                'user_id': user_id,
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
    
    def get_conversation_context(self, user_id: str, n_messages: int = 5) -> str:
        """
        Get formatted conversation context for the LLM.
        
        This helps the AI understand previous conversation flow.
        
        Args:
            user_id: User identifier
            n_messages: Number of recent messages to include
            
        Returns:
            Formatted conversation history as string
        """
        history = self.get_history(user_id)
        messages = history['messages'][-n_messages:]
        
        if not messages:
            return ""
        
        context = "Previous conversation:\n"
        for msg in messages:
            context += f"User: {msg['user']}\n"
            context += f"Assistant: {msg['assistant']}\n\n"
        
        return context
    
    def _save_history(self, user_id: str, history: Dict):
        """Save conversation history to disk."""
        user_file = self._get_user_file(user_id)
        with open(user_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def clear_history(self, user_id: str):
        """Clear conversation history for a user."""
        user_file = self._get_user_file(user_id)
        if user_file.exists():
            user_file.unlink()
            print(f"Cleared history for user: {user_id}")
    
    def get_all_users(self) -> List[str]:
        """Get list of all users with conversation history."""
        return [f.stem for f in self.memory_path.glob('*.json')]