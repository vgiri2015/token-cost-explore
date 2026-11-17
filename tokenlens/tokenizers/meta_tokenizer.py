"""Meta AI tokenizer implementation."""

from typing import List, Optional
import requests
from transformers import AutoTokenizer, LlamaTokenizer
from .base_tokenizer import BaseTokenizer

class MetaTokenizer(BaseTokenizer):
    """Meta AI tokenizer for text encoding and decoding."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-70b-chat-hf", api_key: Optional[str] = None):
        """Initialize Meta tokenizer with model name and API key."""
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        auth_token = {"use_auth_token": api_key} if api_key else {}
        try:
            # Try to use HuggingFace's Llama tokenizer first
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name, **auth_token)
        except Exception as llama_error:
            try:
                # Fall back to AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_token)
            except Exception as auto_error:
                raise ValueError(
                    f"Failed to load tokenizer for {model_name}: "
                    f"LlamaTokenizer error: {llama_error}; "
                    f"AutoTokenizer error: {auto_error}"
                ) from auto_error
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.tokenizer:
            return super().encode(text)
        try:
            return self.tokenizer.encode(text)
        except:
            return super().encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.tokenizer:
            return super().decode(token_ids)
        try:
            return self.tokenizer.decode(token_ids)
        except:
            return super().decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.tokenizer:
            return super().count_tokens(text)
        try:
            return len(self.tokenizer.encode(text))
        except:
            return super().count_tokens(text)
