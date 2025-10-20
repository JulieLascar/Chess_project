"""https://github.com/l-pommeret/chess_char"""

from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import json
import os
from tqdm import tqdm

@dataclass
class TokenizerConfig:
    pad_token: str = '<PAD>'
    initial_tokens: List[str] = None
    special_tokens: List[str] = None
    save_path: Optional[str] = None

class ChessTokenizer:
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab: Dict[str, int] = {self.config.pad_token: 0}
        self.id2token: Dict[int, str] = {0: self.config.pad_token}
        self.token_id: int = 1
        self.valid_chars: Set[str] = set()
        self._initialize_vocab()
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of valid characters: {len(self.valid_chars)}")

    def _initialize_vocab(self) -> None:
        """Initialize the vocabulary with all necessary characters."""
        # 1. Basic chess characters
        basic_chars = set('abcdefgh12345678RNBQKP')
        self.valid_chars.update(basic_chars)
        for char in basic_chars:
            self._add_token(char)

        # 2. Notation symbols
        notation_chars = set('x+#=O-0123456789.')
        self.valid_chars.update(notation_chars)
        for char in notation_chars:
            self._add_token(char)

        # 3. Formatting characters
        format_chars = set(' \n\t(),[]"\'/!')
        self.valid_chars.update(format_chars)
        for char in format_chars:
            self._add_token(char)

        # 4. Special tokens
        if self.config.special_tokens:
            for token in self.config.special_tokens:
                self._add_token(token)

    def _add_token(self, token: str) -> None:
        if token not in self.vocab:
            self.vocab[token] = self.token_id
            self.id2token[self.token_id] = token
            self.token_id += 1

    def clean_text(self, text: str) -> str:
        # Normalize special characters
        replacements = {
            "'": "'",
            """: '"',
            """: '"',
            "–": "-",
            "—": "-",
            "…": "...",
            "\r": "\n",
            "\u2026": "..."
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove invalid characters
        cleaned = ''.join(char for char in text if char in self.valid_chars)
        
        # Normalize spaces
        return ' '.join(cleaned.split())

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the text character by character."""
        text = self.clean_text(text)
        return list(text)

    def is_valid_text(self, text: str) -> bool:
        text = self.clean_text(text)
        
        # Check for invalid characters
        invalid_chars = set(char for char in text if char not in self.valid_chars)
        if invalid_chars:
            return False
            
        # Check that the text is not empty
        if not text.strip():
            return False
            
        return True

    def encode(self, text: str) -> List[int]:
        """Encode the text, handling potential errors."""
        try:
            return [self.vocab[token] for token in self.tokenize(text)]
        except KeyError as e:
            print(f"Encoding error - token not found: {e}")
            print(f"Problematic text: {text}")
            raise

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id2token[id] for id in ids 
                      if id != self.vocab[self.config.pad_token])

    def save(self, path: Optional[str] = None) -> None:
        save_path = path or self.config.save_path
        if save_path is None:
            raise ValueError("No save path specified")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'id2token': {str(k): v for k, v in self.id2token.items()},
            'valid_chars': list(self.valid_chars),
            'config': {
                'pad_token': self.config.pad_token,
                'special_tokens': self.config.special_tokens
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> 'ChessTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        config = TokenizerConfig(
            pad_token=data['config']['pad_token'],
            special_tokens=data['config']['special_tokens'],
            save_path=path
        )
        
        tokenizer = cls(config)
        tokenizer.vocab = data['vocab']
        tokenizer.id2token = {int(k): v for k, v in data['id2token'].items()}
        tokenizer.valid_chars = set(data['valid_chars'])
        tokenizer.token_id = max(map(int, data['id2token'].keys())) + 1
        
        return tokenizer

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None,
                    show_progress: bool = True) -> List[List[int]]:
        iterator = tqdm(texts) if show_progress else texts
        encoded = []
        
        for text in iterator:
            try:
                tokens = self.encode(text)
                if max_length:
                    tokens = tokens[:max_length]
                    if len(tokens) < max_length:
                        tokens += [self.vocab[self.config.pad_token]] * (max_length - len(tokens))
                encoded.append(tokens)
            except Exception as e:
                print(f"Error encoding text: {e}")
                continue
                
        return encoded

    def debug_text(self, text: str) -> None:
        """Display debug information about the text."""
        print("\nDEBUG TEXT:")
        print("Original text:", text)
        print("Length:", len(text))
        print("Unique characters:", sorted(set(text)))
        
        cleaned = self.clean_text(text)
        print("\nCleaned text:", cleaned)
        print("Length after cleaning:", len(cleaned))
        print("Unique characters after cleaning:", sorted(set(cleaned)))
        
        print("\nTokens:", self.tokenize(cleaned))
        
        try:
            encoded = self.encode(cleaned)
            print("Successful encoding:", encoded)
            print("Decoding:", self.decode(encoded))
        except Exception as e:
            print(f"Encoding error: {e}")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.vocab_size

if __name__ == "__main__":
    config = TokenizerConfig(
        pad_token='<PAD>',
        special_tokens=['<START>', '<END>'],
        save_path='./models/chess_tokenizer.json'
    )
    
    tokenizer = ChessTokenizer(config)
    
    # Test
    game = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    print("\nTokenization test:")
    print("Text:", game)
    tokens = tokenizer.tokenize(game)
    print("Tokens:", tokens)
    ids = tokenizer.encode(game)
    print("IDs:", ids)
    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)
    
    tokenizer.save()