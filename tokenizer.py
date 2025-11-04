import numpy as np
import re

special_tokens = ['<|start|>', '<|answer|>', '<|end|>']
special_tokens = []

class Tokenizer:
    def __init__(self, data=None, special_tokens=special_tokens, include_special_tokens=False, stoi={}, itos={}):
        self.data = data
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.include_special_tokens = include_special_tokens
        self.stoi = stoi
        self.itos = itos
        self.use_char_lookup = False
        self.char_lookup = None
        
        # Precompile regex for faster token matching
        if special_tokens:
            pattern = '|'.join(re.escape(token) for token in self.special_tokens)
            self.special_token_regex = re.compile(f'({pattern})')
        else:
            self.special_token_regex = None

    def fit(self):
        # Create character set using set operations
        if self.include_special_tokens:
            chars_set = set(self.data)
        else:
            special_chars = set(''.join(self.special_tokens))
            chars_set = set(c for c in self.data if c not in special_chars)
        
        self.all_chars = sorted(chars_set)
        self.vocab_size = len(self.special_tokens) + len(self.all_chars)
        print("Regular characters:", self.all_chars)
        print(f"Total vocab size (characters + special tokens): {self.vocab_size:,}")
        
        # Create mappings
        self.stoi = {token: i for i, token in enumerate(self.special_tokens)}
        self.stoi.update({ch: i + len(self.special_tokens) for i, ch in enumerate(self.all_chars)})
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Create character lookup array for faster encoding
        self._create_char_lookup()
    
    def _create_char_lookup(self):
        try:
            max_ord = max(ord(c) for c in self.all_chars if len(c) == 1)
            if max_ord < 65536:  # Reasonable size for an array
                self.char_lookup = np.full(max_ord + 1, -1, dtype=np.int32)
                for ch in self.all_chars:
                    if len(ch) == 1:
                        self.char_lookup[ord(ch)] = self.stoi[ch]
                self.use_char_lookup = True
            else:
                self.use_char_lookup = False
        except:
            self.use_char_lookup = False

    def encode(self, s):
        # Fast path for when there are no special tokens
        if not self.special_tokens:
            if self.use_char_lookup:
                char_codes = np.array([ord(c) for c in s], dtype=np.int32)
                return self.char_lookup[char_codes].tolist()
            else:
                return [self.stoi[c] for c in s]
        
        # Use regex-based splitting for strings with special tokens
        result = []
        last_end = 0
        
        for match in self.special_token_regex.finditer(s):
            start, end = match.span()
            
            # Process characters before the special token
            for i in range(last_end, start):
                if self.use_char_lookup:
                    result.append(self.char_lookup[ord(s[i])])
                else:
                    result.append(self.stoi[s[i]])
            
            # Add the special token
            result.append(self.stoi[match.group(0)])
            last_end = end
        
        # Process remaining characters
        for i in range(last_end, len(s)):
            if self.use_char_lookup:
                result.append(self.char_lookup[ord(s[i])])
            else:
                result.append(self.stoi[s[i]])
        
        return result

    def decode(self, l):
        return ''.join(self.itos[i] for i in l)