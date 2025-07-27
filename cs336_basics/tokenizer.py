from collections import Counter
from collections.abc import Iterable, Iterator
import json
import re
import regex


class Tokenizer:
    # GPT-2 style pre-tokenizer regex pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab:dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens:list[str] | None = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str] | None = None):
        """
        Load a tokenizer from vocab and merges files.
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [tuple(line.strip().split()) for line in f.readlines()]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        res = []
        #1, process the text to handle special tokens
        if self.special_tokens:
            escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
            parts = re.split("(" + "|".join(escaped_tokens) + ")", text)
        else:
            parts = [text]
        #2, pretokenize to get words
        tokenizer = regex.compile(self.PAT)
        for part in parts:
            #special tokens are already handled, so we can just tokenize the rest
            if part in self.special_tokens:
                res.append(self.vocab.get(part, -1))
                continue
            #tokenize the part using the regex pattern
            #and apply merges to the tokens
            for match in tokenizer.finditer(part):
                tok = match.group(0).encode('utf-8')
                seq = tuple(bytes([b]) for b in tok)
                #apply merges to the tokens
                while True:
                    merged_seq = []
                    for i in range(len(seq) - 1):
                        pair = (seq[i], seq[i + 1])
                        if pair in self.merges:
                            merged_seq.append(seq[i] + seq[i + 1])
                        else:
                            merged_seq.append(seq[i])
                        i += 1
                    if len(seq) == len(merged_seq):
                        break
                    seq = merged_seq
                #3, convert the tokens to ids
                for token in seq:
                    if token in self.vocab:
                        res.append(self.vocab[token])
                    else:
                        assert False, f"Token {token} not found in vocab"
        #4, return the list of ids
        return res    

        #4, apply merges to the tokens until no more merges can be applied

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # Reverse the vocab to get the tokens
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = []
        for id in ids:
            tokens.append(reverse_vocab.get(id, b"<unk>"))
        return b"".join(tokens).decode("utf-8", errors="replace")