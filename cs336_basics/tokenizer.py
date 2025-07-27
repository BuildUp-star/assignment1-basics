from collections.abc import Iterable, Iterator
import json
import re
import regex


class Tokenizer:
    # GPT-2 style pre-tokenizer regex pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None) -> None:
        """
        Initialize the BPE tokenizer.

        vocab: mapping from token ID (int) to token bytes (bytes)
        merges: list of merge operations in the order they were learned, each a pair of byte-strings
        special_tokens: list of string tokens to preserve (e.g. '<|endoftext|>')
        """
        # copy vocab and build reverse mapping bytes -> id
        self.vocab = dict(vocab)
        self._bytes_to_id = {tok: idx for idx, tok in self.vocab.items()}

        # store merges list (ordered)
        self.merges = list(merges)

        # handle special tokens: add to vocab if missing, keep list of strings
        self.special_tokens: list[str] = []
        if special_tokens:
            for tok in special_tokens:
                # encode string to bytes
                btok = tok.encode('utf-8')
                if btok not in self._bytes_to_id:
                    # assign new ID at end of vocab
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = btok
                    self._bytes_to_id[btok] = new_id
                self.special_tokens.append(tok)

        # compile a regex to split out special tokens
        if self.special_tokens:
            escaped = [re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)]
            pattern = '(' + '|'.join(escaped) + ')'
            self._special_split = re.compile(pattern)
        else:
            self._special_split = None

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        """
        Load tokenizer from serialized vocab and merges files.
        Assumes vocab file is JSON mapping str(id) -> token string,
        and merges file has one merge per line: "token1 token2".
        """
        # load vocab JSON: keys are strings, values are token strings
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        vocab: dict[int, bytes] = {}
        for k, v in raw.items():
            # convert key to int and value to bytes
            vocab[int(k)] = v.encode('utf-8')

        # load merges: each line has two tokens separated by whitespace
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                # encode each part to bytes
                merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.
        Special tokens are preserved; other text is pre-tokenized and BPE merges applied.
        """
        ids: list[int] = []

        # split text on special tokens if provided
        if self._special_split:
            parts = self._special_split.split(text)
        else:
            parts = [text]

        for part in parts:
            # if part is a special token, map directly
            if self._special_split and part in self.special_tokens:
                sid = self._bytes_to_id[part.encode('utf-8')]
                ids.append(sid)
                continue

            # apply GPT-2 style pre-tokenizer
            for match in regex.finditer(self.PAT, part):
                token_str = match.group(0)
                token_bytes = token_str.encode('utf-8')
                # initialize sequence: one-byte tokens
                seq = [bytes([b]) for b in token_bytes]

                # repeatedly apply merges in learned order
                while True:
                    # build list of adjacent pairs
                    pairs = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
                    # find first merge that applies
                    to_merge = None
                    for m in self.merges:
                        if m in pairs:
                            to_merge = m
                            break
                    if to_merge is None:
                        break
                    # apply merge: merge all occurrences of this pair
                    new_seq: list[bytes] = []
                    i = 0
                    while i < len(seq):
                        if i < len(seq) - 1 and (seq[i], seq[i+1]) == to_merge:
                            # merge the two tokens
                            new_seq.append(seq[i] + seq[i+1])
                            i += 2
                        else:
                            new_seq.append(seq[i])
                            i += 1
                    seq = new_seq

                # convert final byte tokens to IDs
                for piece in seq:
                    if piece not in self._bytes_to_id:
                        raise ValueError(f"Byte token {piece} not in vocabulary.")
                    ids.append(self._bytes_to_id[piece])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode each string from an iterable into token IDs.
        Useful for memory-efficient tokenization of large streams.
        """
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a Unicode string.
        Malformed byte sequences are replaced with the Unicode replacement character.
        """
        # reconstruct bytes sequence
        bseq = bytearray()
        for idx in ids:
            token = self.vocab.get(idx, None)
            if token is None:
                # unknown ID, skip or insert replacement
                continue
            bseq.extend(token)
        # decode with replacement of errors
        return bseq.decode('utf-8', errors='replace')
