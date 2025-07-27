import re
import os
import regex
from collections import Counter
from typing import List, Tuple, Dict

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return its vocabulary and merge operations.

    Args:
        input_path (str | os.PathLike): Path to the training corpus file.
        vocab_size (int): Total number of tokens to include in the vocabulary (including special tokens).
        special_tokens (list[str]): List of tokens that must stay intact during tokenization.
            These tokens are never split and always treated as single units.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: Mapping from token IDs to token byte sequences.
            merges: Ordered list of byte-pair merge operations applied during training.
    """
    # GPT-2 style pre-tokenizer regex pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 1. Read the entire input file as UTF-8 text
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # 2. Split on special tokens to ensure merges do not cross their boundaries
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        parts = re.split("(" + "|".join(escaped_tokens) + ")", data)
    else:
        parts = [data]

    # 3. Pre-tokenize each part and count byte-sequence frequencies
    tokenizer = regex.compile(PAT)
    seq_counts: Counter = Counter()
    for part in parts:
        # If this part exactly matches a special token, count it as a single token
        if part in special_tokens:
            token_bytes = part.encode('utf-8')
            #seq_counts[(token_bytes,)] += 1
            continue
        # Otherwise, apply regex to find pre-tokens
        for match in tokenizer.finditer(part):
            tok = match.group(0).encode('utf-8')
            # Break into individual bytes and count
            seq = tuple(bytes([b]) for b in tok)
            seq_counts[seq] += 1

    # 4. Perform byte-pair merges iteratively
    merges: List[Tuple[bytes, bytes]] = []
    num_merges = max(0, vocab_size - (256 + len(special_tokens)))
    for _ in range(num_merges):
        # 4.1 Count all adjacent byte-pair frequencies
        pair_counts: Counter = Counter()
        for seq, count in seq_counts.items():
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += count
        if not pair_counts:
            break

        # 4.2 Choose the most frequent pair (tie-break by lex order)
        max_count = max(pair_counts.values())
        candidates = [p for p, c in pair_counts.items() if c == max_count]
        best_pair = max(candidates)
        merges.append(best_pair)

        # 4.3 Replace occurrences of the selected pair in all sequences
        new_seq_counts: Counter = Counter()
        A, B = best_pair
        for seq, count in seq_counts.items():
            merged_seq: List[bytes] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == A and seq[i+1] == B:
                    # Merge the pair into a single token
                    merged_seq.append(A + B)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_seq_counts[tuple(merged_seq)] += count
        seq_counts = new_seq_counts

    # 5. Build the final vocabulary: special tokens, single-byte tokens, then merge tokens
    vocab: Dict[int, bytes] = {}
    idx = 0
    # 5.1 Add special tokens first
    for tok in special_tokens:
        vocab[idx] = tok.encode('utf-8')
        idx += 1
    # 5.2 Add all 256 possible single-byte tokens
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    # 5.3 Add merged tokens in the order they were created
    for A, B in merges:
        vocab[idx] = A + B
        idx += 1

    return vocab, merges
