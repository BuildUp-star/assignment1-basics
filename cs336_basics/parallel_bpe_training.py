import os
import re
import regex
from collections import Counter
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count

from cs336_basics.pretokenization_example import find_chunk_boundaries

# GPT-2 style pre-tokenizer regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_tokenizer = regex.compile(PAT)


def _count_chunk(args) -> Counter:
    """
    Worker function: read a file chunk, pre-tokenize, count byte-sequence tokens.
    """
    path, start, end, specials = args
    local_ctr = Counter()
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode('utf-8', errors='ignore')

    # Split on special tokens, preserving them as separate parts
    if specials:
        escaped = "|".join(re.escape(tok) for tok in specials)
        parts = re.split(f"({escaped})", text)
    else:
        parts = [text]

    for part in parts:
        if part in specials:
            b = part.encode('utf-8')
            local_ctr[(b,)] += 1
        else:
            for m in _tokenizer.finditer(part):
                tok = m.group(0).encode('utf-8')
                seq = tuple(bytes([b]) for b in tok)
                local_ctr[seq] += 1
    return local_ctr


def run_train_bpe_parallel(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer, using parallel pre-tokenization.

    Args:
      input_path: Path to the training text file.
      vocab_size: Desired maximum vocabulary size (including 256 byte tokens, special tokens, and merges).
      special_tokens: List of strings that should never be split during training.

    Returns:
      vocab: A dict mapping token IDs to token byte sequences.
      merges: A list of byte-pair merges in the order they were created.
    """
    # Determine split token for chunk boundaries
    split_tok = special_tokens[0].encode('utf-8') if special_tokens else b'\n'
    with open(input_path, 'rb') as f:
        num_chunks = cpu_count()
        boundaries = find_chunk_boundaries(f, num_chunks, split_tok)

    # Prepare tasks for worker processes
    tasks = [(input_path, start, end, special_tokens)
             for start, end in zip(boundaries[:-1], boundaries[1:])]

    # Parallel pre-tokenization and counting
    with Pool(num_chunks) as pool:
        partial_counters = pool.map(_count_chunk, tasks)

    # Merge partial counts into global sequence counts
    seq_counts = Counter()
    for pc in partial_counters:
        seq_counts.update(pc)

    # Perform BPE merges iteratively
    merges: List[Tuple[bytes, bytes]] = []
    num_merges = max(0, vocab_size - (256 + len(special_tokens)))
    for _ in range(num_merges):
        pair_counts = Counter()
        for seq, cnt in seq_counts.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i+1])] += cnt
        if not pair_counts:
            break

        max_count = max(pair_counts.values())
        candidates = [p for p, c in pair_counts.items() if c == max_count]
        best = max(candidates)
        merges.append(best)

        # Apply merge to all sequences
        A, B = best
        new_counts = Counter()
        for seq, cnt in seq_counts.items():
            merged_seq: List[bytes] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == A and seq[i+1] == B:
                    merged_seq.append(A + B)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_counts[tuple(merged_seq)] += cnt
        seq_counts = new_counts

    # Build the final vocabulary: special tokens, single-byte tokens, then merges
    vocab: Dict[int, bytes] = {}
    idx = 0
    for tok in special_tokens:
        vocab[idx] = tok.encode('utf-8')
        idx += 1
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    for A, B in merges:
        vocab[idx] = A + B
        idx += 1

    return vocab, merges
