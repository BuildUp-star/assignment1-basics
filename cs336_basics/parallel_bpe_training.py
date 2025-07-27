import os
import re
import regex
from collections import Counter
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from cs336_basics.pretokenization_example import find_chunk_boundaries

# GPT-2 style pre-tokenizer regex pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_tokenizer = regex.compile(PAT)

# Worker function must be at top-level for multiprocessing

def _count_chunk(args) -> Counter:
    """
    Worker: count byte-sequence tokens in a file chunk.
    """
    path, start, end, specials = args
    local_ctr = Counter()
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode('utf-8', errors='ignore')

    # split on special tokens, preserving them
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
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer with parallel pre-tokenization."""
    # 1. determine chunk boundaries via special token
    split_tok = special_tokens[0].encode('utf-8') if special_tokens else b'\n'
    with open(input_path, 'rb') as f:
        num_chunks = cpu_count()
        boundaries = find_chunk_boundaries(f, num_chunks, split_tok)

    # 2. create tasks for each file chunk
    tasks = [(input_path, st, ed, special_tokens)
             for st, ed in zip(boundaries[:-1], boundaries[1:])]

    # 3. parallel count pre-token byte sequences
    with Pool(num_chunks) as pool:
        partials = pool.map(_count_chunk, tasks)

    # 4. merge partial counters
    seq_counts = Counter()
    for p in partials:
        seq_counts.update(p)

    # 5. iterative BPE merges
    merges: List[Tuple[bytes, bytes]] = []
    num_merges = max(0, vocab_size - (256 + len(special_tokens)))
    for _ in range(num_merges):
        pair_counts = Counter()
        for seq, cnt in seq_counts.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i+1])] += cnt
        if not pair_counts:
            break
        max_c = max(pair_counts.values())
        candidates = [p for p, c in pair_counts.items() if c == max_c]
        best = max(candidates)
        merges.append(best)

        # apply merge
        A, B = best
        new_counts = Counter()
        for seq, cnt in seq_counts.items():
            merged_seq: List[bytes] = []
            i = 0
            while i < len(seq):
                if i < len(seq)-1 and seq[i] == A and seq[i+1] == B:
                    merged_seq.append(A + B)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_counts[tuple(merged_seq)] += cnt
        seq_counts = new_counts

    # 6. build final vocab
    vocab: Dict[int, bytes] = {}
    idx = 0
    for tok in special_tokens:
        vocab[idx] = tok.encode('utf-8'); idx += 1
    for b in range(256):
        vocab[idx] = bytes([b]); idx += 1
    for A, B in merges:
        vocab[idx] = A + B; idx += 1

    return vocab, merges