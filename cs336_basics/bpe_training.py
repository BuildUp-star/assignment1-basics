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

    This optimized version avoids recalculating pair statistics from scratch in each
    iteration, providing a significant speedup.

    Args:
        input_path (str | os.PathLike): Path to the training corpus file.
        vocab_size (int): Total number of tokens to include in the vocabulary.
        special_tokens (list[str]): List of tokens that must stay intact.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: Mapping from token IDs to token byte sequences.
            merges: Ordered list of byte-pair merge operations.
    """
    # GPT-2 style pre-tokenizer regex pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 1. Read the entire input file as UTF-8 text
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # 2. Split on special tokens to ensure merges do not cross their boundaries
    # Optimization: Use a set for faster lookups.
    special_tokens_set = set(special_tokens)
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        parts = re.split("(" + "|".join(escaped_tokens) + ")", data)
    else:
        parts = [data]

    # 3. Pre-tokenize each part and count byte-sequence frequencies
    tokenizer = regex.compile(PAT)
    seq_counts: Counter = Counter()
    for part in parts:
        if part in special_tokens_set:
            continue
        for match in tokenizer.finditer(part):
            tok = match.group(0).encode('utf-8')
            seq = tuple(bytes([b]) for b in tok)
            seq_counts[seq] += 1

    # --- Core Optimization: Incremental Stats Update ---
    
    # 4. Pre-calculate initial pair frequencies
    pair_counts: Counter = Counter()
    for seq, count in seq_counts.items():
        for i in range(len(seq) - 1):
            pair_counts[seq[i], seq[i+1]] += count

    # 5. Perform byte-pair merges iteratively
    merges: List[Tuple[bytes, bytes]] = []
    num_merges = max(0, vocab_size - (256 + len(special_tokens)))
    for i in range(num_merges):
        if not pair_counts:
            break

        # 5.1 Choose the most frequent pair.
        # Optimization: Find the best pair in a single pass.
        # The key sorts by count (desc) and then by the pair itself (lexicographically).
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)
        A, B = best_pair
        C = A + B

        # 5.2 Merge the best pair in all sequences and update pair_counts incrementally.
        new_seq_counts: Counter = Counter()
        for seq, count in seq_counts.items():
            # Find occurrences of the best pair in the current sequence
            j = 0
            new_seq: List[bytes] = []
            did_merge = False
            while j < len(seq):
                # Check for the pair (A, B) starting at index j
                is_pair = j < len(seq) - 1 and seq[j] == A and seq[j+1] == B
                if is_pair:
                    # It's a match! Merge it.
                    new_seq.append(C)
                    did_merge = True

                    # Update pair counts affected by this merge
                    # 1. Decrement count for the pair involving the left neighbor and A
                    if j > 0:
                        pair_counts[seq[j-1], A] -= count
                    # 2. Decrement count for the pair involving B and the right neighbor
                    if j < len(seq) - 2:
                        pair_counts[B, seq[j+2]] -= count
                    
                    j += 2 # Move past the merged pair
                else:
                    # No match, just copy the token
                    new_seq.append(seq[j])
                    j += 1

            new_seq_tuple = tuple(new_seq)
            new_seq_counts[new_seq_tuple] += count
            
            # If a merge happened, update counts for new pairs formed with C
            if did_merge:
                for k in range(len(new_seq) - 1):
                    # Check if the pair involves the new token C
                    if new_seq[k] == C or new_seq[k+1] == C:
                        pair_counts[new_seq[k], new_seq[k+1]] += count
        
        seq_counts = new_seq_counts
        # The merged pair itself is no longer a possible pair
        del pair_counts[best_pair]

    # 6. Build the final vocabulary
    vocab: Dict[int, bytes] = {}
    idx = 0
    # 6.1 Add special tokens
    for tok in special_tokens:
        vocab[idx] = tok.encode('utf-8')
        idx += 1
    # 6.2 Add all single-byte tokens
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    # 6.3 Add merged tokens
    for A, B in merges:
        vocab[idx] = A + B
        idx += 1

    return vocab, merges