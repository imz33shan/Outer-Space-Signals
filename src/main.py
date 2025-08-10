"""
WE HAVE ENCOUNTERED SIGNIFICANT DIFFICULTY PROCESSING THE AUDITORY EMISSIONS
"""

import os
import math
import random
import numpy as np
from scipy.stats import entropy
from collections import defaultdict


def score_text_with_quads(message: str, quadgram_scores: dict) -> int:
    """
    Score a message using quadgram probabilities.

    Why: A high score indicates the text structure is closer to real English.
    We use this to compare different decryption attempts and choose the most likely one.
    """
    cleaned = ''.join(ch.upper() for ch in message if ch.isalpha())
    return sum(
        quadgram_scores.get(cleaned[i:i+4], -15)
        for i in range(len(cleaned) - 3)
    )


def decipher(ciphertext: str, key: str) -> str:
    """
    Apply a substitution cipher key to decrypt text.

    Why: Translates ciphertext into plaintext guess based on the current key mapping.
    """
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapping = {alphabet[i]: key[i] for i in range(26)}
    return ''.join(mapping[c] if c.isalpha() else c for c in ciphertext.upper())


def random_key() -> str:
    """
    Generate a random substitution key.

    Why: Provides a random starting point for the key search algorithm.
    """
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    random.shuffle(alphabet)
    return ''.join(alphabet)


def swap_random(key: str) -> str:
    """
    Swap two random letters in the key.

    Why: Creates a new candidate key for simulated annealing search.
    """
    k = list(key)
    i, j = random.sample(range(26), 2)
    k[i], k[j] = k[j], k[i]
    return ''.join(k)


def load_quadgrams(filename: str) -> dict:
    """
    Load quadgram frequency data and convert counts to log probabilities.

    Why: This model is used to measure how likely a piece of text is real English.
    """
    quadgrams = defaultdict(int)
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            quad, count = line.strip().split()
            quadgrams[quad] = int(count)
    total = sum(quadgrams.values())
    for quad in quadgrams:
        quadgrams[quad] = math.log10(quadgrams[quad] / total)
    return quadgrams


def array_entropy(array: np.array) -> float:
    """
    Compute Shannon entropy of letters in an array (ignoring spaces).

    Why: Low entropy suggests structured, non-random text (likely part of the hidden message).
    """
    mask = array != ' '
    letters = array[mask]
    if letters.size == 0:
        return float('inf')
    unique, counts = np.unique(letters, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)


def is_right_array(pos: int, text: str, size: int) -> bool:
    """
    Check if a substring starts and ends at word boundaries.

    Why: Ensures extracted chunks don't cut words in half.
    """
    end = pos + size
    return all([
        0 <= pos,
        end <= len(text),
        text[pos].isalpha(),
        text[end - 1].isalpha(),
        (pos == 0 or text[pos - 1] == ' '),
        (end == len(text) or text[end] == ' ')
    ])


def find_message_position(data: np.array, length: int = 721) -> int:
    """
    Find the position of the most likely hidden message chunk in noisy text.

    Why: The hidden message will have lower entropy than surrounding noise.
    """
    array = np.array(list(data))
    best_score = float('inf')
    best_pos = None

    for start in range(len(array) - length + 1):
        if not is_right_array(start, data, length):
            continue
        chunk_array = array[start:start + length]
        score = array_entropy(chunk_array) - ((chunk_array == ' ').sum() / chunk_array.size)
        if score < best_score:
            best_score = score
            best_pos = start
    return best_pos


def crack_cipher(clean_text: str, key: str, score: int, quadgram_model: dict, max_steps: int) -> int:
    """
    Use simulated annealing to find the key that produces the most English-like text.

    Why: The search space of possible keys is huge; simulated annealing helps escape local maxima.
    """
    best_key, best_score = key, score
    temperature = 20.0
    last_update_step = 0

    for step in range(max_steps):
        candidate_key = swap_random(key)
        candidate_score = score_text_with_quads(
            decipher(clean_text, candidate_key),
            quadgram_model
        )

        if candidate_score > score or random.random() < math.exp((candidate_score - score) / temperature):
            key, score = candidate_key, candidate_score
            if score > best_score:
                best_key = key
                last_update_step = step

        temperature *= 0.9999
        if step - last_update_step > max_steps:
            break

    return best_key


def main() -> None:
    """Main process: locate message, crack cipher, and output decrypted text."""
    print("🛸 NASA Signal Decoder - Deciphering Messages from Planet Dyslexia 🛸")

    SIGNAL_FILE = "../signal.txt" if os.path.exists("../signal.txt") else "./signal.txt"
    QUAD_FILE = "../quadgrams.txt" if os.path.exists("../quadgrams.txt") else "./quadgrams.txt"

    MSG_LENGTH = 721
    MAX_ITERATIONS = 10000

    quadgrams = load_quadgrams(QUAD_FILE)

    with open(SIGNAL_FILE, 'r', encoding='utf-8', errors='replace') as f:
        signal_data = f.read()

    msg_pos = find_message_position(signal_data, MSG_LENGTH)
    ciphertext = signal_data[msg_pos:msg_pos + MSG_LENGTH]
    ciphertext_cleaned = ''.join(char for char in ciphertext if char.isalpha())

    initial_key = random_key()
    initial_score = score_text_with_quads(decipher(ciphertext_cleaned, initial_key), quadgrams)

    cracked_key = crack_cipher(ciphertext_cleaned, initial_key, initial_score, quadgrams, MAX_ITERATIONS)
    decrypted_message = decipher(ciphertext, cracked_key)

    print("\nDeciphered message:\n")
    print(decrypted_message)


if __name__ == "__main__":
    main()
