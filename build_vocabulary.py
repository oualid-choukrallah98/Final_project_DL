"""
Build vocabulary from training captions
Following TODO.md Day 3-4 requirements
"""

import pandas as pd
import json
import re
from collections import Counter

print("="*80)
print("BUILDING VOCABULARY FROM TRAINING DATA")
print("="*80)

# Load training data
train_df = pd.read_csv('data/train_data.csv')
print(f"\nLoaded {len(train_df)} training samples")

# Special tokens
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<BOS>': 1,  # Beginning of sequence
    '<EOS>': 2,  # End of sequence
    '<UNK>': 3   # Unknown word
}

def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = str(text).lower()

    # Keep medical terms intact, but normalize spaces
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text

def tokenize(text):
    """Simple word-level tokenization"""
    # Split on whitespace and punctuation but keep punctuation
    tokens = re.findall(r'\w+|[.,!?;]', text)
    return tokens

# Collect all words from training captions
print("\nTokenizing training captions...")
all_tokens = []

for caption in train_df['findings']:
    if pd.notna(caption):
        cleaned = clean_text(caption)
        tokens = tokenize(cleaned)
        all_tokens.extend(tokens)

print(f"Total tokens in training data: {len(all_tokens)}")

# Count word frequencies
word_freq = Counter(all_tokens)
print(f"Unique tokens: {len(word_freq)}")

# Filter by minimum frequency (words appearing less than 5 times -> <UNK>)
MIN_FREQ = 5
filtered_words = {word: count for word, count in word_freq.items() if count >= MIN_FREQ}
print(f"Tokens appearing >= {MIN_FREQ} times: {len(filtered_words)}")

# Build vocabulary
print("\nBuilding vocabulary...")
vocab = SPECIAL_TOKENS.copy()

# Add words sorted by frequency (most common first)
sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_words:
    if word not in vocab:
        vocab[word] = len(vocab)

print(f"Final vocabulary size: {len(vocab)}")

# Create reverse mapping (index -> word)
idx2word = {idx: word for word, idx in vocab.items()}

# Save vocabulary
vocab_data = {
    'word2idx': vocab,
    'idx2word': idx2word,
    'vocab_size': len(vocab),
    'min_freq': MIN_FREQ,
    'special_tokens': SPECIAL_TOKENS,
    'total_tokens': len(all_tokens),
    'unique_tokens': len(word_freq)
}

with open('data/vocab.json', 'w') as f:
    json.dump(vocab_data, f, indent=2)

print("\n✓ Vocabulary saved to: data/vocab.json")

# Statistics
print("\n" + "="*80)
print("VOCABULARY STATISTICS")
print("="*80)
print(f"Vocabulary size: {len(vocab)}")
print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
print(f"  Medical/clinical words: {len(vocab) - len(SPECIAL_TOKENS)}")
print(f"\nToken coverage:")
print(f"  Tokens in training data: {len(all_tokens):,}")
print(f"  Unique tokens: {len(word_freq):,}")
print(f"  Tokens in vocab (freq >= {MIN_FREQ}): {len(filtered_words):,}")
print(f"  Tokens filtered out (<UNK>): {len(word_freq) - len(filtered_words):,}")

# Show most common words
print("\n" + "="*80)
print("TOP 30 MOST COMMON WORDS")
print("="*80)
for i, (word, count) in enumerate(sorted_words[:30], 1):
    print(f"{i:2d}. {word:<20} ({count:>5} occurrences)")

# Show some example tokenization
print("\n" + "="*80)
print("EXAMPLE TOKENIZATIONS")
print("="*80)

def caption_to_ids(caption, vocab):
    """Convert caption to token IDs"""
    cleaned = clean_text(caption)
    tokens = tokenize(cleaned)

    # Add BOS and EOS tokens
    ids = [vocab['<BOS>']]
    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab['<UNK>'])
    ids.append(vocab['<EOS>'])

    return ids, tokens

# Show 3 examples
for i in range(min(3, len(train_df))):
    caption = train_df.iloc[i]['findings']
    if pd.notna(caption):
        ids, tokens = caption_to_ids(caption, vocab)
        print(f"\nExample {i+1}:")
        print(f"Original: {caption[:100]}...")
        print(f"Tokens ({len(tokens)}): {tokens[:15]}...")
        print(f"IDs ({len(ids)}): {ids[:15]}...")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Create text preprocessing module using this vocabulary")
print("2. Create image preprocessing module")
print("3. Build PyTorch Dataset class")
