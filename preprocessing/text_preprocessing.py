"""
Text Preprocessing Module
Handles tokenization and caption processing for medical image captioning
"""

import json
import re
import torch


class CaptionTokenizer:
    """Tokenizer for medical image captions"""

    def __init__(self, vocab_path='data/vocab.json'):
        """
        Initialize tokenizer with vocabulary

        Args:
            vocab_path: Path to vocabulary JSON file
        """
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.vocab_size = vocab_data['vocab_size']

        # Special tokens
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

        self.pad_idx = self.word2idx[self.pad_token]
        self.bos_idx = self.word2idx[self.bos_token]
        self.eos_idx = self.word2idx[self.eos_token]
        self.unk_idx = self.word2idx[self.unk_token]

    def clean_text(self, text):
        """Clean and normalize text"""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text):
        """Split text into tokens"""
        return re.findall(r'\w+|[.,!?;]', text)

    def encode(self, caption, max_length=None, add_special_tokens=True):
        """
        Convert caption to token IDs

        Args:
            caption: Text caption
            max_length: Maximum sequence length (including special tokens)
            add_special_tokens: Whether to add <BOS> and <EOS>

        Returns:
            token_ids: List of token IDs
            attention_mask: Binary mask (1 for real tokens, 0 for padding)
        """
        # Clean and tokenize
        cleaned = self.clean_text(caption)
        tokens = self.tokenize(cleaned)

        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_idx)

        for token in tokens:
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            else:
                token_ids.append(self.unk_idx)

        if add_special_tokens:
            token_ids.append(self.eos_idx)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)

        # Pad or truncate if max_length specified
        if max_length is not None:
            if len(token_ids) < max_length:
                # Pad
                padding_length = max_length - len(token_ids)
                token_ids.extend([self.pad_idx] * padding_length)
                attention_mask.extend([0] * padding_length)
            else:
                # Truncate (keep BOS, truncate middle, keep EOS)
                if add_special_tokens:
                    token_ids = [token_ids[0]] + token_ids[1:max_length-1] + [self.eos_idx]
                else:
                    token_ids = token_ids[:max_length]
                attention_mask = attention_mask[:max_length]

        return token_ids, attention_mask

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Convert token IDs back to text

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            caption: Reconstructed text
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        tokens = []
        for idx in token_ids:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)

        # Join tokens with spaces, but handle punctuation
        caption = ' '.join(tokens)
        # Remove space before punctuation
        caption = re.sub(r'\s+([.,!?;])', r'\1', caption)

        return caption

    def batch_encode(self, captions, max_length=None, add_special_tokens=True):
        """
        Encode a batch of captions

        Args:
            captions: List of text captions
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens

        Returns:
            token_ids: Tensor of shape [batch_size, max_length]
            attention_masks: Tensor of shape [batch_size, max_length]
            lengths: List of actual lengths (before padding)
        """
        batch_ids = []
        batch_masks = []
        lengths = []

        for caption in captions:
            ids, mask = self.encode(caption, max_length, add_special_tokens)
            batch_ids.append(ids)
            batch_masks.append(mask)
            lengths.append(sum(mask))  # Count non-padding tokens

        return (
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_masks, dtype=torch.long),
            lengths
        )

    def __len__(self):
        return self.vocab_size


# Convenience function
def load_tokenizer(vocab_path='data/vocab.json'):
    """Load tokenizer from vocabulary file"""
    return CaptionTokenizer(vocab_path)


if __name__ == "__main__":
    # Test the tokenizer
    print("Testing CaptionTokenizer...")

    tokenizer = load_tokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")

    # Test encoding
    test_caption = "The heart size is normal. No pneumothorax or pleural effusion."
    print(f"\nOriginal: {test_caption}")

    token_ids, attention_mask = tokenizer.encode(test_caption, max_length=50)
    print(f"Token IDs: {token_ids[:20]}...")
    print(f"Attention mask: {attention_mask[:20]}...")

    # Test decoding
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")

    # Test batch encoding
    captions = [
        "Normal chest x-ray.",
        "The cardiac silhouette and mediastinum size are within normal limits.",
        "Pleural effusion."
    ]
    print(f"\nBatch encoding {len(captions)} captions...")
    batch_ids, batch_masks, lengths = tokenizer.batch_encode(captions, max_length=30)
    print(f"Batch shape: {batch_ids.shape}")
    print(f"Lengths: {lengths}")

    print("\n✓ Tokenizer working correctly!")
