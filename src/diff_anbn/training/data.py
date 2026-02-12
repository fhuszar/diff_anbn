"""Data loading utilities."""

import torch
from torch.utils.data import Dataset, DataLoader

from diff_anbn.languages import FormalLanguage, Tokenizer


class FormalLanguageDataset(Dataset):
    """Dataset that generates samples from a formal language on-the-fly."""

    def __init__(
        self,
        language: FormalLanguage,
        tokenizer: Tokenizer,
        num_samples: int,
        min_n: int = 1,
        max_n: int = 10,
        max_seq_len: int = 64,
    ):
        """Initialize dataset.

        Args:
            language: Formal language to sample from
            tokenizer: Tokenizer for encoding strings
            num_samples: Number of samples per epoch
            min_n: Minimum generation parameter
            max_n: Maximum generation parameter
            max_seq_len: Maximum sequence length (for padding)
        """
        self.language = language
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.min_n = min_n
        self.max_n = max_n
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a sample.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Generate a random valid string
        text = self.language.generate(self.min_n, self.max_n)

        # Encode with BOS/EOS
        ids = self.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
            max_length=self.max_seq_len,
            padding=True,
        )

        input_ids = torch.tensor(ids, dtype=torch.long)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.tokenizer.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def create_dataloader(
    language: FormalLanguage,
    tokenizer: Tokenizer,
    batch_size: int,
    num_samples: int = 10000,
    min_n: int = 1,
    max_n: int = 10,
    max_seq_len: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for training.

    Args:
        language: Formal language to sample from
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        num_samples: Number of samples per epoch
        min_n: Minimum generation parameter
        max_n: Maximum generation parameter
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance
    """
    dataset = FormalLanguageDataset(
        language=language,
        tokenizer=tokenizer,
        num_samples=num_samples,
        min_n=min_n,
        max_n=max_n,
        max_seq_len=max_seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
