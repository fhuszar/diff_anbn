"""Character-level tokenizer for formal languages."""

from dataclasses import dataclass, field

import torch


@dataclass
class Tokenizer:
    """Character-level tokenizer with special tokens.

    Special tokens:
        PAD (0): Padding token
        MASK (1): Mask token for diffusion
        BOS (2): Beginning of sequence
        EOS (3): End of sequence
    """

    vocab: list[str]
    pad_token: str = "<PAD>"
    mask_token: str = "<MASK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"

    # Token IDs
    pad_id: int = field(init=False)
    mask_id: int = field(init=False)
    bos_id: int = field(init=False)
    eos_id: int = field(init=False)

    # Mappings
    _token_to_id: dict[str, int] = field(init=False, repr=False)
    _id_to_token: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        """Build token mappings."""
        # Special tokens first
        special_tokens = [self.pad_token, self.mask_token, self.bos_token, self.eos_token]
        all_tokens = special_tokens + list(self.vocab)

        self._token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self._id_to_token = {i: t for i, t in enumerate(all_tokens)}

        self.pad_id = self._token_to_id[self.pad_token]
        self.mask_id = self._token_to_id[self.mask_token]
        self.bos_id = self._token_to_id[self.bos_token]
        self.eos_id = self._token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._token_to_id)

    @property
    def num_special_tokens(self) -> int:
        """Number of special tokens."""
        return 4

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
        padding: bool = False,
    ) -> list[int]:
        """Encode a string to token IDs.

        Args:
            text: String to encode
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            max_length: Maximum length (will truncate if exceeded)
            padding: Whether to pad to max_length

        Returns:
            List of token IDs
        """
        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for char in text:
            if char in self._token_to_id:
                ids.append(self._token_to_id[char])
            else:
                raise ValueError(f"Unknown character: '{char}'")

        if add_eos:
            ids.append(self.eos_id)

        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            # Ensure EOS is at end if we truncated
            if add_eos and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id

        # Pad if needed
        if padding and max_length is not None:
            while len(ids) < max_length:
                ids.append(self.pad_id)

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs to a string.

        Args:
            ids: List of token IDs
            skip_special: Whether to skip special tokens in output

        Returns:
            Decoded string
        """
        tokens = []
        special_ids = {self.pad_id, self.mask_id, self.bos_id, self.eos_id}

        for id_ in ids:
            if id_ in self._id_to_token:
                if skip_special and id_ in special_ids:
                    continue
                tokens.append(self._id_to_token[id_])
            else:
                tokens.append(f"<UNK:{id_}>")

        return "".join(tokens)

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
        padding: bool = True,
    ) -> torch.Tensor:
        """Encode a batch of strings to a tensor.

        Args:
            texts: List of strings
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            max_length: Maximum length (inferred from batch if None)
            padding: Whether to pad (required for tensor output)

        Returns:
            Tensor of shape (batch_size, seq_len)
        """
        encoded = [self.encode(t, add_bos, add_eos) for t in texts]

        if max_length is None:
            max_length = max(len(e) for e in encoded)

        # Pad all sequences
        padded = []
        for seq in encoded:
            if len(seq) < max_length:
                seq = seq + [self.pad_id] * (max_length - len(seq))
            elif len(seq) > max_length:
                seq = seq[:max_length]
            padded.append(seq)

        return torch.tensor(padded, dtype=torch.long)

    def decode_batch(self, ids: torch.Tensor, skip_special: bool = True) -> list[str]:
        """Decode a batch of token IDs.

        Args:
            ids: Tensor of shape (batch_size, seq_len)
            skip_special: Whether to skip special tokens

        Returns:
            List of decoded strings
        """
        return [self.decode(row.tolist(), skip_special) for row in ids]
