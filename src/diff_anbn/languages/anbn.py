"""a^n b^n language implementation."""

import random

from .base import FormalLanguage, ValidationResult


class AnBnLanguage(FormalLanguage):
    """Language of strings a^n b^n for n >= 1.

    Examples: "ab", "aabb", "aaabbb", etc.
    """

    @property
    def name(self) -> str:
        return "anbn"

    @property
    def vocab(self) -> list[str]:
        return ["a", "b"]

    def generate(self, min_n: int = 1, max_n: int = 10) -> str:
        """Generate a random a^n b^n string.

        Args:
            min_n: Minimum value of n
            max_n: Maximum value of n

        Returns:
            String of form a^n b^n
        """
        n = random.randint(min_n, max_n)
        return "a" * n + "b" * n

    def validate(self, s: str) -> bool:
        """Check if string is of form a^n b^n."""
        return self.validate_detailed(s).is_valid

    def validate_detailed(self, s: str) -> ValidationResult:
        """Validate with detailed error information."""
        if not s:
            return ValidationResult(False, "Empty string", 0)

        # Check for invalid characters
        for i, c in enumerate(s):
            if c not in self.vocab:
                return ValidationResult(False, f"Invalid character '{c}'", i)

        # Find transition point from 'a' to 'b'
        a_count = 0
        for i, c in enumerate(s):
            if c == "a":
                a_count += 1
            else:
                break

        # Check that all 'a's come before 'b's
        b_start = a_count
        for i in range(b_start, len(s)):
            if s[i] != "b":
                return ValidationResult(
                    False, f"Expected 'b' but found '{s[i]}' (a's must precede b's)", i
                )

        b_count = len(s) - a_count

        # Check equal counts
        if a_count != b_count:
            return ValidationResult(
                False,
                f"Unequal counts: {a_count} a's and {b_count} b's",
                len(s) - 1 if a_count < b_count else a_count,
            )

        if a_count == 0:
            return ValidationResult(False, "String must contain at least one 'a'", 0)

        return ValidationResult(True)
