"""Dyck language implementation (balanced parentheses)."""

import random

from .base import FormalLanguage, ValidationResult


class DyckLanguage(FormalLanguage):
    """Dyck-1 language: balanced parentheses.

    Examples: "()", "(())", "()()", "(()())", etc.
    """

    def __init__(self, open_char: str = "(", close_char: str = ")"):
        """Initialize Dyck language with bracket characters.

        Args:
            open_char: Opening bracket character
            close_char: Closing bracket character
        """
        self._open = open_char
        self._close = close_char

    @property
    def name(self) -> str:
        return "dyck"

    @property
    def vocab(self) -> list[str]:
        return [self._open, self._close]

    def generate(self, min_n: int = 1, max_n: int = 10) -> str:
        """Generate a random balanced parentheses string.

        Uses a random walk that stays non-negative, generating n pairs.

        Args:
            min_n: Minimum number of bracket pairs
            max_n: Maximum number of bracket pairs

        Returns:
            Balanced parentheses string
        """
        n = random.randint(min_n, max_n)
        return self._generate_dyck(n)

    def _generate_dyck(self, n: int) -> str:
        """Generate a Dyck word with exactly n pairs using ballot sequences."""
        if n == 0:
            return ""

        result = []
        opens_remaining = n
        closes_remaining = n
        depth = 0

        while opens_remaining > 0 or closes_remaining > 0:
            can_open = opens_remaining > 0
            can_close = closes_remaining > 0 and depth > 0

            if can_open and can_close:
                # Choose randomly, but bias toward opening if depth is low
                if random.random() < 0.5:
                    result.append(self._open)
                    opens_remaining -= 1
                    depth += 1
                else:
                    result.append(self._close)
                    closes_remaining -= 1
                    depth -= 1
            elif can_open:
                result.append(self._open)
                opens_remaining -= 1
                depth += 1
            else:
                result.append(self._close)
                closes_remaining -= 1
                depth -= 1

        return "".join(result)

    def validate(self, s: str) -> bool:
        """Check if string is balanced parentheses."""
        return self.validate_detailed(s).is_valid

    def validate_detailed(self, s: str) -> ValidationResult:
        """Validate with detailed error information using stack."""
        if not s:
            return ValidationResult(False, "Empty string", 0)

        # Check for invalid characters
        for i, c in enumerate(s):
            if c not in self.vocab:
                return ValidationResult(False, f"Invalid character '{c}'", i)

        depth = 0
        for i, c in enumerate(s):
            if c == self._open:
                depth += 1
            else:  # close
                if depth == 0:
                    return ValidationResult(False, "Unmatched closing bracket", i)
                depth -= 1

        if depth > 0:
            return ValidationResult(
                False, f"Unmatched opening bracket(s): {depth} unclosed", len(s) - 1
            )

        return ValidationResult(True)


class Dyck2Language(FormalLanguage):
    """Dyck-2 language: two types of balanced brackets.

    Examples: "()", "[]", "([])", "([()])", etc.
    """

    def __init__(self):
        self._brackets = [("(", ")"), ("[", "]")]

    @property
    def name(self) -> str:
        return "dyck2"

    @property
    def vocab(self) -> list[str]:
        return ["(", ")", "[", "]"]

    def generate(self, min_n: int = 1, max_n: int = 10) -> str:
        """Generate a random balanced string with two bracket types."""
        n = random.randint(min_n, max_n)
        return self._generate_dyck2(n)

    def _generate_dyck2(self, n: int) -> str:
        """Generate a Dyck-2 word with n total pairs."""
        if n == 0:
            return ""

        result = []
        stack = []
        opens_remaining = n
        closes_remaining = n

        while opens_remaining > 0 or closes_remaining > 0:
            can_open = opens_remaining > 0
            can_close = len(stack) > 0

            if can_open and can_close:
                if random.random() < 0.5:
                    # Open a random bracket type
                    open_char, close_char = random.choice(self._brackets)
                    result.append(open_char)
                    stack.append(close_char)
                    opens_remaining -= 1
                else:
                    # Close the most recent
                    result.append(stack.pop())
                    closes_remaining -= 1
            elif can_open:
                open_char, close_char = random.choice(self._brackets)
                result.append(open_char)
                stack.append(close_char)
                opens_remaining -= 1
            else:
                result.append(stack.pop())
                closes_remaining -= 1

        return "".join(result)

    def validate(self, s: str) -> bool:
        """Check if string has balanced brackets of both types."""
        return self.validate_detailed(s).is_valid

    def validate_detailed(self, s: str) -> ValidationResult:
        """Validate with detailed error information."""
        if not s:
            return ValidationResult(False, "Empty string", 0)

        # Check for invalid characters
        for i, c in enumerate(s):
            if c not in self.vocab:
                return ValidationResult(False, f"Invalid character '{c}'", i)

        matching = {"(": ")", "[": "]"}
        opening = set(matching.keys())
        stack = []

        for i, c in enumerate(s):
            if c in opening:
                stack.append((c, i))
            else:
                if not stack:
                    return ValidationResult(False, "Unmatched closing bracket", i)
                open_char, open_pos = stack.pop()
                if matching[open_char] != c:
                    return ValidationResult(
                        False,
                        f"Mismatched brackets: '{open_char}' at {open_pos} closed by '{c}'",
                        i,
                    )

        if stack:
            _, pos = stack[-1]
            return ValidationResult(
                False, f"Unmatched opening bracket(s): {len(stack)} unclosed", pos
            )

        return ValidationResult(True)
